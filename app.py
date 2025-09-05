import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import torch
import torchvision.transforms as transforms
from transformers import DPTImageProcessor, DPTForDepthEstimation
import google.generativeai as genai
from moviepy.editor import VideoFileClip, ImageSequenceClip
import io
import time

# Configure the page
st.set_page_config(
    page_title="VR 180 Converter",
    page_icon="🥽",
    layout="wide"
)

# Initialize session state
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

# Configure Gemini API
def setup_gemini():
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = None
    
    api_key = st.sidebar.text_input(
        "Enter Google Gemini API Key:",
        type="password",
        value=st.session_state.gemini_api_key or "",
        help="This key will be used for AI-powered depth analysis and optimization"
    )
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        genai.configure(api_key=api_key)
        return True
    return False

# Load depth estimation model
@st.cache_resource
def load_depth_model():
    try:
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        return processor, model
    except Exception as e:
        st.error(f"Error loading depth model: {str(e)}")
        return None, None

# Extract frames from video
def extract_frames(video_path, max_frames=100):
    """Extract frames from video for processing"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip to limit processing
    frame_skip = max(1, total_frames // max_frames)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames, fps

# Generate depth map for a single frame
def generate_depth_map(frame, processor, model):
    """Generate depth map using DPT model"""
    try:
        # Convert frame to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Process image
        inputs = processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Convert to numpy and normalize
        depth = predicted_depth.squeeze().cpu().numpy()
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth_normalized
    except Exception as e:
        st.error(f"Error generating depth map: {str(e)}")
        return None

# Create stereoscopic frame from depth
def create_stereoscopic_frame(frame, depth_map, baseline=0.1):
    """Create left and right eye views for VR 180"""
    height, width = frame.shape[:2]

    # --- THE FIX ---
    # The original depth_map has a fixed size (e.g., 384x384) from the AI model.
    # We MUST resize it to match the video frame's dimensions (e.g., 640x360).
    depth_map_resized = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

    # Now, use the CORRECTED `depth_map_resized` variable for all calculations.
    max_disparity = int(width * baseline)
    disparity = (depth_map_resized * max_disparity).astype(np.int32) # <<< Ensure you are using depth_map_resized here!
    
    # The rest of the function remains the same, but it's now using correctly-sized data.
    left_img = frame.copy()
    right_img = np.zeros_like(frame)
    
    for y in range(height):
        for x in range(width):
            shift = disparity[y, x]
            new_x = min(width - 1, max(0, x - shift))
            right_img[y, new_x] = frame[y, x]
    
    mask = np.all(right_img == 0, axis=2)
    right_img[mask] = left_img[mask]
    
    stereo_frame = np.hstack([left_img, right_img])
    
    return stereo_frame

# Process video with Gemini insights
def get_gemini_insights(frame_sample):
    """Get AI insights about the video content for better depth processing"""
    if st.session_state.gemini_api_key:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Convert frame to PIL Image for Gemini
            pil_image = Image.fromarray(frame_sample)
            
            prompt = """
            Analyze this video frame for VR 180 conversion. Identify:
            1. Main subjects and their depth layers (foreground, middle, background)
            2. Camera movement type (static, pan, zoom, etc.)
            3. Scene complexity (simple, moderate, complex)
            4. Recommended depth baseline (0.05-0.2 range)
            
            Provide a brief analysis in 2-3 sentences focusing on depth optimization.
            """
            
            response = model.generate_content([prompt, pil_image])
            return response.text
        except Exception as e:
            return f"Gemini analysis unavailable: {str(e)}"
    return "Gemini API key not provided. Using default depth processing."

# Main processing function
def process_video_to_vr180(video_file):
    """Main function to convert 2D video to VR 180"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_video_path = tmp_file.name
    
    try:
        st.session_state.processing_status = "Extracting frames from video..."
        
        # Extract frames
        frames, fps = extract_frames(temp_video_path)
        
        if not frames:
            st.error("Could not extract frames from video")
            return None
        
        st.session_state.processing_status = f"Extracted {len(frames)} frames. Loading AI models..."
        
        # Load depth estimation model
        processor, model = load_depth_model()
        
        if processor is None or model is None:
            st.error("Could not load depth estimation model")
            return None
        
        # Get Gemini insights from first frame
        st.session_state.processing_status = "Getting AI insights for optimal depth processing..."
        gemini_insights = get_gemini_insights(frames[0])
        
        st.info(f"AI Analysis: {gemini_insights}")
        
        # Process frames
        st.session_state.processing_status = "Processing frames for VR 180 conversion..."
        
        stereo_frames = []
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        for i, frame in enumerate(frames):
            try:
                status_placeholder.text(f"Processing frame {i+1}/{len(frames)}...")
                
                # Generate depth map
                depth_map = generate_depth_map(frame, processor, model)
                
                if depth_map is not None:
                    # Create stereoscopic frame
                    stereo_frame = create_stereoscopic_frame(frame, depth_map)
                    
                    if stereo_frame is not None:
                        stereo_frames.append(stereo_frame)
                        status_placeholder.text(f"✅ Frame {i+1}/{len(frames)} processed successfully")
                    else:
                        status_placeholder.text(f"⚠️ Failed to create stereo frame {i+1}")
                else:
                    status_placeholder.text(f"⚠️ Failed to generate depth map for frame {i+1}")
                
            except Exception as e:
                status_placeholder.text(f"❌ Error processing frame {i+1}: {str(e)}")
                # Continue with next frame instead of stopping
                continue
            
            # Update progress
            progress_bar.progress((i + 1) / len(frames))
            
            # Small delay to show progress
            time.sleep(0.1)
        
        if not stereo_frames:
            st.error("No frames could be processed")
            return None
        
        st.session_state.processing_status = "Creating VR 180 video..."
        
        # Create video from processed frames
        output_path = tempfile.mktemp(suffix='_vr180.mp4')
        
        # Use moviepy to create video
        clip = ImageSequenceClip(stereo_frames, fps=fps/2)  # Reduce fps due to processing
        clip.write_videofile(output_path, codec='libx264', verbose=False)
        
        st.session_state.processing_status = "VR 180 video created successfully!"
        
        return output_path
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

# Main UI
def main():
    st.title("🥽 VR 180 Video Converter")
    st.markdown("Convert your 2D videos into immersive VR 180 experiences using AI-powered depth estimation")
    
    # Setup Gemini API
    has_gemini = setup_gemini()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload Your Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your 2D video to convert to VR 180 format"
        )
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            # Processing options
            st.subheader("Conversion Settings")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                depth_strength = st.slider(
                    "Depth Effect Strength",
                    min_value=0.05,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                    help="Higher values create more pronounced 3D effect"
                )
            
            with col_b:
                quality_setting = st.selectbox(
                    "Processing Quality",
                    ["Fast (Low Quality)", "Balanced", "High Quality (Slow)"],
                    index=1
                )
            
            # Convert button
            if st.button("🚀 Convert to VR 180", type="primary"):
                if not has_gemini:
                    st.warning("For best results, please provide a Gemini API key in the sidebar")
                
                with st.spinner("Processing video..."):
                    result_path = process_video_to_vr180(uploaded_file)
                    
                    if result_path:
                        st.session_state.processed_video_path = result_path
                        st.success("✅ Conversion completed!")
                        
                        # Show download button
                        with open(result_path, 'rb') as f:
                            st.download_button(
                                "📥 Download VR 180 Video",
                                data=f.read(),
                                file_name="converted_vr180.mp4",
                                mime="video/mp4"
                            )
                        
                        st.info("Your VR 180 video is ready! You can now view it in any VR headset that supports side-by-side 180° content.")
    
    with col2:
        st.header("How It Works")
        
        with st.expander("🔍 AI Depth Analysis"):
            st.write("""
            Our platform uses advanced AI models to:
            - Analyze each frame for depth information
            - Identify foreground and background elements
            - Generate accurate depth maps
            """)
        
        with st.expander("🎬 VR 180 Creation"):
            st.write("""
            The conversion process:
            1. Extracts frames from your video
            2. Generates depth maps using AI
            3. Creates left/right eye views
            4. Combines into side-by-side VR format
            """)
        
        with st.expander("📱 VR Compatibility"):
            st.write("""
            Compatible with:
            - Meta Quest/Quest 2/Quest 3
            - PICO VR headsets
            - Google Cardboard
            - Any VR player supporting SBS 180°
            """)
        
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Tips for best results:**
    - Use videos with clear depth layers (foreground/background)
    - Avoid videos with extreme camera movements
    - Shorter clips (30-60 seconds) process faster
    - Ensure good lighting in your source video
    """)

if __name__ == "__main__":
    main()
