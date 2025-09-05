# Hackathon-VR-180-Immersive-Experience

# VR 180 Video Converter - Deployment Guide

## Overview
This platform converts 2D video clips into immersive VR 180 experiences using AI-powered depth estimation. The application uses advanced machine learning models to analyze video content and generate stereoscopic 3D effects.

## Features
- **Simple Upload Interface**: Drag-and-drop video upload
- **AI-Powered Depth Analysis**: Uses Intel DPT models for accurate depth estimation  
- **Gemini AI Integration**: Enhanced processing with Google's Gemini API
- **Real-time Progress Tracking**: Visual feedback during conversion
- **VR-Ready Output**: Side-by-side 180° format compatible with all major VR headsets
- **Customizable Settings**: Adjustable depth strength and processing quality

## Technical Architecture

### Core Components
1. **Depth Estimation**: Intel DPT (Dense Prediction Transformer) model
2. **Video Processing**: OpenCV and MoviePy for frame extraction and video creation
3. **AI Enhancement**: Google Gemini API for content analysis and optimization
4. **Stereoscopic Rendering**: Custom algorithm for creating left/right eye views

### Processing Pipeline
1. **Frame Extraction**: Extract frames from uploaded 2D video
2. **AI Analysis**: Gemini analyzes content for optimal depth processing
3. **Depth Map Generation**: DPT model creates depth information for each frame
4. **Stereoscopic Creation**: Generate left/right eye views based on depth data
5. **VR Video Assembly**: Combine frames into side-by-side VR 180 format

## Deployment Instructions

### Method 1: Streamlit Cloud (Recommended)
1. Fork the repository or create a new one with the provided files
2. Add `app.py` and `requirements.txt` to your repository
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repository
5. Deploy the app

### Method 2: Local Deployment
```bash
# Clone repository
git clone <your-repo-url>
cd vr180-converter

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Method 3: Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## API Configuration

### Google Gemini API Setup
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Enter the key in the sidebar of the application
4. The API enhances depth analysis and provides optimization recommendations

## Usage Instructions

### For End Users
1. **Upload Video**: Click "Choose a video file" and select your 2D clip
2. **Configure Settings**: 
   - Adjust depth effect strength (0.05-0.3)
   - Select processing quality (Fast/Balanced/High Quality)
3. **Add API Key** (Optional): Enter Gemini API key for enhanced processing
4. **Convert**: Click "Convert to VR 180" button
5. **Download**: Once processing completes, download your VR 180 video

### VR Viewing Instructions
1. Transfer the downloaded video to your VR headset
2. Open a VR video player that supports side-by-side 180° content
3. Select "Side-by-Side 180°" or "SBS 180°" viewing mode
4. Enjoy your immersive VR experience!

## Supported Formats

### Input Formats
- MP4, AVI, MOV, MKV
- Resolution: Up to 4K (processing time varies)
- Duration: Optimized for clips under 2 minutes

### Output Format
- Side-by-side stereoscopic MP4
- VR 180° compatible
- Reduced frame rate for stability

### VR Headset Compatibility
- Meta Quest (all models)
- PICO VR headsets
- HTC Vive
- Valve Index
- Google Cardboard
- Any headset supporting SBS 180° content

## Performance Optimization

### Processing Speed
- **Fast Mode**: Lower quality, faster processing (~30 seconds per minute of video)
- **Balanced Mode**: Good quality-speed balance (~60 seconds per minute of video)  
- **High Quality Mode**: Best results, slower processing (~120 seconds per minute of video)

### Best Practices for Source Videos
- Use videos with clear depth layers (foreground/background separation)
- Avoid excessive camera movement
- Ensure good lighting conditions
- Keep clips under 2 minutes for optimal processing speed
- Higher resolution input generally produces better VR results

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Ensure sufficient system memory (4GB+ recommended)
2. **Video Upload Fails**: Check file format and size (max 200MB recommended)
3. **Processing Timeout**: Try shorter clips or Fast processing mode
4. **Poor Depth Quality**: Use Gemini API key for enhanced analysis

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- GPU support optional but recommended for faster processing
- Internet connection for model downloads and API access

## Technical Specifications

### Models Used
- **Intel DPT-Large**: Primary depth estimation model
- **Google Gemini 2.5**: Content analysis and optimization
- **Custom Stereoscopic Algorithm**: Proprietary depth-to-stereo conversion

### Processing Capabilities
- Handles videos up to 4K resolution
- Processes up to 100 frames per conversion
- Automatic frame rate optimization
- Real-time progress tracking

## Future Enhancements
- Support for batch processing
- Custom depth map editing
- Advanced VR format options (360°, volumetric)
- Real-time preview functionality
- Mobile app version

## Support
For technical support or feature requests, please contact the development team or submit issues through the project repository.
