import streamlit as st
try:
    from moviepy.editor import VideoFileClip
    st.success("✅ MoviePy imported successfully!")
except Exception as e:
    st.error(f"❌ MoviePy import failed: {e}")
