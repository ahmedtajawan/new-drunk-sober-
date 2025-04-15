import streamlit as st
import numpy as np
from audiorecorder import audiorecorder
import tempfile

st.set_page_config(
    page_title="🎧 Drunk or Sober Audio Classifier",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: white;
        }
        .stButton button {
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stFileUploader, .st-audio {
            background-color: #1E1E1E;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Drunk or Sober Audio Classifier")
st.write("🎙️ Upload or record an audio file and let the AI analyze it!")

st.subheader("📂 Upload Audio File")
uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

st.subheader("🎤 Or Record Your Voice")
audio = audiorecorder("Click to Record", "Recording...")

# -- Display Uploaded Audio --
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav")
    st.success("✅ Audio file uploaded successfully.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
        st.info(f"📁 Saved uploaded audio at: `{audio_path}`")

# -- Display Recorded Audio --
elif len(audio) > 0:
    st.audio(audio.tobytes(), format="audio/wav")
    st.success("✅ Audio recorded successfully.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.tobytes())
        audio_path = tmp.name
        st.info(f"📁 Saved recorded audio at: `{audio_path}`")

# 🔮 Placeholder for prediction logic
if st.button("🎯 Analyze"):
    if 'audio_path' in locals():
        st.warning("🔧 Model prediction logic not added yet. Ready for integration.")
    else:
        st.error("❌ Please upload or record an audio file first.")
