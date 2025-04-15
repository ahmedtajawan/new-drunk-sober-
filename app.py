import streamlit as st
import tempfile
from pydub import AudioSegment
import os

st.set_page_config(page_title="Audio Upload & Record", layout="centered")
st.title("🎧 Audio Input")

option = st.sidebar.radio("Choose Audio Input Method", ("Upload Audio File", "Record Audio"))
temp_path = None

def split_audio(file_path, chunk_duration_sec=10):
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    chunk_length = chunk_duration_sec * 1000
    chunks = []

    for i in range(0, duration_ms, chunk_length):
        chunk = audio[i:i + chunk_length]
        chunk_path = f"{file_path}_chunk_{i // chunk_length}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        st.audio(temp_path)

        chunk_paths = split_audio(temp_path)
        st.info(f"✅ File split into {len(chunk_paths)} chunks.")

elif option == "Record Audio":
    st.write("🎙️ Record Audio Below")
    audio_file = st.audio_input("Record")
    if audio_file:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())
        st.audio(temp_path)

        chunk_paths = split_audio(temp_path)
        st.info(f"✅ File split into {len(chunk_paths)} chunks.")
