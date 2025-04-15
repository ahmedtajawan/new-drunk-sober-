import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
import os

st.set_page_config(page_title="Audio Upload & Record", layout="centered")
st.title("🎧 Audio Input")

option = st.sidebar.radio("Choose Audio Input Method", ("Upload Audio File", "Record Audio"))
temp_path = None

def split_audio(file_path, chunk_duration_sec=10):
    data, samplerate = sf.read(file_path)
    chunk_size = chunk_duration_sec * samplerate
    total_chunks = int(np.ceil(len(data) / chunk_size))
    chunks = []

    for i in range(total_chunks):
        start = int(i * chunk_size)
        end = int(min((i + 1) * chunk_size, len(data)))
        chunk_data = data[start:end]
        chunk_path = f"{file_path}_chunk_{i}.wav"
        sf.write(chunk_path, chunk_data, samplerate)
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
