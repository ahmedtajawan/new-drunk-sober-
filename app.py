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
    total_samples = len(data)
    duration_sec = total_samples / samplerate

    if duration_sec < chunk_duration_sec:
        return {
            "status": "short",
            "duration": round(duration_sec, 2),
            "chunks": [],
            "chunk_count": 0,
            "last_discarded": False,
        }

    chunk_size = chunk_duration_sec * samplerate
    full_chunks = total_samples // int(chunk_size)
    chunks = []

    for i in range(int(full_chunks)):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size)
        chunk_data = data[start:end]
        chunk_path = f"{file_path}_chunk_{i}.wav"
        sf.write(chunk_path, chunk_data, samplerate)
        chunks.append(chunk_path)

    last_discarded = (total_samples % int(chunk_size)) != 0

    return {
        "status": "ok",
        "duration": round(duration_sec, 2),
        "chunks": chunks,
        "chunk_count": len(chunks),
        "last_discarded": last_discarded,
    }

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        st.audio(temp_path)

        result = split_audio(temp_path)
        if result["status"] == "short":
            st.warning(f"⚠️ Audio too short: {result['duration']} seconds. Minimum 10 seconds required.")
        else:
            st.info(
                f"✅ Audio duration: {result['duration']} sec\n"
                f"🧩 Chunks created: {result['chunk_count']}\n"
                f"🗑️ Last chunk discarded: {'Yes' if result['last_discarded'] else 'No'}"
            )

elif option == "Record Audio":
    st.write("🎙️ Record Audio Below")
    audio_file = st.audio_input("Record")
    if audio_file:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())
        st.audio(temp_path)

        result = split_audio(temp_path)
        if result["status"] == "short":
            st.warning(f"⚠️ Audio too short: {result['duration']} seconds. Minimum 10 seconds required.")
        else:
            st.info(
                f"✅ Audio duration: {result['duration']} sec\n"
                f"🧩 Chunks created: {result['chunk_count']}\n"
                f"🗑️ Last chunk discarded: {'Yes' if result['last_discarded'] else 'No'}"
            )
