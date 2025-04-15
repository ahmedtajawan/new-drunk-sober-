import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
import os
import joblib
import librosa
import pandas as pd

st.set_page_config(page_title="Drunk/Sober Audio Classifier", layout="centered")
st.title("🎧 Drunk/Sober Audio Classifier")

option = st.sidebar.radio("Choose Audio Input Method", ("Upload Audio File", "Record Audio"))
temp_path = None

# Load model and encoder
@st.cache_resource
def load_model():
    model = joblib.load("final_voting_model_new.pkl")
    encoder = joblib.load("label_encoder_drunk.pkl")
    return model, encoder

model, le = load_model()

# Format colored verdict
def format_verdict_label(label, confidence, was_tie):
    if label == "DRUNK":
        emoji = "🔴"
        color = "red"
    else:
        emoji = "🟢"
        color = "green"

    text = f"<span style='color:{color}; font-weight:bold; font-size:24px'>{emoji} {label}</span><br><span style='font-size:16px'>Confidence: {confidence * 100:.2f}%</span>"

    if was_tie:
        text += "<br><span style='color:orange'>⚠️ Tie detected – confidence based on probabilities</span>"

    return text

# Feature extraction
def extract_all_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = {
        **{f"mfcc_mean_{i}": np.mean(mfcc[i]) for i in range(13)},
        **{f"mfcc_std_{i}": np.std(mfcc[i]) for i in range(13)},
        **{f"delta_mean_{i}": np.mean(delta[i]) for i in range(13)},
        **{f"delta_std_{i}": np.std(delta[i]) for i in range(13)},
        **{f"delta2_mean_{i}": np.mean(delta2[i]) for i in range(13)},
        **{f"delta2_std_{i}": np.std(delta2[i]) for i in range(13)},
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "tempo": librosa.beat.beat_track(y=y, sr=sr)[0],
    }

    return pd.DataFrame([features])

# Split into 10-sec chunks
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

# Predict on chunks
def predict_from_chunks(chunk_paths):
    preds = []
    drunk_count = 0
    sober_count = 0
    drunk_index = list(le.classes_).index("drunk")
    sober_index = list(le.classes_).index("sober")
    drunk_score_total = 0.0
    sober_score_total = 0.0

    for path in chunk_paths:
        features = extract_all_features(path)
        prob = model.predict_proba(features)[0]
        pred = model.predict(features)[0]
        label = le.inverse_transform([pred])[0]
        preds.append(label)

        if label == "drunk":
            drunk_count += 1
        else:
            sober_count += 1

        drunk_score_total += prob[drunk_index]
        sober_score_total += prob[sober_index]

    if drunk_count > sober_count:
        final = "DRUNK"
        confidence = drunk_count / len(preds)
        was_tie = False
    elif sober_count > drunk_count:
        final = "SOBER"
        confidence = sober_count / len(preds)
        was_tie = False
    else:
        was_tie = True
        if drunk_score_total > sober_score_total:
            final = "DRUNK"
            confidence = drunk_score_total / (drunk_score_total + sober_score_total)
        else:
            final = "SOBER"
            confidence = sober_score_total / (drunk_score_total + sober_score_total)

    return final, confidence, preds, was_tie

# Handle full flow
def handle_audio(temp_path):
    st.audio(temp_path)

    result = split_audio(temp_path)
    if result["status"] == "short":
        st.warning(f"⚠️ Audio too short: {result['duration']} seconds. Minimum 10 seconds required.")
    else:
        st.info(
            f"✅ Audio duration: {result['duration']} sec\n"
        #    f"🧩 Chunks created: {result['chunk_count']}\n"
         #   f"🗑️ Last chunk discarded: {'Yes' if result['last_discarded'] else 'No'}"
        )

        with st.spinner("🔍 Analyzing audio... This might take a few seconds."):
            final, confidence, all_preds, was_tie = predict_from_chunks(result["chunks"])

        st.markdown(format_verdict_label(final, confidence, was_tie), unsafe_allow_html=True)
       # st.markdown(f"🎯 Chunk-wise Prediction: `{all_preds}`")

# Upload or record
if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        handle_audio(temp_path)

elif option == "Record Audio":
    st.write("🎙️ Record Audio Below")
    audio_file = st.audio_input("Record")
    if audio_file:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())
        handle_audio(temp_path)
