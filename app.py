import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
import os
import joblib
import librosa
import pandas as pd
import parselmouth
from parselmouth.praat import call
import wave, contextlib, math
from pydub import AudioSegment
import io


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

# Load new model and scaler (add below your existing model)
@st.cache_resource
def load_new_model():
    new_model = joblib.load("final_ensemble_model_pret_numpy.pkl")
    new_scaler = joblib.load("scaler_pret_numpy.pkl")
    return new_model, new_scaler

new_model, new_scaler = load_new_model()

def format_verdict_label(label, confidence, was_tie):
    """Format verdict with color, emoji, confidence, and optional tie warning."""
    if label == "DRUNK":
        emoji = "🔴"
        color = "red"
    else:
        emoji = "🟢"
        color = "green"

    text = f"<span style='color:{color}; font-weight:bold; font-size:24px'>{emoji} {label}</span><br>"
    text += f"<span style='font-size:16px'>Confidence: {confidence * 100:.2f}%</span>"

    if was_tie:
        text += "<br><span style='color:orange; font-weight:bold'>⚠️ Tie detected – confidence based on probabilities</span>"

    return text


    return text


def save_features_record(audio_name, threshold_feats, new_feats, final_label, confidence):
    """Append extracted features + prediction to a CSV log inside Streamlit app folder."""
    log_path = "all_uploaded_features.csv"

    combined_feats = {
        "audio_name": audio_name,
        "final_prediction": final_label,
        "confidence": round(confidence, 4),
    }
    # merge both dicts
    combined_feats.update(threshold_feats)
    combined_feats.update(new_feats.iloc[0].to_dict())

    df_new = pd.DataFrame([combined_feats])

    if os.path.exists(log_path):
        df_existing = pd.read_csv(log_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(log_path, index=False)
    return log_path


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


def extract_13_features(file_path):
    """Extract 13 new features for future ML model"""
    import librosa
    import numpy as np
    import parselmouth
    from parselmouth.praat import call

    y, sr = librosa.load(file_path, sr=16000)
    y = y - np.mean(y)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.normalize(y)

    # --- Spectral / RMS / Flatness / Bandwidth ---
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low_band = (freqs >= 300) & (freqs <= 3000)
    high_band = (freqs > 3000)
    low_energy = np.mean(S[low_band]) if np.any(low_band) else np.nan
    high_energy = np.mean(S[high_band]) if np.any(high_band) else np.nan
    highfreq_ratio = high_energy / (low_energy + 1e-6)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # --- Pause / Nucleus / Articulation ---
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    total = len(energy)
    low_thr = np.percentile(energy, 10)
    high_thr = np.percentile(energy, 75)
    pause_percent = float(np.sum(energy < low_thr) / total * 100)
    nucleus_percent = float(np.sum(energy > high_thr) / total * 100)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    articulation_rate = (tempo / 60) * 1.5

    # --- VSA / Formant / Jitter / Shimmer / f0 ---
    sound = parselmouth.Sound(file_path)
    formant = call(sound, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
    n_frames = int(call(formant, "Get number of frames"))
    F1, F2 = [], []
    for i in range(n_frames):
        t = call(formant, "Get time from frame number", i + 1)
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        if f1 > 0 and f2 > 0:
            F1.append(f1)
            F2.append(f2)
    vsa = (np.nanmax(F1)-np.nanmin(F1))*(np.nanmax(F2)-np.nanmin(F2)) if len(F1)>0 else np.nan

    F2 = np.array(F2)
    if len(F2) < 3 or np.all(np.isnan(F2)):
        F2_MAS, F2_JumpRate = np.nan, np.nan
    else:
        diffs = np.abs(np.diff(F2))
        F2_MAS = float(np.nanmean(diffs))
        F2_JumpRate = float(np.mean(diffs>100)*100)

    # PointProcess for jitter/shimmer/f0
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    try:
        jitter = float(call(point_process, "Get jitter (local)",0,0,0.0001,0.02,1.3))
        shimmer = float(call([sound, point_process], "Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6))
    except:
        jitter = shimmer = np.nan
    mean_period = call(point_process,"Get mean period",0,0,0.0001,0.02,1.3)
    f0 = float(1/mean_period if mean_period and mean_period>0 else np.nan)

    features = {
        "highfreq_ratio": highfreq_ratio,
        "flatness": flatness,
        "bandwidth": bandwidth,
        "rms": rms,
        "pause_percent": pause_percent,
        "nucleus_percent": nucleus_percent,
        "articulation_rate": articulation_rate,
        "vsa": vsa,
        "F2_MAS": F2_MAS,
        "F2_JumpRate": F2_JumpRate,
        "jitter": jitter,
        "shimmer": shimmer,
        "f0": f0
    }

    return pd.DataFrame([features])

def extract_threshold_features(file_path):
    """Extract the 5 threshold features for a single audio file."""
    
    # --- VSA ---
    try:
        sound = parselmouth.Sound(file_path)
        formant = call(sound, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
        n_frames = int(call(formant, "Get number of frames"))
        F1, F2 = [], []
        for i in range(n_frames):
            t = call(formant, "Get time from frame number", i + 1)
            f1 = call(formant, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formant, "Get value at time", 2, t, 'Hertz', 'Linear')
            if f1 > 0 and f2 > 0:
                F1.append(f1)
                F2.append(f2)
        vsa = (np.nanmax(F1)-np.nanmin(F1)) * (np.nanmax(F2)-np.nanmin(F2)) if F1 and F2 else np.nan
    except:
        vsa = np.nan

    # --- Load audio for RMS, flatness, bandwidth ---
    with contextlib.closing(wave.open(file_path,'rb')) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
    audio = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    audio = audio / (np.max(np.abs(audio)) + 1e-12)

    frame_len, hop = int(0.03*sr), int(0.01*sr)
    window = np.hamming(frame_len)

    rms_list, flatness_list, bandwidth_list = [], [], []

    for start in range(0, len(audio)-frame_len+1, hop):
        frame = audio[start:start+frame_len]
        rms = math.sqrt(np.mean(frame**2)+1e-12)
        rms_list.append(rms)
        fft = np.fft.rfft(frame*window)
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(frame), d=1.0/sr)
        mag_sum = mag.sum()+1e-12
        centroid = (freqs*mag).sum()/mag_sum
        bw = np.sqrt(((freqs-centroid)**2*mag).sum()/mag_sum)
        geom_mean = np.exp(np.log(mag+1e-12).mean())
        flat = geom_mean/(mag.mean()+1e-12)
        bandwidth_list.append(bw)
        flatness_list.append(flat)

    mean_rms = float(np.mean(rms_list))
    std_rms = float(np.std(rms_list))
    mean_flatness = float(np.mean(flatness_list))
    bandwidth = float(np.mean(bandwidth_list))

    return {
        "vsa": vsa,
        "bandwidth": bandwidth,
        "mean_flatness": mean_flatness,
        "mean_rms": mean_rms,
        "std_rms": std_rms
    }


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
import numpy as np

def predict_from_chunks(chunk_paths):
    """Predict drunk/sober using the old model on each chunk, with average probability and tie handling."""
    preds = []
    drunk_index = list(le.classes_).index("drunk")
    sober_index = list(le.classes_).index("sober")
    
    # accumulate per-chunk probabilities
    drunk_probs = []
    sober_probs = []

    for path in chunk_paths:
        features = extract_all_features(path)
        prob = model.predict_proba(features)[0]
        pred = model.predict(features)[0]
        label = le.inverse_transform([pred])[0]
        preds.append(label)

        drunk_probs.append(prob[drunk_index])
        sober_probs.append(prob[sober_index])

    # average probabilities
    avg_drunk_prob = np.mean(drunk_probs)
    avg_sober_prob = np.mean(sober_probs)

    # tie handling: consider as tie if probabilities are close
    diff = abs(avg_drunk_prob - avg_sober_prob)
    tie_threshold = 0.05  # you can adjust this
    was_tie = diff <= tie_threshold

    # determine final label
    final = "DRUNK" if avg_drunk_prob > avg_sober_prob else "SOBER"
    confidence = max(avg_drunk_prob, avg_sober_prob)

    # if tie, reduce confidence slightly to reflect uncertainty
    if was_tie:
        confidence *= 0.9

    return final, confidence, preds, was_tie


def predict_from_chunks_new_model(chunk_paths):
    """Predict drunk/sober using the new ensemble model on each chunk, with average probability and tie handling."""
    preds = []
    drunk_probs = []
    sober_probs = []

    for path in chunk_paths:
        threshold_feats = extract_threshold_features(path)
        new_feats = extract_13_features(path)
        combined_feats = {**threshold_feats, **new_feats.iloc[0].to_dict()}
        df_combined = pd.DataFrame([combined_feats])

        # Reindex columns to match scaler and fill missing values
        expected_cols = new_scaler.feature_names_in_
        df_combined = df_combined.reindex(columns=expected_cols, fill_value=0).fillna(0)
        
        # Scale and predict
        X_scaled = new_scaler.transform(df_combined)
        prob = new_model.predict_proba(X_scaled)[0]
        pred = new_model.predict(X_scaled)[0]

        label = "DRUNK" if pred == 1 else "SOBER"
        preds.append(label)

        drunk_probs.append(prob[1])
        sober_probs.append(prob[0])

    # average probabilities
    avg_drunk_prob = np.mean(drunk_probs)
    avg_sober_prob = np.mean(sober_probs)

    # tie handling
    diff = abs(avg_drunk_prob - avg_sober_prob)
    tie_threshold = 0.05
    was_tie = diff <= tie_threshold

    final = "DRUNK" if avg_drunk_prob > avg_sober_prob else "SOBER"
    confidence = max(avg_drunk_prob, avg_sober_prob)
    if was_tie:
        confidence *= 0.9

    return final, confidence, preds, was_tie



def predict_drunk_sober_threshold(audio_file_path, chunk_duration=10, sr=16000):
    """
    Chunk-based threshold drunk/sober predictor.
    Uses user's thresholds for: vsa, mean_flatness, bandwidth, mean_rms, std_rms.
    """

    # ---- USER DEFINED THRESHOLDS ----
    thresholds = {
        'vsa': 5.99e6,
        'mean_flatness': 0.449,
        'bandwidth': 1849,
        'mean_rms': 0.0579,
        'std_rms': 0.0512
    }
    # ---------------------------------

    # Load audio
    y, sr = librosa.load(audio_file_path, sr=sr)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Chunk parameters
    chunk_size = int(chunk_duration * sr)
    num_chunks = int(np.ceil(len(y) / chunk_size))

    chunk_predictions = []
    all_features = []

    # Process each chunk
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(y))
        chunk = y[start:end]

        if len(chunk) < sr * 0.5:  # Skip very short chunks (<0.5s)
            continue

        # --- FEATURE EXTRACTION ---
        S = np.abs(librosa.stft(chunk))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(S=S))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sr))
        rms = librosa.feature.rms(y=chunk)[0]
        mean_rms = np.mean(rms)
        std_rms = np.std(rms)

        # VSA (Voice Segment Area) approximation = sum of RMS * bandwidth
        vsa = np.sum(rms) * bandwidth

        features = {
            'vsa': vsa,
            'mean_flatness': spectral_flatness,
            'bandwidth': bandwidth,
            'mean_rms': mean_rms,
            'std_rms': std_rms
        }
        all_features.append(features)

        # --- THRESHOLD LOGIC ---
        drunk_score = 0
        drunk_score += 1 if vsa < thresholds['vsa'] else 0
        drunk_score += 1 if spectral_flatness > thresholds['mean_flatness'] else 0
        drunk_score += 1 if bandwidth < thresholds['bandwidth'] else 0
        drunk_score += 1 if mean_rms < thresholds['mean_rms'] else 0
        drunk_score += 1 if std_rms > thresholds['std_rms'] else 0

        label = "Drunk" if drunk_score >= 3 else "Sober"
        chunk_predictions.append(label)

    # --- FINAL DECISION ---
    drunk_ratio = chunk_predictions.count("Drunk") / len(chunk_predictions)
    final_label = "Drunk" if drunk_ratio >= 0.5 else "Sober"

    return {
        "final_label": final_label,
        "chunk_predictions": chunk_predictions,
        "drunk_ratio": drunk_ratio,
        "features": all_features,
        "total_chunks": len(chunk_predictions),
        "thresholds_used": thresholds
    }


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
      # --- Run the new ensemble model (chunk-based) ---
        with st.spinner("🧠 Running ensemble model on chunks..."):
            new_final, new_confidence, new_preds, new_tie = predict_from_chunks_new_model(result["chunks"])
        
        st.markdown("### 🧠 Ensemble Model (Chunk-based) Prediction")
        st.markdown(format_verdict_label(new_final, new_confidence, new_tie), unsafe_allow_html=True)
        
                 # --- Run threshold-based rule system (chunk-based) ---
        with st.spinner("📏 Running threshold-based system on chunks..."):
            th_final, th_conf, th_preds, th_tie = predict_drunk_sober_threshold(result["chunks"])
        
        st.markdown("### 📏 Threshold-Based System (Chunk-based)")
        st.markdown(format_verdict_label(th_final, th_conf, was_tie=th_tie), unsafe_allow_html=True)
        
        # Show quick chunk summary
        st.write(f"🧩 Chunks analyzed: {len(th_preds)}")
        st.write(f"🔴 Drunk chunks: {th_preds.count('DRUNK')} | 🟢 Sober chunks: {th_preds.count('SOBER')}")

            # --- Show threshold features ---
        threshold_feats = extract_threshold_features(temp_path)
        st.subheader("🧪 Threshold Features")
        st.write(pd.DataFrame([threshold_feats]).T.rename(columns={0:"Value"}))
        # --- Show new 13 features ---
        new_feats = extract_13_features(temp_path)
        st.subheader("🧩 New 13 Features")
        st.write(new_feats.T.rename(columns={0:"Value"}))
    
        
        
        # --- Save all extracted features + prediction result ---
        audio_name = os.path.basename(temp_path)
        csv_path = save_features_record(audio_name, threshold_feats, new_feats, final, confidence)
        st.success(f"✅ Features for '{audio_name}' appended to `{csv_path}`")
        
        # --- Optional: Show current record in sidebar ---
        if os.path.exists(csv_path):
            st.sidebar.subheader("📊 Feature Log")
            df_log = pd.read_csv(csv_path)
            st.sidebar.write(f"Total Records: {len(df_log)}")
            if st.sidebar.button("📥 View Log"):
                st.write("### 🧾 All Saved Audio Features")
                st.dataframe(df_log)
            st.sidebar.download_button(
                "⬇️ Download CSV",
                data=open(csv_path, "rb"),
                file_name="all_uploaded_features.csv",
                mime="text/csv"
            )


# Upload or record
if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
   
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            file_bytes = uploaded_file.read()
    
            if uploaded_file.name.lower().endswith(".mp3"):
                audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
                audio.export(f.name, format="wav")
            else:
                f.write(file_bytes)
    
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
