import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import sounddevice as sd
import wavio

# Load trained CNN model
model = tf.keras.models.load_model("cnn_model.keras")

# Path to dataset
AUDIO_DIR = r"C:\Users\asmet\OneDrive\Desktop\AI-Driven-Speaker-Identification-master\16000_pcm_speeches"

# Speaker labels
speaker_labels = {0: "Benjamin Netanyau", 1: "Jens Stoltenberg", 2: "Julia Gillard", 3: "Margaret Thatcher", 4: "Nelson Mandela"}

# Function to extract MFCC features
def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc, axis=1)  # Take mean across time axis
        return mfcc, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# Function to predict speaker
def predict_speaker(audio_path):
    mfcc_features, error = extract_mfcc(audio_path)
    if mfcc_features is None:
        return error
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # Add channel dimension if required
    prediction = model.predict(mfcc_features)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    return f"Predicted Speaker: {speaker_labels.get(predicted_label, 'Unknown')} (Confidence: {confidence:.2f})"

# Function to record audio
def record_audio(filename="recorded_audio.wav", duration=3, samplerate=16000):
    st.info("Recording... Speak now!")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wavio.write(filename, recording, samplerate, sampwidth=2)
    return filename

# Streamlit UI
st.set_page_config(page_title="🎙️ Speaker Recognition", page_icon="🎙️", layout="centered")

# Title and Subtitle
st.markdown("<h1 style='text-align: center; color: black;'>🎙️ Speaker Recognition System</h1>", unsafe_allow_html=True)

# Custom CSS for spacing, button styles, and file uploader width
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50 !important;  
        color: white !important;
        font-size: 18px !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        width: auto !important;
        display: flex;
        justify-content: center;
        margin-left: 180px;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
    }
    .stFileUploader>div>div>input {
        width: 500% !important;  /* Make file uploader take full width */
        justify-content: center;
    }
    .stFileUploader {
        width: 150% !important;  /* You can adjust this value as needed */
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Center elements using columns
col1, col2, col3 = st.columns([0.5, 2.5, 1.5])

with col2:
    # File Upload
    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

    # Add space
    st.markdown("<br>", unsafe_allow_html=True)

    # Record Audio Button (Centered)
    if st.button("🎤 Record Audio"):
        audio_path = record_audio()
        st.success("Recording saved!")

    # Add more space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Instruction text
    st.markdown('<p style="text-align: right; margin-left: 100px; font-size: 15px; color: red;"><b>Press "Predict" to identify the speaker</b></p>', unsafe_allow_html=True)

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # Predict Speaker Button (Centered)
    predict_button = st.button("Predict Speaker")

# Process the uploaded or recorded audio after prediction button is clicked
if predict_button:
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())  # Save the file
        result = predict_speaker("temp_audio.wav")
        st.success(result)
    elif os.path.exists("recorded_audio.wav"):
        st.audio("recorded_audio.wav", format="audio/wav")
        result = predict_speaker("recorded_audio.wav")
        st.success(result)