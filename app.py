import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os


# Load the trained model
model = joblib.load('emotion_recognition_model.pkl')

# Load the label encoder
label_encoder = LabelEncoder()
print(os.getcwd())
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

# Function to extract features from audio
def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

# Streamlit app interface
st.title("Speech Emotion Recognition System")
st.write("Upload an audio file to predict the emotion.")

# File upload widget
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Extract features and make a prediction
    with st.spinner("Extracting features and predicting emotion..."):
        features = extract_features(uploaded_file)
        prediction = model.predict(features)
        emotion = label_encoder.inverse_transform(prediction)[0]

    st.success(f"The predicted emotion is: **{emotion}**")
