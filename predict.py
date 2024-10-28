import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from mfcc_extraction import get_train_test_data
import os
import librosa

# Load the trained model
model = load_model('cnn_model.h5')

# Get the original training data to refit the LabelEncoder
_, _, y_train, _ = get_train_test_data()

# Refit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Define a threshold for confidence
CONFIDENCE_THRESHOLD = 0.7

def extract_mfcc_safe(file_path):
    try:
        # Try to load the audio file
        audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=1)  # Change to n_mfcc=1
        
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # Transpose and reshape to (time_steps, 1)
        mfcc = mfcc.T
        
        # Ensure the time dimension matches the expected input
        expected_time_steps = model.input_shape[1]
        if mfcc.shape[0] < expected_time_steps:
            mfcc = np.pad(mfcc, ((0, expected_time_steps - mfcc.shape[0]), (0, 0)), mode='constant')
        elif mfcc.shape[0] > expected_time_steps:
            mfcc = mfcc[:expected_time_steps, :]
        
        return mfcc
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

def predict_speaker(audio_path):
    # Extract MFCC features from the audio file
    mfcc_features = extract_mfcc_safe(audio_path)
    
    if mfcc_features is None:
        return "Error processing audio file", 0.0
    
    # Reshape the features to match the input shape of the model
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    
    # Make prediction
    prediction = model.predict(mfcc_features)
    
    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    
    if confidence >= CONFIDENCE_THRESHOLD:
        # Get the predicted speaker name
        predicted_speaker = label_encoder.inverse_transform([predicted_class_index])[0]
        return predicted_speaker, confidence
    else:
        return "Not matched with any known speaker", confidence

# Main execution
if __name__ == "__main__":
    # Get input audio file from user
    audio_path = input("Enter the path to the audio file: ")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print("Error: The specified audio file does not exist.")
    elif os.path.isdir(audio_path):
        print("Error: The specified path is a directory, not a file.")
    else:
        # Predict speaker
        predicted_speaker, confidence = predict_speaker(audio_path)
        
        # Print results
        print(f"Predicted speaker: {predicted_speaker}")
        print(f"Confidence: {confidence:.2f}")