import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Extracting MFCC features from an audio file
def extract_mfcc(file_path, n_mfcc=13):
    # Loading audio file
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # MFCC features extraction
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Mean of MFCC coefficients over time
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    return mfccs_mean

# Function to process the dataset and extract MFCC features for all audio files
def process_dataset(directory_path):
    mfcc_features = []
    labels = []
    
    # Loop through each speaker directory
    for speaker in os.listdir(directory_path):
        speaker_path = os.path.join(directory_path, speaker)
        
        # Ignore directories like '_background_noise_' and any other irrelevant folders
        if not os.path.isdir(speaker_path) or speaker.startswith('_'):
            continue
        
        # Loop through each audio file in the speaker's folder
        for file_name in os.listdir(speaker_path):
            if file_name.endswith('.wav'):  # Ensure it's a .wav file
                file_path = os.path.join(speaker_path, file_name)
                
                # Extract MFCC features and append to list
                mfcc = extract_mfcc(file_path)
                mfcc_features.append(mfcc)
                
                # Append corresponding label (speaker's name)
                labels.append(speaker)
    
    # Convert lists to a DataFrame
    df = pd.DataFrame(mfcc_features)
    df['label'] = labels
    
    return df

# Define the path to your dataset
dataset_path = "16000_pcm_speeches"  # Change this to your actual dataset path

# Process the dataset to extract MFCC features and labels
df = process_dataset(dataset_path)

# Split the data into train and test sets
X = df.drop('label', axis=1)  # Features (MFCCs)
y = df['label']              # Labels (Speakers)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for CNN input
X_train_cnn = X_train.values.reshape(X_train.shape[0], 13, 1)  # 13 MFCC coefficients, 1 channel
X_test_cnn = X_test.values.reshape(X_test.shape[0], 13, 1)

# Print shapes to confirm
print("Training data shape:", X_train_cnn.shape)
print("Test data shape:", X_test_cnn.shape)

# Function to return train and test data
def get_train_test_data():
    return X_train_cnn, X_test_cnn, y_train, y_test
