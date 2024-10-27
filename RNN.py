import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from mfcc_extraction import get_train_test_data  # Import the function
from sklearn.preprocessing import LabelEncoder

# Get the training and test data
X_train, X_test, y_train, y_test = get_train_test_data()

# Ensure data types are correct
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Convert labels to numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Reshape the data for RNN input
X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Adding a channel dimension
X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the RNN model
model = models.Sequential([
    layers.LSTM(128, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')  # Adjust output layer for number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model with EarlyStopping
history = model.fit(X_train_rnn, y_train_encoded, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping])

# Check if EarlyStopping triggered
if early_stopping.stopped_epoch > 0:
    print("Early stopping triggered at epoch", early_stopping.stopped_epoch + 1)
else:
    print("Training completed without early stopping")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_rnn, y_test_encoded)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training vs validation loss
plt.figure(figsize=(12, 5))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
# At the end of your RNN.py code, after fitting the label encoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Save the label classes to a .npy file
np.save("C:/Users/shiksha/OneDrive/Desktop/python/label_classes.npy", label_encoder.classes_)  # Adjust the path as necessary

