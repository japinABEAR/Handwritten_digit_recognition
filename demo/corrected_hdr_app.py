
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import tkinter as tk
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Set environment variable to turn off oneDNN optimizations if required
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data: normalize and reshape
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Split the data for training the SVM
X_train_features, X_valid_features, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and train the SVM classifier
svm_classifier = svm.SVC(kernel='linear')

# Train the classifier
svm_classifier.fit(X_train_features, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM Classifier: {accuracy:.4f}")

# CNN Model (if you need to compare results)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape data for CNN and one-hot encode labels
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
y_train_cnn = to_categorical(y_train, 10)
y_test_cnn = to_categorical(y_test, 10)

# Train the CNN model
model.fit(X_train_cnn, y_train_cnn, epochs=5, validation_data=(X_test_cnn, y_test_cnn))

# Evaluate CNN model
cnn_accuracy = model.evaluate(X_test_cnn, y_test_cnn)[1]
print(f"Accuracy of CNN Model: {cnn_accuracy:.4f}")
