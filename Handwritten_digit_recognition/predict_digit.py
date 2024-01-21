# predict_digit.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt

# Load the trained model 
model = load_model('best_model.h5')

def preprocess_image(image_path):
    # Get the absolute path to the image file
    image_path = os.path.abspath(image_path)

    # Load image as grayscale
    img = cv2.imread(image_path, 0)  
    
    # Check if the image was successfully loaded
    if img is None:
        print(f"Error: Unable to load image at path: {image_path}")
        exit()

    # Resize to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    
    # Normalize pixel values
    img = img / 255.0

    # Reshape image
    img = np.reshape(img, (1, 28, 28, 1))

    return img

def predict_digit(image_path):
    # Preprocess image
    img = preprocess_image(image_path)  

    # Predict digit
    predictions = model.predict(img)
    digit_class = np.argmax(predictions)

    return digit_class

def display_result(image_path, predicted_digit):
    # Load image
    img = cv2.imread(image_path, 0)
  
    # Display image and predicted digit 
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.axis('off')
  
    # Show image
    plt.show()

# Example
image_path = 'digit2.png' #mention the name of this actual image file you are going to use!!
predicted_digit = predict_digit(image_path)
display_result(image_path, predicted_digit)
