# predict_digit.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('enhanced_digit_model.h5')

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 pixels (the size expected by the model)
    img = cv2.resize(img, (28, 28))

    # Normalize pixel values to be between 0 and 1
    img = img / 255.0

    # Reshape the image to match the model's expected shape
    img = np.reshape(img, (1, 28, 28, 1))

    return img

def predict_digit(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Make a prediction
    predictions = model.predict(img)
    digit_class = np.argmax(predictions)

    return digit_class

def display_result(image_path, predicted_digit):
    # Display the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.axis('off')
    plt.show()

# Example: Replace 'your_image_path.png' with the path to your own image file
image_path = 'your_image_path.png'
predicted_digit = predict_digit(image_path)
display_result(image_path, predicted_digit)
