import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV

# Set environment variable to turn off oneDNN optimizations if required
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data: normalize and reshape
X_train = X_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(10, activation='softmax'))  # 10 classes for digits 0-9
    return model

# Instantiate and compile the model
cnn_model = create_cnn_model()
cnn_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Train the CNN model
history = cnn_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Save the trained CNN model for later use
cnn_model.save('digit_recognition_cnn.h5')

# Extract features using the CNN
X_train_features = cnn_model.predict(X_train)
X_test_features = cnn_model.predict(X_test)

# Train an SVM classifier on the CNN features
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, refit=True)
grid.fit(X_train_features, y_train.argmax(axis=1))

# Predict on the test set
y_pred = grid.predict(X_test_features)

# Print SVM accuracy
print("SVM Accuracy:", accuracy_score(y_test.argmax(axis=1), y_pred))

# Plotting training history
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()

# Call the plotting function
plot_training_history(history)

# Build the GUI using Tkinter
class DigitRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognition")
        self.geometry("400x400")

        # Create a canvas for drawing
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack(pady=20)

        # Create buttons for recognition and clearing the canvas
        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button_predict.pack(pady=5)

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(pady=5)

        # Label for showing the prediction result
        self.label_result = tk.Label(self, text="Draw a digit", font=("Helvetica", 24))
        self.label_result.pack(pady=20)

        # Initialize drawing variables
        self.image = Image.new("L", (280, 280), 255)  # Create a white image
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black', outline='black')
        self.draw.ellipse([x-5, y-5, x+5, y+5], fill='black')

    def clear_canvas(self):
        # Clear the canvas and reset the image
        self.canvas.delete("all")
        self.label_result.config(text="Draw a digit")
        self.image = Image.new("L", (280, 280), 255)  # Reset the image

    def predict_digit(self):
        # Prepare the image for prediction
        img = self.image.resize((28, 28), Image.LANCZOS)  # Resize to the MNIST image size
        img = ImageOps.invert(img)  # Invert colors (black to white, white to black)
        img = np.array(img) / 255.0  # Normalize the image
        img = img.reshape(1, 28, 28, 1)  # Reshape for the model

        # Use CNN model to extract features
        features = cnn_model.predict(img)

        # Predict digit using SVM
        digit = grid.predict(features)

        # Update the result label
        self.label_result.config(text=f"Prediction: {str(digit[0])}")

# Run the application
if __name__ == "__main__":
    app = DigitRecognitionApp()
    app.mainloop()
