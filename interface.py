import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('handwritten_digits_model.h5')

# Create the UI window
window = tk.Tk()
window.title("Handwritten Digit Recognition")
window.geometry("400x400")

# Create a canvas to draw on
canvas = tk.Canvas(window, width=200, height=200, bg="white")
canvas.pack()

# Create a label to display the prediction
label = tk.Label(window, text="Draw a digit", font=("Helvetica", 20))
label.pack()

# Function to recognize the digit
def recognize_digit():
    # Convert the canvas drawing to a numpy array
    image = canvas.postscript(colormode='gray')
    image = Image.open(io.BytesIO(image.encode('utf-8')))
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1) / 255.0

    # Make the prediction
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    # Update the label with the predicted digit
    label.config(text=f"Predicted digit: {digit}")

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    label.config(text="Draw a digit")

# Function to draw on the canvas
def draw(event):
    x = event.x
    y = event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")

# Bind events to the canvas
canvas.bind("<B1-Motion>", draw)

# Create buttons
recognize_button = tk.Button(window, text="Recognize", command=recognize_digit)
recognize_button.pack()

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

# Start the UI event loop
window.mainloop()
