import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
MODEL_PATH = "cnn_cifar10_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# CIFAR-10 classes
classes = [
    "Airplane", "Automobile", "Bird", "Cat", 
    "Deer", "Dog", "Frog", "Horse", 
    "Ship", "Truck"
]

# Streamlit app
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image, and this app will classify it into one of the 10 CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Resize and normalize the image
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Display result
    st.write(f"Prediction: **{classes[predicted_class]}**")
    st.bar_chart(predictions[0])
