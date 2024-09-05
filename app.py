import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
from io import BytesIO

# Load your trained model
model = tf.keras.models.load_model('path/to/your/model.h5')

def predict_sign(image):
    # Process the image and make prediction
    # Note: Implement the actual prediction logic here
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

def preprocess_image(image):
    # Preprocess the image according to your model requirements
    return np.expand_dims(image, axis=0)  # Example preprocessing

def play_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    return audio_bytes

# Streamlit UI
st.title("Sign Language Recognition")

menu = ["Home", "About Us", "Tutorials", "Project"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.header("Welcome to Indian Sign Language Recognition")
    st.write("This application helps you recognize Indian Sign Language signs.")
    st.write("It also provides tutorials and a project where you can test your sign recognition.")

elif choice == "About Us":
    st.header("About Us")
    st.write("This project is aimed at improving communication through Indian Sign Language.")
    st.image("static/image1.jpg", caption="Image 1")
    st.image("static/image2.jpg", caption="Image 2")
    st.image("static/image3.jpg", caption="Image 3")

elif choice == "Tutorials":
    st.header("Tutorials")
    st.write("Here are some tutorials on Indian Sign Language signs.")
    st.video("static/video1.mp4", format="video/mp4", caption="How are you sign")
    st.video("static/video2.mp4", format="video/mp4", caption="Another sign")

elif choice == "Project":
    st.header("Sign Recognition Project")
    st.write("Capture your sign and let the system recognize it.")
    
    # Camera capture
    run = st.button("Start Camera")
    if run:
        st.write("Starting camera...")
        # Note: Streamlit currently does not support direct camera access for video capture.
        # You'll need a different approach or workaround for capturing video.
        
    # Display and compare results
    result = st.empty()
    if st.button("Submit"):
        # Use placeholder data for demonstration
        sample_image = np.zeros((64, 64, 3))  # Replace with actual image capture
        prediction = predict_sign(sample_image)
        result.text(f"Predicted Sign: {prediction}")

    # Listen button
    if st.button("Listen"):
        text = result.text
        audio_bytes = play_audio(text)
        st.audio(audio_bytes, format='audio/mp3')

