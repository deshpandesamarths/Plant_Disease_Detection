import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import h5py
import streamlit as st

# Path to Google Drive where files are stored
model_path = '/content/drive/MyDrive/Colab Notebooks/plant_disease_model.h5'
class_indices_path = '/content/drive/MyDrive/Colab Notebooks/class_indices.json'
remedies_path = '/content/drive/MyDrive/Colab Notebooks/remedies.json'

def custom_load_model(filepath):
    with h5py.File(filepath, 'r') as f:
        # Load the model configuration
        model_config = f.attrs['model_config']
        
        # Convert the configuration to JSON format
        model_json = json.loads(model_config)
        
        # Fix the BatchNormalization layer configuration if necessary
        for layer in model_json['config']['layers']:
            if layer['class_name'] == 'BatchNormalization' and isinstance(layer['config']['axis'], list):
                layer['config']['axis'] = layer['config']['axis'][0]

        # Create model from the fixed configuration
        model = tf.keras.models.model_from_json(json.dumps(model_json))

        # Load weights
        for layer in model.layers:
            g = f[layer.name]
            weights = [g[weight_name] for weight_name in g.attrs['weight_names']]
            layer.set_weights(weights)

    return model

# Load the pre-trained model using the custom function
try:
    model = custom_load_model(model_path)
    st.write("Model loaded successfully!")
except FileNotFoundError as e:
    st.write(f"Error loading model: {e}")

# Load the class names
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Load the remedies
with open(remedies_path, 'r') as f:
    remedies = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit app
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, image, class_indices)
            st.success(f'Prediction: {prediction}')
            if st.button('Show Remedies'):
                disease_info = remedies.get(prediction, None)
                if disease_info:
                    st.write(f"**Explanation:** {disease_info['Explanation']}")
                    st.write("### Artificial Treatment")
                    for treatment in disease_info['Artificial Treatment']:
                        st.write(f"- {treatment}")
                    st.write("### Organic Treatment")
                    for treatment in disease_info['Organic Treatment']:
                        st.write(f"- {treatment}")
                else:
                    st.write("No remedies found for this disease.")
