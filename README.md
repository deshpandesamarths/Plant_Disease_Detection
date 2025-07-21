
# Plant Disease Detection and Treatment Recommendation

## Overview

This project is a deep learning–powered web application for diagnosing plant diseases from leaf images and recommending appropriate treatments. It uses a pre-trained Convolutional Neural Network (CNN) model for classification and suggests both chemical and organic remedies. The app is built using **Streamlit**, providing an intuitive interface for users to upload images and receive instant predictions along with actionable treatment guidance.

---

## Features

* **Plant Disease Classification**
  Upload a leaf image, and the system will predict whether the plant is healthy or infected, and if so, identify the disease.

* **Treatment Recommendation**
  View detailed explanations, artificial (chemical), and organic treatment suggestions based on the predicted disease.

* **Streamlit Interface**
  Simple, browser-based UI for non-technical users.

---

## Supported Plant Classes

The model supports classification for several common crop diseases including:

* Apple: Apple Scab, Healthy
* Corn (Maize): Northern Leaf Blight, Healthy
* Grape: Black Rot, Healthy
* Peach: Bacterial Spot, Healthy
* Potato: Late Blight, Healthy
* Tomato: Septoria Leaf Spot, Healthy

---

## Project Structure

```
plant-disease-detector/
├── app.py                    # Main Streamlit application
├── class_indices.json        # Mapping of model output indices to disease labels
├── requirements.txt          # Python dependencies
├── plant_disease_detection.ipynb # Development and experimentation notebook
└── README.md                 # Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/deshpandesamarths/Plant_Disease_Detection.git
cd Plant_Disease_Detection
```

### 2. Install Required Packages

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

---

## Running the Application

To start the Streamlit app locally:

```bash
streamlit run app.py
```

---

## Using the Application

Once the app is running in your browser:

1. **Upload the Model File:**
   Upload your pre-trained model file: `plant_disease_model.h5`.

2. **Upload the Remedies File:**
   Upload the file `remedies.json`, which contains treatment information.

3. **Upload a Leaf Image:**
   Upload a `.jpg`, `.jpeg`, or `.png` image of a plant leaf.

4. **Classify the Image:**
   Click the **"Classify"** button to see the predicted disease.

5. **View Remedies:**
   Click **"Show Remedies"** to see detailed explanations and treatment options.

---


## Notes

* The `class_indices.json` file (included in the repo) maps model output indices to human-readable disease names.
* The `plant_disease_model.h5` and `remedies.json` files must be uploaded by the user during app execution.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

## Acknowledgments

* TensorFlow/Keras for deep learning model support
* Streamlit for rapid app development
* Public datasets from PlantVillage and other sources
* Open-source contributors for remedy data

---

