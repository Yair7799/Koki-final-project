import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from collections import defaultdict
import joblib
import re
import PyPDF2

# Paths to model and scaler
model_path = os.path.abspath(r'C:\Users\yehus\pythonFiles\Koki-final-project\Code\Pickles\employment-avenues-recomendations.h5')
scaler_path = os.path.abspath(r'C:\Users\yehus\pythonFiles\Koki-final-project\Code\Pickles\scaler.pkl')

# Load model and scaler
scaler = joblib.load(scaler_path)
model = load_model(model_path)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def preprocess_text(text):
    # Example feature extraction
    features = [len(text)]  # Feature: length of the text
    
    # Ensure features list has exactly 19 elements
    while len(features) < 19:
        features.append(0)  # Fill with 0 to match the scaler's expectations
    
    # Create a DataFrame with 19 feature columns
    feature_names = [f'feature_{i}' for i in range(19)]
    return pd.DataFrame([features], columns=feature_names)

def predict_character_traits(text):
    features_df = preprocess_text(text)
    try:
        scaled_features = scaler.transform(features_df)
        predictions = model.predict(scaled_features)
        return predictions.tolist()
    except Exception as e:
        print(f"Error making predictions: {e}")
        return []