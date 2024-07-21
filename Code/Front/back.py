import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from collections import defaultdict
import joblib
import re








# def find_employment_avenues_recommendations(user_input):
#     model_path = os.path.join('Code', 'Pickles', 'employment-avenues-recomendations.h5')
#     model_path = os.path.abspath(model_path)
#     model = load_model(model_path)
#     trait_sums = defaultdict(lambda: {'sum': 0, 'count': 0})

#     # Process the original dictionary
#     for key, value in user_input.items():
#         # Extract the trait name by removing digits
#         trait_name = re.sub(r'\d', '', key).strip()
#         # Update the sum and count for this trait
#         trait_sums[trait_name]['sum'] += int(value)
#         trait_sums[trait_name]['count'] += 1

#     # New dictionary to store the averages
#     traits_dict = {trait: round(values['sum'] / values['count'], 0) for trait, values in trait_sums.items()}
#     user_df = pd.DataFrame([traits_dict])
#     user_df = user_df.drop(['job'], axis=1)

#     scaler_path = os.path.join('Code', 'Pickles', 'scaler.pkl')
#     scaler_path = os.path.abspath(scaler_path)
#     scaler = joblib.load(scaler_path)
#     user_scaled = scaler.transform(user_df)

#     num_predictions = 5
#     recommendations = {}
#     for i in range(num_predictions):
#         recommendations[str(i)] = model.predict(user_scaled)


#     return recommendations

# a = {'job': '123', 'Leadership1': '4', 'Leadership2': '4'}

def find_employment_avenues_recommendations(user_input):
    # Paths to the model and scaler
    model_path = os.path.join('Code', 'Pickles', 'employment-avenues-recomendations.h5')
    model_path = os.path.abspath(model_path)
    scaler_path = os.path.join('Code', 'Pickles', 'scaler.pkl')
    scaler_path = os.path.abspath(scaler_path)
    
    try:
        # Load model and scaler
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return {}
    
    # Dictionary to store sum and count of each trait
    trait_sums = defaultdict(lambda: {'sum': 0, 'count': 0})
    
    # Process the original dictionary
    for key, value in user_input.items():
        try:
            # Extract the trait name by removing digits
            trait_name = re.sub(r'\d', '', key).strip()
            # Update the sum and count for this trait
            trait_sums[trait_name]['sum'] += int(value)
            trait_sums[trait_name]['count'] += 1
        except ValueError as ve:
            print(f"ValueError for key {key} with value {value}: {ve}")
    
    # New dictionary to store the averages
    traits_dict = {trait: round(values['sum'] / values['count'], 0) for trait, values in trait_sums.items()}
    
    # Create DataFrame
    user_df = pd.DataFrame([traits_dict])
    
    # Check if 'job' column exists before dropping it
    if 'job' in user_df.columns:
        user_df = user_df.drop(['job'], axis=1)
    
    # Scale the data
    try:
        user_scaled = scaler.transform(user_df)
    except Exception as e:
        print(f"Error scaling data: {e}")
        return {}
    
    # Make predictions
    num_predictions = 5
    recommendations = {}
    try:
        for i in range(num_predictions):
            prediction = model.predict(user_scaled)
            recommendations[f'recommendation_{i+1}'] = prediction[i]  
        print(f"Error during model prediction: {e}")
    except Exception as e:
        print(f"Error with recommendations: {e}")
        return {}
    
    return recommendations

a = {'job': '123', 'Leadership1': '4', 'Leadership2': '4'}