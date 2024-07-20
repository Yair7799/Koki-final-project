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








def find_employment_avenues_recommendations(user_input):
    model_path = os.path.join('Code', 'Pickles', 'employment-avenues-recomendations.h5')
    model_path = os.path.abspath(model_path)
    model = load_model(model_path)
    trait_sums = defaultdict(lambda: {'sum': 0, 'count': 0})

    # Process the original dictionary
    for key, value in user_input.items():
        # Extract the trait name by removing digits
        trait_name = re.sub(r'\d', '', key).strip()
        # Update the sum and count for this trait
        trait_sums[trait_name]['sum'] += int(value)
        trait_sums[trait_name]['count'] += 1

    # New dictionary to store the averages
    traits_dict = {trait: round(values['sum'] / values['count'], 0) for trait, values in trait_sums.items()}
    user_df = pd.DataFrame([traits_dict])
    user_df = user_df.drop(['job'], axis=1)

    scaler_path = os.path.join('Code', 'Pickles', 'scaler.pkl')
    scaler_path = os.path.abspath(scaler_path)
    scaler = joblib.load(scaler_path)
    user_scaled = scaler.transform(user_df)

    num_predictions = 5
    recommendations = {}
    for i in range(num_predictions):
        recommendations[str(i)] = model.predict(user_scaled)


    return recommendations

a = {'job': '123', 'Leadership1': '4', 'Leadership2': '4'}

def find_employment_avenues_recommendations(user_input):
    model_path = os.path.join('Code', 'Pickles', 'employment-avenues-recomendations.h5')
    model_path = os.path.abspath(model_path)
    model = load_model(model_path)
    trait_sums = defaultdict(lambda: {'sum': 0, 'count': 0})

    # Process the original dictionary
    for key, value in user_input.items():
        # Extract the trait name by removing digits
        trait_name = re.sub(r'\d', '', key).strip()
        # Update the sum and count for this trait
        trait_sums[trait_name]['sum'] += int(value)
        trait_sums[trait_name]['count'] += 1

    # New dictionary to store the averages
    traits_dict = {trait: round(values['sum'] / values['count'], 0) for trait, values in trait_sums.items()}
    user_df = pd.DataFrame([traits_dict])
    user_df = user_df.drop(['job'], axis=1)

    scaler_path = os.path.join('Code', 'Pickles', 'scaler.pkl')
    scaler_path = os.path.abspath(scaler_path)
    scaler = joblib.load(scaler_path)
    user_scaled = scaler.transform(user_df)

    num_predictions = 5
    recommendations = {}
    for i in range(num_predictions):
        recommendations[str(i)] = model.predict(user_scaled)


    return recommendations

a = {'job': '123', 'Leadership1': '4', 'Leadership2': '4'}