import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

import os










def predict_5_jobs(preprocessed_text):
    trait_scores = list(preprocessed_text.values())
    # trait_scores = [2, 1, 3, 3, 0, 1, 1, 2, 1, 2, 0, 1, 0, 0, 2, 1, 0, 1, 0]

    # Load data
    csv_path = os.path.join('..', 'Data', 'Diagested_data', 'Numeric_resume_data.csv')
    data = pd.read_csv(csv_path)
    df = pd.DataFrame(data)

    # Define trait columns
    trait_columns = [
        'Leadership', 'Communication', 'Teamwork', 'Problem Solving', 'Creativity',
        'Adaptability', 'Work Ethic', 'Time Management', 'Interpersonal Skills', 
        'Attention to Detail', 'Initiative', 'Analytical Thinking', 'Emotional Intelligence', 
        'Integrity', 'Resilience', 'Cultural Awareness', 'Programming Languages', 
        'Technical Skills', 'Office Tools'
    ]

    # Convert job titles to numerical labels
    job_titles = df['Job Title'].unique()
    job_domains = df[['Job Title', 'Domain']].drop_duplicates().set_index('Job Title')['Domain'].to_dict()
    job_title_to_index = {title: idx for idx, title in enumerate(job_titles)}
    index_to_job_title = {idx: title for title, idx in job_title_to_index.items()}
    df['Job Label'] = df['Job Title'].map(job_title_to_index)

    # Prepare feature and target arrays
    X = df[trait_columns].values
    y = df['Job Label'].values

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding
    num_classes = len(job_titles)
    lb = LabelBinarizer()
    lb.fit(y)  # Fit on the entire set of labels to ensure it has all classes
    y_train_onehot = lb.transform(y_train)
    y_test_onehot = lb.transform(y_test)

    # Load the saved model from a file
    model_save_path = os.path.join('..', 'Code', 'Pickles', 'job_recommendation_algorith.h5')
    model = load_model(model_save_path)
    print(f'Model loaded from {model_save_path}')
    trait_scores = np.array(trait_scores).reshape(1, -1)
    trait_scores_scaled = scaler.transform(trait_scores)  # Scale the input trait scores
    predictions = model.predict(trait_scores_scaled)
    
    # Get the indices of the top N probabilities
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    
    # Map indices back to job titles and domains
    top_jobs = [(index_to_job_title[idx], job_domains[index_to_job_title[idx]]) for idx in top_indices]
    print(top_jobs)
    final_list = []
    for job, domain in top_jobs:
        final_list.append((f' - {job} in the field of {domain}'))
    final_dict = {}
    for i in range(len(final_list)):
        final_dict[i+1] = final_list[i]
    


    return final_dict
# print(predict_5_jobs({1:2}))