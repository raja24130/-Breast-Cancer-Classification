import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
breast_cancer_dataset = load_breast_cancer()

# Create a DataFrame from the dataset
breast_cancer_dataset = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(data['feature_names'], ['target']))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

# Create a logistic regression classifier
clf = LogisticRegression(random_state=42)

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Define a function to take in user inputs and make a prediction
def predict(model, input_data):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        pred = 'Malignant'
    else:
        pred = 'Benign'
    return pred

# Create the Streamlit app
def app():
    st.title("Breast Cancer Prediction App")
    
    # Add a sidebar with user inputs
    st.sidebar.header("User Input Features")
    
    mean_radius = st.sidebar.slider("Mean radius", float(df.mean_radius.min()), float(df.mean_radius.max()), float(df.mean_radius.mean()))
    mean_texture = st.sidebar.slider("Mean texture", float(df.mean_texture.min()), float(df.mean_texture.max()), float(df.mean_texture.mean()))
    mean_perimeter = st.sidebar.slider("Mean perimeter", float(df.mean_perimeter.min()), float(df.mean_perimeter.max()), float(df.mean_perimeter.mean()))
    mean_area = st.sidebar.slider("Mean area", float(df.mean_area.min()), float(df.mean_area.max()), float(df.mean_area.mean()))
    mean_smoothness = st.sidebar.slider("Mean smoothness", float(df.mean_smoothness.min()), float(df.mean_smooth

# Add a sidebar with input sliders for each feature
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

for feature in data.feature_names:
    input_data[feature] = st.sidebar.slider(feature, X[feature].min(), X[feature].max(), X[feature].mean())
# Make prediction and display result
result = predict(clf, pd.DataFrame(input_data, index=[0]))
st.write("Prediction:", result)
