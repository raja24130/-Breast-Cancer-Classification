import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Create Streamlit app
st.title("Breast Cancer Classification")
st.write("Accuracy:", accuracy)

# Add input fields for user to enter features
col1, col2 = st.beta_columns(2)
with col1:
    mean_radius = st.number_input("Mean radius", value=15.0, min_value=0.0, max_value=50.0)
    mean_texture = st.number_input("Mean texture", value=20.0, min_value=0.0, max_value=50.0)
    mean_perimeter = st.number_input("Mean perimeter", value=100.0, min_value=0.0, max_value=300.0)

with col2:
    mean_area = st.number_input("Mean area", value=500.0, min_value=0.0, max_value=2500.0)
    mean_smoothness = st.number_input("Mean smoothness", value=0.1, min_value=0.0, max_value=1.0)

# Create a prediction button
if st.button("Predict"):
    # Create a numpy array with user input data
    user_input = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    # Make prediction using trained classifier
    prediction = classifier.predict(user_input)
    # Display prediction result
    if prediction[0] == 0:
        st.write("Result: Benign")
    else:
        st.write("Result: Malignant")
