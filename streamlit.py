import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train the logistic regression model
clf = LogisticRegression()
clf.fit(X, y)

# Define a function to make predictions
def predict(data):
    prediction = clf.predict(data)
    if prediction[0] == 0:
        return "Malignant"
    else:
        return "Benign"

# Create the Streamlit app
st.title("Breast Cancer Prediction using Logistic Regression")

# Create a form to allow the user to input their data
form = st.form("my_form")
for feature in data.feature_names:
    form.slider(feature, X[feature].min(), X[feature].max(), X[feature].mean(), key=feature)
submit_button = form.form_submit_button("Predict")

# When the "Predict" button is clicked, make a prediction and show the result
if submit_button:
    inputs = form.form_values()
    data = pd.DataFrame(inputs, index=[0])
    result = predict(data)
    st.write("Prediction:", result)
                 
