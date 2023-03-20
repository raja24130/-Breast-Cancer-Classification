import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
# Train the logistic regression model
clf = LogisticRegression()
clf.fit(X, y)

# Define a function to make predictions
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')

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
                 
