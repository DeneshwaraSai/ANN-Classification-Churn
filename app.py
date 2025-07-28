import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

model = load_model('ann_model.h5')

with open('labelEncoder_gender.pkl', 'rb') as file:
    labelEncoder_gender = pickle.load(file)

with open('oneHotEncoder_geo.pkl', 'rb') as file:
    oneHotEncoder_geo = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

## streamlit app
st.title('Customer Churn PRediction') 

creditScore = st.slider('CreditScore', 500, 900)
geography = st.selectbox('Geography',  oneHotEncoder_geo.categories_[0] )
gender = st.selectbox('Gender', labelEncoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimatedSalary = st.number_input('EstimatedSalary')
 

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [creditScore],
    'Gender': [labelEncoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimatedSalary]
})

# One-hot encode 'Geography'
geo_encoded = oneHotEncoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=oneHotEncoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

st.dataframe(input_data)

# Scale the input data
input_data_scaled = scalar.transform(input_data)

pred = model.predict(input_data_scaled)
pred_prob = pred[0][0]
st.write(f"The prediction prob is :  {pred_prob}")

if pred_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
