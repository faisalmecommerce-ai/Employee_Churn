import streamlit as st
import pandas as pd
from joblib import load
import dill

# Load the pretrained model
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')


# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction


st.title('Employee Churn Prediction App')
st.subheader('Based on Employee Data Set')
st.subheader('Created By Faisal Memon')


# Display categorical features
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('categorical')




cat_cols = my_feature_dict.get('categorical_features',[])
cat_options = my_feature_dict.get('categorical_options',{})

categorical_input_vals={}

for col in cat_cols:
    options = cat_options.get(col,[])

    if options:
        categorical_input_vals[col] = st.selectbox(
            col, 
            options,
            key=col
        )
    else:
        st.warning(f"No options found for {col}")

# Load numerical features
numerical_input = my_feature_dict.get('numerical')

# Display numerical features
st.subheader('Numerical Features')


num_cols = my_feature_dict.get('numerical_features', [])
numerical_input_vals = {}

for col in  num_cols:
    numerical_input_vals[col] = st.number_input(f"Enter {col}",value=0,step=1, format="%d", key=f"num_{col}")

# Combine numerical and categorical input dicts
##input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))

##input_data= pd.DataFrame.from_dict(input_data,orient='index').T

input_data_dict = {**categorical_input_vals, **numerical_input_vals}
input_data = pd.DataFrame([input_data_dict])

expected_columns = cat_cols + num_cols
input_data = input_data[expected_columns]

# Churn Prediction
if st.button('Predict'):
    try:
        prediction = predict_churn(input_data)[0]
        translation_dict = {"Yes": "Expected", "No": "Not Expected"}
        prediction_translate = translation_dict.get(prediction)
        st.write(f'The Prediction is **{prediction}**, Hence customer is **{prediction_translate}** to churn.')
    except Exception as e:
        st.error(f"Prediction Error {e}")