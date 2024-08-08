# 1. Import packages

import os, mlflow
import mlflow.sklearn
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, preprocessing, pipeline
import streamlit as st

# 2. Load in your resources
# Use cache
# (A) Function to load pickle object
@st.cache_resource
def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        pickle_object = pkl.load(f)
    return pickle_object

# (B) Function to load ML model
@st.cache_resource
def load_model(uri):
    model= mlflow.sklearn.load_model(uri)
    return model

# Use the functions to load in the resources
os.chdir(r"C:\Users\MSFRotPC04\Desktop\CIAST_ML_Siri 2\ml-project-1")
# print(os.getcwd())
# (A) ordinal encoder
encoder_1 = load_pickle(r"src\ordinal_encoder_1.pkl")
encoder_2 = load_pickle(r"src\ordinal_encoder_2.pkl")
model = load_model("models:/titanic_model_production@champion")

# For testing purpose only
# st.write(type(encoder_1))
# st.write(type(encoder_2))
# st.write(type(model))

# Add a title
st.title("TITANIC SURVIVAL PREDICTION")

# Create the input widgets for user input

# Form
with st.form("User Input"):
    
    # (A) Pclass
    pclass = st.number_input("Pclass", min_value=1, max_value=3)

    # (B) Sex
    sex = st.selectbox("Sex", options = ['male', 'female'])

    # (C) Sibsp
    sibsp = st.number_input("Number of siblings and Spouse", min_value=0, max_value=10)

    # (D) Parch
    parch = st.number_input("Parch", min_value=0, max_value=10)

    # (E) Fare
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, step=0.50)

    # (F) Embarked
    embarked = st.selectbox("Embarked", options = ['C', 'Q', 'S'])

    # (G) Age Group
    age = st.selectbox("Age Group", options = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior'])

    # Form submit button
    submit = st.form_submit_button("Submit")

# Create a label map
label_map = {0: "No", 1: "Yes"}
columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked','AgeGroup']
user_input = pd.DataFrame(np.array([[pclass, sex, sibsp, parch, fare, embarked, age]]), columns=columns)

# Process the categorical inputs
user_input['AgeGroup'] = encoder_1.transform(user_input['AgeGroup'].values.reshape(1,-1))
user_input[['Sex', 'Embarked']] = encoder_2.transform(user_input[['Sex', 'Embarked']].values)
st.write(user_input)

prediction = model.predict(user_input.values)
prediction_class = label_map[prediction[0]]
st.write(" Will I survive the Titanic? ")
st.write(prediction_class)