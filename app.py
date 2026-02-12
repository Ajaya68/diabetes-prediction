import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Diabetes Prediction App")

# Inputs (Order: age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking)
age = st.number_input("Age", value=30.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.number_input("BMI", value=25.0)
hba1c = st.number_input("HbA1c Level", value=5.5)
glucose = st.number_input("Blood Glucose Level", value=100.0)
gender = st.selectbox("Gender (0:Female, 1:Male, 2:Other)", [0, 1, 2])
smoking = st.selectbox("Smoking (0:No Info, 1:current, 2:ever, 3:former, 4:never, 5:not current)", [0, 1, 2, 3, 4, 5])

if st.button("Predict"):
    features = np.array([[age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking]])
    prediction = model.predict(features)
    st.write("Result: ", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
