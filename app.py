import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model (Ensure 'model.pkl' is in your repo)
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("Diabetes Prediction System")
st.write("Enter patient details to predict the likelihood of diabetes.")

# Create input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=100)
    smoking = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])

# Map inputs to match your LabelEncoder values from the notebook
gender_map = {"Female": 0, "Male": 1, "Other": 2}
smoking_map = {"No Info": 0, "current": 1, "ever": 2, "former": 3, "never": 4, "not current": 5}
binary_map = {"No": 0, "Yes": 1}

if st.button("Predict Diabetes Risk"):
    # Prepare features in the EXACT order used in your notebook training:
    # Order: age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, sex, Smoking_history
    features = np.array([[
        age, 
        binary_map[hypertension], 
        binary_map[heart_disease], 
        bmi, 
        hba1c, 
        glucose, 
        gender_map[gender], 
        smoking_map[smoking]
    ]])

    prediction = model.predict(features)
    
    st.subheader("Results:")
    if prediction[0] == 1:
        st.error("The model predicts the patient is **Diabetic**.")
    else:
        st.success("The model predicts the patient is **Not Diabetic**.")
