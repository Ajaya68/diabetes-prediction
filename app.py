import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 2. Page Styling
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction System")
st.write("Please fill in the patient details below to get a prediction.")

# 3. Create Input Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
    # Using text labels for better UX
    gender_choice = st.selectbox("Gender", ["Female", "Male", "Other"])
    hypertension_choice = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease_choice = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0, value=100.0)
    smoking_choice = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])

# 4. Map text labels back to the numbers your model expects
gender_map = {"Female": 0, "Male": 1, "Other": 2}
smoking_map = {"No Info": 0, "current": 1, "ever": 2, "former": 3, "never": 4, "not current": 5}
binary_map = {"No": 0, "Yes": 1}

# 5. Prediction Logic
if st.button("Run Prediction"):
    # Feature order MUST match your notebook: age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking
    features = np.array([[
        age, 
        binary_map[hypertension_choice], 
        binary_map[heart_disease_choice], 
        bmi, 
        hba1c, 
        glucose, 
        gender_map[gender_choice], 
        smoking_map[smoking_choice]
    ]])

    prediction = model.predict(features)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: High Risk of Diabetes")
    else:
        st.success("### Result: Low Risk of Diabetes")
