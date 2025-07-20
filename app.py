import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("rf_model.pkl")

# Set the app title
st.title("📝 Writing Score Predictor")

st.write("Fill in the student's details to predict their **writing score**.")

# Define categorical options
gender_options = ["female", "male"]
race_options = ["group A", "group B", "group C", "group D", "group E"]
education_options = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
]
lunch_options = ["standard", "free/reduced"]
prep_options = ["none", "completed"]

# Input widgets
gender = st.selectbox("Gender", gender_options)
race = st.selectbox("Race/Ethnicity", race_options)
parental_education = st.selectbox("Parental Level of Education", education_options)
lunch = st.selectbox("Lunch Type", lunch_options)
prep_course = st.selectbox("Test Preparation Course", prep_options)

math_score = st.number_input("Math Score", min_value=0, max_value=100, value=70)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)

# Predict button
if st.button("🎯 Score"):
    # Create a single row DataFrame with inputs
    input_df = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parental_education,
        "lunch": lunch,
        "test preparation course": prep_course,
        "math score": math_score,
        "reading score": reading_score
    }])

    # Predict using the loaded model
    predicted_score = model.predict(input_df)[0]

    st.success(f"📘 Predicted Writing Score: **{predicted_score:.2f}**")
