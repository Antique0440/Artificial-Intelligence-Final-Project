
# create simple web app for testing our scholarhip prediction Model
# a student/ financial aid office can input students' data and at a click of a button, get to kow thier chance of been accepted for a scholarship


import streamlit as st
import pandas as pd
import joblib
import numpy as np

# load the pre-trained ensemble model and scaler
ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')  
#title of the app
st.title("🎓 Scholarship Eligibility Prediction App")

st.markdown("""
Welcome! Fill out the information below to check if you might be eligible for a scholarship.  
The model uses an ensemble of multiple machine learning models to give you the best prediction.
""")

# create user input fields
gpa = st.number_input("Enter GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
family_income = st.number_input("Enter Family Income (USD)", min_value=0, value=50000)
extracurriculars = st.number_input("Number of Extracurricular Activities", min_value=0, value=2)
leadership = st.selectbox("Leadership Experience", options=['No', 'Yes'])
volunteer_hours = st.number_input("Volunteer Hours", min_value=0, value=50)
essay_score = st.number_input("Essay Score (out of 10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
recommendation_strength = st.selectbox("Recommendation Strength", options=['Weak', 'Average', 'Strong'])
region = st.selectbox("Region", options=['Urban', 'Rural'])
first_gen = st.selectbox("First Generation Student?", options=['Yes', 'No'])
wassce_score = st.number_input("WASSCE Score (out of 600)", min_value=0, max_value=600, value=350)

# map categorical inputs to numerical values
region_map = {'Urban': 1, 'Rural': 0}
first_gen_map = {'Yes': 1, 'No': 0}
recommendation_map = {'Weak': 0, 'Average': 1, 'Strong': 2}
leadership_map = {'No': 0, 'Yes': 1}

# compute the eligibility_score from other features
eligibility_score = gpa * 0.70 + essay_score * 0. + extracurriculars * 0.2  

# Create a DataFrame with user input
input_data = pd.DataFrame([{
    'gpa': gpa,
    'family_income': family_income,
    'extracurriculars': extracurriculars,
    'leadership': leadership_map[leadership],
    'volunteer_hours': volunteer_hours,
    'essay_score': essay_score,
    'recommendation_strength': recommendation_map[recommendation_strength],
    'eligibility_score': eligibility_score,  # Include eligibility_score
    'region': region_map[region],  # Ensure 'region' is included
    'first_gen': first_gen_map[first_gen],
    'wassce_score': wassce_score    
}])

# do scaling as during training
scaled_input = scaler.transform(input_data)  # use loaded scaler to transform

# set prediction threshold which is adjustable
threshold = st.slider('Prediction Threshold', min_value=0.0, max_value=1.0, value=0.35, step=0.01)

#  create prediction button
if st.button('Predict Scholarship Eligibility'):
    # probability for class 1 (Eligible)
    probability = ensemble_model.predict_proba(scaled_input)[:, 1]
    
    # apply threshold
    prediction = (probability >= threshold).astype(int)
    
    # display results
    if prediction[0] == 1:
        st.success(f"🎉 Congratulations! Based on the inputs, you are **likely eligible** for a scholarship! (Confidence: {probability[0]:.2f})")
    else:
        st.error(f"❌ Sorry! Based on the inputs, you are **likely not eligible**. (Confidence: {probability[0]:.2f})")

    st.markdown("---")
    st.subheader("Model's Probability Output")
    st.progress(float(probability[0]))
