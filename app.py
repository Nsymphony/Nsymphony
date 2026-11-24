
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. Load and train the model (we do this fast so the app works immediately) ---
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- 2. Build the Web App UI ---
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient details below to predict the likelihood of diabetes.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# --- 3. Make Prediction ---
if st.button("Predict Diabetes Risk"):
    # Prepare the input data
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
                              columns=column_names[:-1])
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes.** (Probability: {probability*100:.1f}%)")
        st.write("Please consult a healthcare professional for further testing.")
    else:
        st.success(f"✅ **Low Risk of Diabetes.** (Probability: {probability*100:.1f}%)")
        st.write("Keep up the healthy lifestyle!")
