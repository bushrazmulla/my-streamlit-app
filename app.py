# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏭 Manufacturing Equipment Output Predictor")
st.write("Predict hourly machine output (Parts per Hour) based on input parameters.")

# Input fields
Injection_Temperature = st.number_input("Injection Temperature (°C)", 180, 250, 200)
Injection_Pressure   = st.number_input("Injection Pressure (bar)", 80, 150, 100)
Cycle_Time           = st.number_input("Cycle Time (sec)", 15, 45, 30)
Cooling_Time         = st.number_input("Cooling Time (sec)", 8, 20, 10)
Material_Viscosity   = st.number_input("Material Viscosity (Pa·s)", 100, 400, 200)
Ambient_Temperature  = st.number_input("Ambient Temperature (°C)", 18, 28, 22)
Machine_Age          = st.number_input("Machine Age (years)", 1, 15, 5)
Operator_Experience  = st.number_input("Operator Experience (months)", 1, 120, 24)
Maintenance_Hours    = st.number_input("Maintenance Hours since last check", 0, 200, 50)

# Prediction button
if st.button("Predict Output"):
    input_data = np.array([[Injection_Temperature, Injection_Pressure, Cycle_Time,
                            Cooling_Time, Material_Viscosity, Ambient_Temperature,
                            Machine_Age, Operator_Experience, Maintenance_Hours]])
    
    prediction = model.predict(input_data)[0]
    st.success(f"🔧 Predicted Output: {prediction:.2f} parts/hour")
