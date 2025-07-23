import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Mental Health Anomaly Detection")
st.markdown("Enter your information below (only numeric values accepted):")

# Feature names with examples
feature_examples = {
    'Age': '25',
    'Gender': '1 (Male), 0 (Female)',
    'Weight': '70 (kg)',
    'Height': '175 (cm)',
    'Medical_Conditions': '0 (None), 1 (Yes)',
    'Medication': '0 (No), 1 (Yes)',
    'Smoker': '0 (No), 1 (Yes)',
    'Alcohol_Consumption': '0 (No), 1 (Yes)',
    'Day_of_Week': '0 (Monday) to 6 (Sunday)',
    'Sleep_Duration': '7.5 (hours)',
    'Deep_Sleep_Duration': '2.5 (hours)',
    'REM_Sleep_Duration': '1.2 (hours)',
    'Wakeups': '2 (times)',
    'Snoring': '0 (No), 1 (Yes)',
    'Heart_Rate': '72 (bpm)',
    'Blood_Oxygen_Level': '97 (%)',
    'ECG': '0.85',
    'Calories_Intake': '2200 (kcal)',
    'Water_Intake': '2500 (ml)',
    'Stress_Level': '3 (scale of 0-5)',
    'Mood': '1 (Happy), 0 (Sad)',
    'Skin_Temperature': '36.5 (°C)',
    'Body_Fat_Percentage': '20.5 (%)',
    'Muscle_Mass': '40.2 (%)',
    'Health_Score': '85',
    'Height_m': '1.75 (m)',
    'BMI': '22.9',
    'Sleep_Efficiency': '92 (%)'
}

user_inputs = []
error_fields = []

# Generate input fields
for feature, example in feature_examples.items():
    label = f"{feature} (e.g., {example}):"
    value = st.text_input(label)
    try:
        val = float(value) if value else None
        if val is None:
            error_fields.append(feature)
        user_inputs.append(val)
    except ValueError:
        error_fields.append(feature)
        user_inputs.append(None)

# Predict button
if st.button("Predict Mental Health Status"):
    if error_fields:
        st.error(f"Invalid input in: {', '.join(error_fields)}. Please enter numeric values as per the examples.")
    else:
        try:
            input_array = np.array(user_inputs).reshape(1, -1)
            if input_array.shape[1] != model.n_features_in_:
                st.error(f"Model expects {model.n_features_in_} features, but received {input_array.shape[1]}.")
            else:
                prediction = model.predict(input_array)[0]
                confidence = max(model.predict_proba(input_array)[0])
                st.success(f"Prediction: **{prediction}** with confidence **{confidence:.2f}**")
                if prediction == 0:
                    st.info("🟢 No anomalies detected. Your mental health appears stable.")
                else:
                    st.warning("🔴 Anomalies detected. Consider consulting a mental health professional.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
