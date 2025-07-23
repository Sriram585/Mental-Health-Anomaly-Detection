# Mental Health Anomaly Detection

A machine learning project to detect potential mental health anomalies using biometric, lifestyle, and physiological data. Built with a clean modular structure and deployed via a Streamlit web interface.

---

## Project Overview

This project takes various health-related inputs like age, sleep, mood, heart rate, and stress level, and uses a trained ML model to predict potential anomalies in mental health.

---

## Tech Stack

- **Python**
- **scikit-learn**
- **pandas, numpy**
- **Streamlit** for interactive web UI
- **Pickle** for model serialization

---

## Features

- Data preprocessing and feature engineering
- Model selection using performance metrics
- Model saving and loading
- Streamlit-based frontend for prediction
- Validation and error messaging for inputs

---

## Input Features

| Feature                 | Description                          | Example        |
|------------------------|--------------------------------------|----------------|
| Age                    | Age of the person                    | `25`           |
| Gender                 | `1` = Male, `0` = Female             | `1`            |
| Weight                 | Weight in kg                         | `70`           |
| Height                 | Height in cm                         | `175`          |
| Medical_Conditions     | `1` = Yes, `0` = No                  | `0`            |
| Medication             | `1` = Yes, `0` = No                  | `0`            |
| Smoker                 | `1` = Yes, `0` = No                  | `0`            |
| Alcohol_Consumption    | `1` = Yes, `0` = No                  | `0`            |
| Day_of_Week            | 0 = Monday ... 6 = Sunday            | `2`            |
| Sleep_Duration         | Total hours slept                    | `7.5`          |
| Deep_Sleep_Duration    | Hours of deep sleep                  | `2.5`          |
| REM_Sleep_Duration     | Hours of REM sleep                   | `1.2`          |
| Wakeups                | Number of wakeups                    | `2`            |
| Snoring                | `1` = Yes, `0` = No                  | `0`            |
| Heart_Rate             | BPM (beats per minute)               | `72`           |
| Blood_Oxygen_Level     | Oxygen saturation (%)                | `97`           |
| ECG                    | Raw ECG feature                      | `0.85`         |
| Calories_Intake        | kcal/day                             | `2200`         |
| Water_Intake           | ml/day                               | `2500`         |
| Stress_Level           | Scale 0-5                            | `3`            |
| Mood                   | `1` = Happy, `0` = Sad               | `1`            |
| Skin_Temperature       | In Celsius                           | `36.5`         |
| Body_Fat_Percentage    | %                                    | `20.5`         |
| Muscle_Mass            | %                                    | `40.2`         |
| Health_Score           | Out of 100                           | `85`           |
| Height_m               | Height in meters                     | `1.75`         |
| BMI                    | Body Mass Index                      | `22.9`         |
| Sleep_Efficiency       | Percentage                           | `92`           |

---
