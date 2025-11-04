import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ---------------- LOAD MODEL ---------------- #
model = CatBoostClassifier()
model.load_model("catboost_hr_model.cbm")

st.set_page_config(page_title="HR Analytics – Employee Promotion Prediction System", layout="centered")

st.title("📊 HR Analytics – Employee Promotion Prediction System")

st.write("🎉 Welcome to the Employee Promotion Prediction System! 👋")
st.write("Enter employee details below to predict whether the employee is likely to be promoted.🚀")

# ---------------- INPUTS ---------------- #
age = st.number_input("Age", min_value=18, max_value=60, value=30)
length_of_service = st.number_input("Length of Service (years)", min_value=0, max_value=40, value=5)
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=60)
no_of_trainings = st.number_input("Number of Trainings", min_value=0, max_value=20, value=3)
previous_year_rating = st.selectbox("Previous Year Rating", [0, 1, 2, 3, 4, 5])
KPIs_met = st.selectbox("KPIs Met > 80%", ["Yes", "No"])
awards_won = st.selectbox("Awards Won?", ["Yes", "No"])
education = st.selectbox("Education Level", ["Bachelor’s", "Master’s & above", "Below Secondary"])
gender = st.selectbox("Gender", ["m", "f"])
region = st.selectbox("Region", [f"region_{i}" for i in range(1, 35)])  # 34 regions

# ---------------- CREATE INPUT DATAFRAME ---------------- #
input_data = pd.DataFrame({
    "region": [region],
    "education": [education],
    "gender": [gender],
    "no_of_trainings": [no_of_trainings],
    "age": [age],
    "previous_year_rating": [previous_year_rating],
    "length_of_service": [length_of_service],
    "KPIs_met >80%": [KPIs_met],
    "awards_won?": [awards_won],
    "avg_training_score": [avg_training_score]
})

# ---------------- ENSURE STRING TYPE FOR CATEGORICALS ---------------- #
input_data["KPIs_met >80%"] = input_data["KPIs_met >80%"].map({"Yes": 1, "No": 0})
input_data["awards_won?"] = input_data["awards_won?"].map({"Yes": 1, "No": 0})

# ---------------- PREDICTION ---------------- #
if st.button("Predict Promotion"):
    try:
        # Directly predict (model remembers which columns were categorical)
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]

        result = "🎉 Promoted" if prediction[0] == 1 else "❌ Not Promoted"
        st.success(f"Prediction: {result}")
        st.info(f"Probability of promotion: {prediction_proba*100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
