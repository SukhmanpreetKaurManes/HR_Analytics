import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Promotion Prediction",
    page_icon="📈",
    layout="centered"
)


# ============================================================
# 2. LOAD MODEL AND PREPROCESSING
# ============================================================

model_package = joblib.load("promotion_model.pkl")
preprocessing = joblib.load("preprocessing.pkl")

model = model_package["model"]
threshold = model_package["threshold"]
cat_cols = model_package["cat_cols"]

feature_order = preprocessing["feature_order"]


# ============================================================
# 3. TITLE
# ============================================================

st.title("📈 Employee Promotion Prediction")

st.write(
    "Enter the employee's details below to predict whether "
    "the employee is likely to be promoted."
)

st.divider()


# ============================================================
# 4. USER INPUTS
# ============================================================

# ------------------------------------------------------------
# Categorical Features
# ------------------------------------------------------------

st.subheader("Employee Information")

department = st.selectbox(
    "Department",
    [
        "Sales & Marketing",
        "Operations",
        "Technology",
        "Analytics",
        "R&D",
        "Procurement",
        "Finance",
        "HR",
        "Legal"
    ]
)

region = st.text_input(
    "Region",
    placeholder="e.g. region_2"
)

education = st.selectbox(
    "Education",
    [
        "Bachelor's",
        "Master's & above",
        "Below Secondary"
    ]
)

gender = st.selectbox(
    "Gender",
    [
        "m",
        "f"
    ]
)

recruitment_channel = st.selectbox(
    "Recruitment Channel",
    [
        "sourcing",
        "other",
        "referred"
    ]
)


# ------------------------------------------------------------
# Numerical Features
# ------------------------------------------------------------

st.subheader("Employee Performance & Experience")

no_of_trainings = st.number_input(
    "Number of Trainings",
    min_value=1,
    max_value=20,
    value=1,
    step=1
)

age = st.number_input(
    "Age",
    min_value=20,
    max_value=60,
    value=30,
    step=1
)

previous_year_rating = st.selectbox(
    "Previous Year Rating",
    [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0
    ]
)

length_of_service = st.number_input(
    "Length of Service (years)",
    min_value=1,
    max_value=40,
    value=5,
    step=1
)

kpi = st.selectbox(
    "KPIs Met > 80%",
    [
        0,
        1
    ],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

awards = st.selectbox(
    "Awards Won?",
    [
        0,
        1
    ],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

avg_training_score = st.number_input(
    "Average Training Score",
    min_value=0,
    max_value=100,
    value=60,
    step=1
)


# ============================================================
# 5. PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "Predict Promotion",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validate region
    # --------------------------------------------------------

    if not region.strip():
        st.error("Please enter the employee's region.")
        st.stop()

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "department": department,
        "region": region.strip(),
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "KPIs_met >80%": kpi,
        "awards_won?": awards,
        "avg_training_score": avg_training_score
    }])


    # ========================================================
    # 6. PREPROCESSING
    # ========================================================

    # Handle missing values using the same values
    # used during model training

    input_data["education"] = input_data["education"].fillna(
        preprocessing["education_fill_value"]
    )

    input_data["previous_year_rating"] = input_data[
        "previous_year_rating"
    ].fillna(
        preprocessing["previous_year_rating_fill_value"]
    )


    # Make sure categorical columns are strings
    for col in cat_cols:
        input_data[col] = input_data[col].astype(str)


    # Ensure exactly the same feature order as training
    input_data = input_data[feature_order]


    # ========================================================
    # 7. PREDICTION
    # ========================================================

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)


    # ========================================================
    # 8. DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction Result")

    st.metric(
        "Promotion Probability",
        f"{probability:.2%}"
    )

    st.write(
        f"Classification threshold: **{threshold:.2f}**"
    )

    if prediction == 1:

        st.success(
            "🎉 The employee is likely to be promoted."
        )

    else:

        st.warning(
            "The employee is unlikely to be promoted."
        )


    # ========================================================
    # 9. SHOW INPUT DATA
    # ========================================================

    with st.expander("View Employee Information"):

        st.dataframe(
            input_data,
            use_container_width=True
        )