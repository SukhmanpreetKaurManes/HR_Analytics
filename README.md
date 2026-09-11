# Employee Promotion Prediction

## 📌 Project Overview

Employee promotion is an important decision in an organization. This project uses machine learning to predict whether an employee is likely to be promoted based on their performance, training, experience, and other employee-related attributes.

The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis, model comparison, handling class imbalance, model evaluation, probability threshold tuning, and deployment using Streamlit.

---

## 🎯 Objective

The main objective of this project is to build a classification model that predicts:

- `1` → Employee is likely to be promoted
- `0` → Employee is unlikely to be promoted

The model can help HR teams identify employees who may be suitable for promotion based on historical employee data.

---

## 📊 Dataset

The dataset contains approximately **54,800 employee records** with 14 attributes.

### Features

| Feature | Description |
|---|---|
| `employee_id` | Unique employee identifier |
| `department` | Employee's department |
| `region` | Region where the employee works |
| `education` | Employee's education level |
| `gender` | Employee's gender |
| `recruitment_channel` | Recruitment source |
| `no_of_trainings` | Number of trainings completed |
| `age` | Employee's age |
| `previous_year_rating` | Rating from the previous year |
| `length_of_service` | Number of years of service |
| `KPIs_met >80%` | Whether more than 80% of KPIs were met |
| `awards_won?` | Whether the employee won an award |
| `avg_training_score` | Average training score |
| `is_promoted` | Target variable |

---

## ⚠️ Class Imbalance

One of the major challenges in the dataset is class imbalance.

Approximately:

- **91.5%** employees were not promoted
- **8.5%** employees were promoted

Because of this imbalance, accuracy alone is not a suitable metric.

Therefore, the project focuses primarily on:

- Precision
- Recall
- F1-score
- ROC-AUC

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

1. Removed `employee_id` from the model features because it is an identifier.
2. Handled missing values in `education`.
3. Handled missing values in `previous_year_rating`.
4. Used an **80-20 stratified train-validation split**.
5. Categorical features were handled natively by CatBoost.

Categorical features used by the final model include:

- Department
- Region
- Education
- Gender
- Recruitment Channel

---

## 🤖 Models Evaluated

Several machine learning algorithms were experimented with:

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Additional experiments included:

- Class weighting
- Probability threshold tuning
- Feature engineering
- Feature selection
- Ensemble modeling
- Synthetic data augmentation experiments

After comparing the approaches, **CatBoost** was selected as the final model.

---

## 🏆 Final Model

The final model is a `CatBoostClassifier`.

### Model Configuration

```python
CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=10,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42
)
