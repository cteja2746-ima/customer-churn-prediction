import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv(
    "Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("Customer Churn Prediction Dashboard")

st.write("""
This dashboard presents Exploratory Data Analysis,
Machine Learning model evaluation,
FastAPI integration details,
and final project insights.
""")

# ===================================================
# WEEK 1 - EDA
# ===================================================

st.header("Week 1 - Exploratory Data Analysis")

# Dataset Preview

st.subheader("Dataset Preview")

st.dataframe(df.head())

# Churn Distribution

st.subheader("Customer Churn Distribution")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    x="Churn",
    data=df,
    ax=ax
)

plt.title("Customer Churn Distribution")

st.pyplot(fig)

# Gender vs Churn

st.subheader("Gender vs Churn")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    x="gender",
    hue="Churn",
    data=df,
    ax=ax
)

plt.title("Gender vs Churn")

st.pyplot(fig)

# Contract Type vs Churn

st.subheader("Contract Type vs Churn")

fig, ax = plt.subplots(figsize=(8,5))

sns.countplot(
    x="Contract",
    hue="Churn",
    data=df,
    ax=ax
)

plt.xticks(rotation=20)

plt.title("Contract Type vs Churn")

st.pyplot(fig)

# Internet Service vs Churn

st.subheader("Internet Service vs Churn")

fig, ax = plt.subplots(figsize=(8,5))

sns.countplot(
    x="InternetService",
    hue="Churn",
    data=df,
    ax=ax
)

plt.title("Internet Service vs Churn")

st.pyplot(fig)

# Monthly Charges vs Churn

st.subheader("Monthly Charges vs Churn")

fig, ax = plt.subplots(figsize=(8,5))

sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df,
    ax=ax
)

plt.title("Monthly Charges vs Churn")

st.pyplot(fig)

# Tenure Distribution

st.subheader("Tenure Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    df["tenure"],
    bins=30,
    kde=True,
    ax=ax
)

plt.title("Customer Tenure Distribution")

st.pyplot(fig)

# Correlation Heatmap

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(
    include=['int64', 'float64']
)

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=True,
    ax=ax
)

plt.title("Correlation Heatmap")

st.pyplot(fig)

# ===================================================
# WEEK 2 - MODEL EVALUATION
# ===================================================

st.header("Week 2 - Machine Learning Model Evaluation")

# Accuracy Comparison

st.subheader("Model Accuracy Comparison")

st.image(
    "images/accuracy_comparison.png",
    caption="Accuracy Comparison"
)

# Confusion Matrix

st.subheader("Confusion Matrix")

st.image(
    "images/confusion_matrix.png",
    caption="Random Forest Confusion Matrix"
)

# SHAP Analysis

st.subheader("SHAP Feature Importance")

st.image(
    "images/shap_feature_importance.png",
    caption="SHAP Feature Importance"
)

# Feature Importance

st.subheader("Feature Importance")

feature_names = [
    "TotalCharges",
    "tenure",
    "MonthlyCharges",
    "AvgMonthlyValue",
    "InternetService_Fiber optic"
]

importance_scores = [
    0.15,
    0.14,
    0.13,
    0.13,
    0.04
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_scores
})

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df,
    ax=ax
)

plt.title("Top Feature Importance")

st.pyplot(fig)

# ===================================================
# WEEK 3 - FASTAPI
# ===================================================

st.header("Week 3 - FastAPI Integration")

st.write("""
FastAPI was integrated successfully
for real-time customer churn prediction.
""")

st.subheader("Available API Endpoints")

st.code("""
GET /

POST /predict_churn

POST /predict_ltv

POST /batch_predict
""")

st.subheader("Swagger Documentation")

st.write("""
Swagger API documentation is available at:
http://127.0.0.1:8000/docs
""")

# ===================================================
# WEEK 4 - FINAL RESULTS
# ===================================================

st.header("Week 4 - Final Dashboard and Insights")

# Metrics

st.subheader("Project Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dataset Rows",
    "7043"
)

col2.metric(
    "Models Used",
    "3"
)

col3.metric(
    "Best Accuracy",
    "88%"
)

# Final Insights

st.subheader("Business Insights")

st.write("""
• Customers with higher monthly charges are more likely to churn.

• Customers using month-to-month contracts show higher churn probability.

• Fiber optic internet users have higher churn rates.

• Technical support and long-term contracts improve customer retention.

• XGBoost achieved the highest prediction performance.
""")

# Final Conclusion

st.subheader("Conclusion")

st.write("""
The project successfully analyzed customer behavior
and predicted customer churn using Machine Learning models.
FastAPI integration and dashboard visualization
improved the usability and presentation of the project.
""")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.write("---")

st.write("Customer Churn Prediction Project")

st.write("""
Developed using Python,
Machine Learning,
FastAPI,
Streamlit,
and PostgreSQL
""")