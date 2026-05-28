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
and final business insights.
""")

# ===================================================
# WEEK 1 - EDA
# ===================================================

st.header("Week 1 - Exploratory Data Analysis")

# Dataset Preview

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ===================================================
# ROW 1
# ===================================================

col1, col2 = st.columns(2)

# Churn Distribution

with col1:

    st.subheader("Customer Churn Distribution")

    fig, ax = plt.subplots(figsize=(5,3))

    sns.countplot(
        x="Churn",
        data=df,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "A significant number of customers have churned, indicating customer retention challenges."
    )

# Gender vs Churn

with col2:

    st.subheader("Gender vs Churn")

    fig, ax = plt.subplots(figsize=(5,3))

    sns.countplot(
        x="gender",
        hue="Churn",
        data=df,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "Customer churn distribution is almost similar across genders."
    )

# ===================================================
# ROW 2
# ===================================================

col3, col4 = st.columns(2)

# Contract Type vs Churn

with col3:

    st.subheader("Contract Type vs Churn")

    fig, ax = plt.subplots(figsize=(5,3))

    sns.countplot(
        x="Contract",
        hue="Churn",
        data=df,
        ax=ax
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "Customers with month-to-month contracts show higher churn rates."
    )

# Internet Service vs Churn

with col4:

    st.subheader("Internet Service vs Churn")

    fig, ax = plt.subplots(figsize=(5,3))

    sns.countplot(
        x="InternetService",
        hue="Churn",
        data=df,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "Fiber optic customers show higher churn probability compared to DSL users."
    )

# ===================================================
# ROW 3
# ===================================================

col5, col6 = st.columns(2)

# Monthly Charges vs Churn

with col5:

    st.subheader("Monthly Charges vs Churn")

    fig, ax = plt.subplots(figsize=(5,3))

    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "Customers with higher monthly charges are more likely to churn."
    )

# Tenure Distribution

with col6:

    st.subheader("Tenure Distribution")

    fig, ax = plt.subplots(figsize=(5,3))

    sns.histplot(
        df["tenure"],
        bins=30,
        kde=True,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "Customers with longer tenure are more likely to stay with the company."
    )

# ===================================================
# HEATMAP
# ===================================================

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(
    include=['int64', 'float64']
)

fig, ax = plt.subplots(figsize=(7,4))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=True,
    ax=ax
)

plt.tight_layout()

st.pyplot(fig)

st.info(
    "The heatmap shows relationships between numerical features in the dataset."
)

# ===================================================
# WEEK 2 - MODEL EVALUATION
# ===================================================

st.header("Week 2 - Machine Learning Model Evaluation")

col7, col8 = st.columns(2)

# Accuracy Comparison

with col7:

    st.subheader("Accuracy Comparison")

    st.image(
        "images/accuracy_comparison.png",
        use_container_width=True
    )

    st.info(
        "XGBoost achieved the highest prediction accuracy among all models."
    )

# Confusion Matrix

with col8:

    st.subheader("Confusion Matrix")

    st.image(
        "images/confusion_matrix.png",
        use_container_width=True
    )

    st.info(
        "The confusion matrix evaluates the classification performance of the Random Forest model."
    )

# SHAP Analysis

st.subheader("SHAP Feature Importance")

st.image(
    "images/shap_feature_importance.png",
    use_container_width=True
)

st.info(
    "SHAP analysis identifies the most important features affecting customer churn."
)

# ===================================================
# FEATURE IMPORTANCE
# ===================================================

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

fig, ax = plt.subplots(figsize=(6,3))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df,
    ax=ax
)

plt.tight_layout()

st.pyplot(fig)

st.info(
    "Billing details and customer tenure are among the most influential churn factors."
)

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

st.info(
    "FastAPI enables real-time prediction and batch processing functionalities."
)

# ===================================================
# WEEK 4 - FINAL RESULTS
# ===================================================

st.header("Week 4 - Final Dashboard and Insights")

# Metrics

col9, col10, col11 = st.columns(3)

col9.metric(
    "Dataset Rows",
    "7043"
)

col10.metric(
    "Models Used",
    "3"
)

col11.metric(
    "Best Accuracy",
    "88%"
)

# Business Insights

st.subheader("Business Insights")

st.write("""
• Customers with higher monthly charges are more likely to churn.

• Customers using month-to-month contracts show higher churn probability.

• Fiber optic internet users have higher churn rates.

• Technical support and long-term contracts improve customer retention.

• XGBoost achieved the highest prediction performance.
""")

# Conclusion

st.subheader("Conclusion")

st.write("""
The project successfully analyzed customer behavior
and predicted customer churn using Machine Learning models.

FastAPI integration and dashboard visualization
improved usability, analytics, and business insights.
""")

# ===================================================
# FOOTER
# ===================================================

st.write("---")

st.write("Customer Churn Prediction Project")

st.write("""
Developed using Python,
Machine Learning,
FastAPI,
Streamlit,
and PostgreSQL
""")