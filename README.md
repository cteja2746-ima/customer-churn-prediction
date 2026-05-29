# Customer Churn Prediction Using Machine Learning

## Project Overview

Customer Churn Prediction is a Machine Learning-based project developed to identify customers who are likely to discontinue telecom services. The project analyzes customer demographics, subscription details, service usage patterns, billing information, and customer behavior to predict churn effectively.

The primary goal of this project is to help businesses improve customer retention strategies and reduce revenue loss caused by customer attrition.

The project was implemented using Python and integrated with FastAPI to provide real-time prediction APIs. Multiple Machine Learning models were trained, evaluated, and compared to achieve better prediction performance and business insights.

---

## Technologies Used

* Python
* PostgreSQL
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* FastAPI
* Streamlit

---

## Machine Learning Models Implemented

The following Machine Learning algorithms were used in this project:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

These models were trained and evaluated using customer churn data, and their performances were compared using accuracy scores, confusion matrices, and classification reports.

---

## Key Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training and evaluation
* Accuracy comparison between models
* Confusion matrix visualization
* SHAP feature importance analysis
* FastAPI integration
* Batch prediction support
* Customer Lifetime Value (LTV) prediction
* Interactive Streamlit dashboard

---

## Project Workflow

1. Data Ingestion
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. SHAP Analysis
8. API Development using FastAPI
9. Dashboard Development using Streamlit
10. Final Testing and Validation

---

## API Endpoints

The FastAPI application provides the following endpoints:

### Predict Customer Churn

POST `/predict_churn`

### Predict Customer Lifetime Value

POST `/predict_ltv`

### Batch Prediction

POST `/batch_predict`

### Swagger Documentation

After running FastAPI, Swagger API documentation can be accessed using:

```bash
http://127.0.0.1:8000/docs
```

---

## Dashboard

A Streamlit dashboard was developed to visualize customer churn insights and model evaluation results.

### Dashboard Features

* Customer churn distribution
* Contract type vs churn
* Monthly charges analysis
* Internet service analysis
* Correlation heatmap
* Model accuracy comparison
* SHAP feature importance
* FastAPI integration overview
* Business insights and conclusions

### Run Dashboard

```bash
streamlit run dashboard.py
```

---

## Project Structure

```text
customer-churn-prediction/
│
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── images/
│   ├── accuracy_comparison.png
│   ├── confusion_matrix.png
│   └── shap_feature_importance.png
│
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── model_features.pkl
│
├── presentation/
│   └── Customer_Churn_Prediction_PPT.pptx
│
├── documentation/
│   └── Customer_Churn_Prediction_Internship_Report.pdf
│
├── dashboard.py
├── main.py
├── data_ingestion.py
├── model_evaluation.py
├── week1_eda.ipynb
├── week2_eda.ipynb
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <repository-link>
cd customer-churn-prediction
```

### Step 2: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the FastAPI Application

```bash
uvicorn main:app --reload
```

### Step 4: Open Swagger Documentation

```bash
http://127.0.0.1:8000/docs
```

### Step 5: Run Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

## Project Results

* Successfully trained and evaluated multiple Machine Learning models
* XGBoost achieved the best prediction performance
* SHAP analysis identified important features influencing customer churn
* FastAPI endpoints tested successfully
* Batch prediction functionality implemented successfully
* Interactive dashboard developed for visualization and business insights

---

## Business Insights

* Customers with month-to-month contracts show higher churn rates.
* Customers with higher monthly charges are more likely to churn.
* Fiber optic internet users exhibit higher churn probability.
* Longer customer tenure is associated with lower churn rates.
* Customer retention strategies can significantly reduce revenue loss.

---

## Future Scope

* Deploy the project using Streamlit Cloud, Render, or Railway
* Add real-time customer prediction support
* Enhance dashboard with advanced analytics
* Improve model accuracy through hyperparameter tuning
* Integrate cloud monitoring and logging
* Implement automated customer retention recommendations

---

## Project Contributors

* Perumalla Charan Teja – Project Coordination, Dashboard Development, GitHub Management, Deployment Support
* Prabhas Bollam – Documentation, Project Report Preparation, and Week 2 Model Evaluation
* Prasanna Jampa – SHAP Analysis, Presentation Design, and Project PPT Preparation
* Sirisha Yerra – FastAPI Integration, API Testing, and Project Support
* Keerthi – Team Member

---

## Internship Details

**Company:** Zaalima Development Private Limited

**Internship Role:** Data Analytics Intern

**Project:** Customer Churn Prediction Using Machine Learning

**Internship Duration:** May 2026 – July 2026

---

## Conclusion

This project successfully predicts customer churn using Machine Learning techniques and provides actionable business insights through FastAPI APIs and an interactive Streamlit dashboard.

The solution helps organizations proactively identify customers at risk of leaving and supports data-driven customer retention strategies.
