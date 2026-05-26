## Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning based project developed to identify customers who are likely to discontinue telecom services. The project analyzes customer demographics, subscription details, service usage patterns, billing information, and customer behavior to predict churn effectively. The primary goal of this project is to help businesses improve customer retention strategies and reduce revenue loss caused by customer attrition.The project was implemented using Python and integrated with FastAPI to provide real-time prediction APIs. Multiple Machine Learning models were trained, evaluated, and compared to achieve better prediction performance and business insights.

## Technologies Used
Python
PostgreSQL
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
Matplotlib
Seaborn
FastAPI
Machine Learning Models Implemented

## The following Machine Learning algorithms were used in this project:

Logistic Regression
Random Forest Classifier
XGBoost Classifier
These models were trained and evaluated using customer churn data, and their performances were compared using accuracy scores, confusion matrices, and classification reports.

## Key Features

Data preprocessing and cleaning
Exploratory Data Analysis (EDA)
Feature engineering
Model training and evaluation
Accuracy comparison between models
Confusion matrix visualization
SHAP feature importance analysis
FastAPI integration
Batch prediction support
Customer Lifetime Value prediction

## Project Workflow

Data Ingestion
Data Cleaning and Preprocessing
Exploratory Data Analysis
Feature Engineering
Model Training
Model Evaluation
SHAP Analysis
API Development using FastAPI
Final Testing and Validation
API Endpoints

## The FastAPI application provides the following endpoints:

1. Predict Customer Churn
POST /predict_churn
2. Predict Customer Lifetime Value
POST /predict_ltv
3. Batch Prediction
POST /batch_predict

## Swagger API documentation can be accessed using:

http://127.0.0.1:8000/docs


## Project Structure

customer-churn-prediction/
│
|
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
├── data_ingestion.py
├── model_evaluation.py
├── main.py
├── week1_eda.ipynb
├── week2_eda.ipynb
├── requirements.txt
└── README.md

## How to Run the Project

Step 1: Clone the Repository
git clone <repository-link>
cd customer-churn-prediction
Step 2: Install Required Dependencies
pip install -r requirements.txt
Step 3: Run the FastAPI Application
uvicorn main:app --reload
Step 4: Open Swagger Documentation

## Open the following URL in your browser:

http://127.0.0.1:8000/docs
Project Results
Successfully trained and evaluated multiple Machine Learning models
Random Forest and XGBoost achieved strong prediction performance
SHAP analysis identified important features influencing customer churn
FastAPI endpoints tested successfully
Batch prediction functionality implemented successfully

## Future Scope

Deploy the project using Render or Railway
Build an interactive dashboard using Streamlit or Superset
Add real-time customer prediction support
Improve model accuracy through advanced hyperparameter tuning
Integrate cloud deployment and monitoring

## Team Members

Perumalla Charan Teja
Prabhas
Prasanna Jampa
Sirisha Yerra



## Dashboard

A Streamlit dashboard was developed to visualize customer churn insights and model evaluation results.

Dashboard Features:
- Customer churn distribution
- Contract type vs churn
- Monthly charges analysis
- Internet service analysis
- Correlation heatmap
- Model accuracy comparison
- SHAP feature importance
- FastAPI integration overview

Run the dashboard using:

streamlit run dashboard.py
