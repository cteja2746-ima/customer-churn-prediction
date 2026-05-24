import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# LOAD DATASET
df = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# DATA PREPROCESSING
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove null values
df.dropna(inplace=True)

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# TARGET VARIABLE
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LOGISTIC REGRESSION
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# RANDOM FOREST
rf_model = RandomForestClassifier(random_state = 42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# XGBOOST MODEL

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',random_state = 42)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBoost Accuracy:", xgb_accuracy)

# ACCURACY SCORES
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nLogistic Regression Accuracy:", lr_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# CONFUSION MATRIX
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()
plt.close()

# ACCURACY COMPARISON CHART
models = ["Logistic Regression", "Random Forest", "XGBoost"]

accuracies = [lr_accuracy, rf_accuracy, xgb_accuracy]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("accuracy_comparison.png")
plt.show()
plt.close()

# CLASSIFICATION REPORT
print("\nXGBoost Classification Report:\n")
print(classification_report(y_test, xgb_pred))

import shap

# SHAP ANALYSIS

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_test)

print("\nTop Churn Features based on SHAP:\n")

shap.summary_plot(shap_values, X_test,show=False)

plt.savefig("shap_feature_importance.png")
plt.show()
plt.close()

best_accuracy = max(
    lr_accuracy,
    rf_accuracy,
    xgb_accuracy
)

if best_accuracy == lr_accuracy:
    print("\nBest Model: Logistic Regression")

elif best_accuracy == rf_accuracy:
    print("\nBest Model: Random Forest")

else:
    print("\nBest Model: XGBoost")

print("Best Accuracy:", best_accuracy)

