"""
ML Assignment 2 - Model Training Script
BITS Pilani WILP - M.Tech (AIML/DSE) - Machine Learning

Dataset: Telco Customer Churn (Kaggle - IBM sample dataset)
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
  ~7043 rows, 19 usable features (>=12 required), binary target 'Churn' (>=500 rows required)

Run this on the BITS Virtual Lab (take the required screenshot while it runs).

Before running:
  1. Download "WA_Fn-UseC_-Telco-Customer-Churn.csv" from the Kaggle link above.
  2. Place it at: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

What this script does:
  1. Loads and cleans the data
  2. Builds one sklearn Pipeline per model (preprocessing + classifier bundled together,
     so app.py never has to re-implement preprocessing)
  3. Trains all 5 required models on an 80/20 stratified split
  4. Computes Accuracy, AUC, Precision, Recall, F1, MCC for each model
  5. Saves each fitted pipeline to model/*.joblib
  6. Saves a comparison_table.csv (paste straight into your README)
  7. Saves a small test_data.csv sample (raw, unprocessed rows) for the
     GitHub repo and for uploading into the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "model"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# TotalCharges has some blank strings instead of numbers -> coerce + drop
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

df = df.drop(columns=["customerID"])

target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col].map({"Yes": 1, "No": 0})

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"Rows: {len(df)}  |  Features: {X.shape[1]}  |  Positive class rate: {y.mean():.3f}")

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Shared preprocessing (fit inside each model's pipeline separately so
#    every saved .joblib file is fully self-contained for the Streamlit app)
# ---------------------------------------------------------------------------
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE
    ),
}

# GaussianNB doesn't accept sparse input from OneHotEncoder -> densify for it
def build_pipeline(name, clf):
    pre = make_preprocessor()
    if name == "Naive Bayes":
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )
    return Pipeline(steps=[("preprocess", pre), ("classifier", clf)])

# ---------------------------------------------------------------------------
# 4. Train, evaluate, save
# ---------------------------------------------------------------------------
results = []
filename_map = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

for name, clf in models.items():
    pipe = build_pipeline(name, clf)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    joblib.dump(pipe, os.path.join(MODEL_DIR, filename_map[name]))

# ---------------------------------------------------------------------------
# 5. Save comparison table (paste into README.md)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("comparison_table.csv", index=False)
print("\nSaved comparison_table.csv")
print(results_df.to_markdown(index=False))

# ---------------------------------------------------------------------------
# 6. Save a small RAW test_data.csv sample (for repo + Streamlit upload demo)
#    Kept as raw/unprocessed columns -- app.py's saved pipelines handle
#    preprocessing internally, exactly like they will for grader-uploaded data.
# ---------------------------------------------------------------------------
sample = X_test.copy()
sample[target_col] = y_test.map({1: "Yes", 0: "No"}).values
sample = sample.sample(n=min(150, len(sample)), random_state=RANDOM_STATE)
sample.to_csv("test_data.csv", index=False)
print(f"\nSaved test_data.csv with {len(sample)} rows for the Streamlit app / repo")

# Save feature schema so app.py can validate uploads
schema = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target_col": target_col,
}
with open(os.path.join(MODEL_DIR, "schema.json"), "w") as f:
    json.dump(schema, f, indent=2)

print("\nDone. Copy model/*.joblib, schema.json, comparison_table.csv and test_data.csv")
print("into your project folder before pushing to GitHub.")
