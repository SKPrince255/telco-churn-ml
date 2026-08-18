"""
ML Assignment 2 - Streamlit Demo App
BITS Pilani WILP - M.Tech (AIML/DSE) - Machine Learning

Required features implemented:
  a. Dataset upload option (CSV)               -> sidebar uploader
  b. Model selection dropdown                   -> sidebar selectbox
  c. Display of evaluation metrics               -> metrics section
  d. Confusion matrix / classification report    -> plotted + printed
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Telco Churn - Model Comparison", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest.joblib",
}

with open("model/schema.json") as f:
    SCHEMA = json.load(f)
TARGET_COL = SCHEMA["target_col"]


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("Customer Churn Prediction - Multi-Model Comparison")
st.caption(
    "Assignment 2 · Machine Learning · BITS Pilani WILP M.Tech (AIML/DSE) · "
    "Dataset: Telco Customer Churn (Kaggle)"
)

# --- Sidebar: (a) dataset upload, (b) model selection ----------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

use_default = False
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No file uploaded — using bundled test_data.csv sample.")
    data = pd.read_csv("test_data.csv")
    use_default = True

st.subheader("Preview of test data")
st.dataframe(data.head(10), use_container_width=True)

# --- Load selected model -----------------------------------------------------
pipe = load_model(MODEL_FILES[model_name])

has_labels = TARGET_COL in data.columns
if has_labels:
    y_true = data[TARGET_COL].map({"Yes": 1, "No": 0})
    X_input = data.drop(columns=[TARGET_COL])
else:
    X_input = data

# --- Predict -----------------------------------------------------------------
try:
    y_pred = pipe.predict(X_input)
    y_proba = pipe.predict_proba(X_input)[:, 1]
except Exception as e:
    st.error(
        "Could not run predictions on the uploaded file. Make sure it has the "
        f"same columns as the training data (minus the target). Error: {e}"
    )
    st.stop()

result_df = data.copy()
result_df["Predicted_Churn"] = np.where(y_pred == 1, "Yes", "No")
result_df["Churn_Probability"] = np.round(y_proba, 3)

st.subheader(f"Predictions — {model_name}")
st.dataframe(result_df.head(20), use_container_width=True)

# --- (c) Evaluation metrics ---------------------------------------------------
if has_labels:
    st.subheader("Evaluation metrics")
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

    # --- (d) Confusion matrix + classification report -------------------------
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, target_names=["No", "Yes"])
    st.code(report)
else:
    st.info(
        "Uploaded file has no 'Churn' column, so only predictions are shown "
        "(no metrics/confusion matrix — those require ground-truth labels)."
    )

st.divider()
st.caption(
    "Tip: switch the model in the sidebar to compare Accuracy / AUC / Precision / "
    "Recall / F1 / MCC across Logistic Regression, Decision Tree, kNN, Naive Bayes "
    "and Random Forest on the same test data."
)
