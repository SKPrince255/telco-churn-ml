# Customer Churn Prediction — ML Assignment 2

## a. Problem Statement
Telecom companies lose significant revenue to customer churn. This project builds
and compares 5 classification models to predict whether a customer will churn
(`Churn = Yes/No`) based on their account and service usage attributes, so that
at-risk customers can be identified and targeted with retention offers.

## b. Dataset Description
- **Source:** Telco Customer Churn — Kaggle (IBM sample dataset)
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Rows:** ~7,043 customers (after removing rows with missing `TotalCharges`)
- **Features:** 19 (exceeds the minimum of 12) — mix of numeric
  (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) and categorical
  (contract type, internet service, payment method, add-on services, etc.)
- **Target:** `Churn` — binary (`Yes` / `No`)
- **Preprocessing:** dropped `customerID`; coerced `TotalCharges` to numeric and
  dropped the ~11 blank rows; `StandardScaler` on numeric features,
  `OneHotEncoder` on categorical features (bundled into each model's sklearn
  `Pipeline`, so no separate preprocessing step is needed at inference time).

## c. GitHub Repository Link
<!-- TODO: paste your repo URL here, e.g. https://github.com/<you>/ml-assignment-2 -->

## d. Models Used

### Comparison Table
<!-- TODO: paste the contents of comparison_table.csv here as a markdown table.
     Run: python train_models.py   -- it prints this table for you. -->

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <!-- TODO --> |
| Decision Tree | <!-- TODO --> |
| kNN | <!-- TODO --> |
| Naive Bayes | <!-- TODO --> |
| Random Forest (Ensemble) | <!-- TODO --> |
| **Overall Winner for your dataset?** | <!-- TODO --> |

*(Suggested talking points once you have real numbers: which model has the
highest AUC/MCC — MCC is the most reliable single metric on imbalanced data
like churn; whether Naive Bayes underperforms due to its independence
assumption; whether the Decision Tree overfits vs. the Random Forest;
whether kNN suffers from the high-dimensional one-hot-encoded feature space.)*

## Live App
- **Streamlit App Link:** <!-- TODO: paste after deploying on Streamlit Community Cloud -->

## Repository Structure
```
project-folder/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── comparison_table.csv
├── test_data.csv
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   (not committed if too large — optional)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── schema.json
```

## How to Run Locally
```bash
pip install -r requirements.txt
python train_models.py     # trains all 5 models, saves them + comparison_table.csv + test_data.csv
streamlit run app.py       # launches the demo app
```
