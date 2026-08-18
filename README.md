# Telco Customer Churn Prediction — ML Assignment 2

## Metadata
| Author | BITS ID | Subject |
|---|---|---|
| RamaDeepak | 2025AD05068 | Machine Learning |

**Live Link:** https://telco-churn-ml-ecbf4eal4vgsh3xhzeuxqt.streamlit.app/

## A. Problem Statement
**Objective:** Predict whether a telecom customer will churn (`Churn = Yes` vs `No`)
in the coming billing cycle, using their demographic profile, account details, and
the services they've subscribed to.

**Business Impact:** Customer acquisition costs far more than retention. Flagging
at-risk customers early lets a retention team intervene with targeted offers,
contract upgrades, or support outreach before the customer actually leaves.

**Key Challenges:**
- **Class Imbalance:** Non-churning customers outnumber churners roughly 3:1
  (~73% stay vs. ~27% churn), so raw accuracy is misleading — Precision, Recall,
  F1, AUC and MCC all need to be reported together.
- **Cost Asymmetry:** Missing a customer who was about to churn (False Negative)
  is more costly than flagging a loyal customer for a retention call they didn't
  need (False Positive) — so Recall matters more than Precision in practice here.

## B. Dataset Description
- **Dataset:** Telco Customer Churn (IBM sample dataset)
- **Source:** Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Target Variable:** `Churn` (`Yes` = 1, `No` = 0)
- **Total Features:** 19 (after dropping the non-predictive `customerID`)
- **Rows:** ~7,032 (after removing ~11 rows with blank `TotalCharges`)

### Feature Grouping
| Feature Category | Column Names | Description |
|---|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Basic customer profile |
| Account Info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | How long they've stayed, how they're billed, and what they pay |
| Services Subscribed | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Which add-on services the customer has active |

## C. GitHub Repository Link
https://github.com/SKPrince255/telco-churn-ml

## D. Models Used
Five classical supervised classifiers are benchmarked on an identical 80/20
stratified train/test split.

**i. Logistic Regression**
Model Type: Probabilistic Linear Model.
Description: Fits a linear decision boundary and maps it to a churn probability
via the sigmoid function.
Preprocessing: Numeric features standardized (`StandardScaler`); categoricals one-hot encoded.

**ii. Decision Tree Classifier**
Model Type: Non-parametric, rule-based model.
Description: Recursively splits customers into groups (e.g. by contract type,
then tenure) to maximize class separation at each split.
Preprocessing: Works directly on encoded features; no scaling required, but the
same pipeline is used for consistency.

**iii. K-Nearest Neighbors**
Model Type: Instance-based / distance classifier.
Description: Classifies a customer by majority vote among their k=15 nearest
neighbors in the (scaled, one-hot encoded) feature space.
Preprocessing: Sensitive to feature scale — standardization is essential here.

**iv. Gaussian Naive Bayes**
Model Type: Probabilistic Bayesian classifier.
Description: Applies Bayes' theorem assuming features are conditionally
independent given the class — a strong assumption given how correlated the
service add-on columns are with each other.
Preprocessing: Standardized numeric features; one-hot encoded categoricals
(densified, since GaussianNB doesn't accept sparse input).

**v. Random Forest (Ensemble)**
Model Type: Bagging ensemble of decision trees.
Description: Averages predictions across 300 de-correlated trees, each trained
on a bootstrap sample with random feature subsets, to reduce variance versus a
single tree.
Preprocessing: Same encoded feature set as the other models; tree ensembles
don't strictly need scaling but it's included for pipeline consistency.

### Evaluation Metrics Across Models

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8038 | 0.8359 | 0.6485 | 0.5722 | 0.6080 | 0.4795 |
| Decision Tree | 0.7690 | 0.7919 | 0.5679 | 0.5481 | 0.5578 | 0.4017 |
| K-Nearest Neighbors | 0.7733 | 0.8155 | 0.5741 | 0.5695 | 0.5718 | 0.4176 |
| Naive Bayes | 0.6823 | 0.8049 | 0.4472 | 0.8262 | 0.5803 | 0.4033 |
| Random Forest | 0.7946 | 0.8305 | 0.6403 | 0.5187 | 0.5731 | 0.4441 |

*(Dataset: 7,032 rows, 19 features, 26.6% positive/churn class rate — 80/20 stratified split.)*

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Best all-round performer — highest AUC (0.836) and MCC (0.480) of all five models, with a reasonably balanced Precision/Recall trade-off. This suggests churn is close to linearly separable on this feature set (contract type, tenure, and monthly charges are strong, roughly linear signals). |
| Decision Tree | Weakest model overall — lowest AUC (0.792) and MCC (0.402). A single unpruned-depth tree tends to fit noise in the training split rather than generalizable patterns, without the variance reduction an ensemble provides. |
| K-Nearest Neighbors | Middling performance across the board. Distance-based classification is diluted by the high-dimensional, mostly-binary one-hot encoded feature space, where "nearness" is less meaningful than it would be on purely numeric data. |
| Naive Bayes | Lowest Accuracy (0.682) but by far the highest Recall (0.826) — it over-predicts churn, catching most true churners at the cost of many false positives. This lines up with its core assumption (feature independence given the class), which is violated here since the service add-on columns are strongly correlated with each other. |
| Random Forest | Second-best on Accuracy (0.795) and AUC (0.831), but its Recall (0.519) is the lowest of all models — it plays conservative, only flagging churn when fairly confident, which lowers false positives but misses more actual churners than Logistic Regression or Naive Bayes. |
| **Overall Winner** | **Logistic Regression** — best MCC and AUC, indicating the most reliable overall separation between churners and non-churners on this dataset. However, if the business goal is to minimize *missed* churners (Recall matters more than Precision), **Naive Bayes** would be the more defensible choice despite its lower accuracy. |

## Repository Structure
```
project-folder/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── comparison_table.csv
├── test_data.csv
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
