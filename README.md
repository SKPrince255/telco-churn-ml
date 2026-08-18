# Telco Customer Churn Prediction

**Student:** RamaDeepak &nbsp;·&nbsp; **BITS ID:** 2025AD05068 &nbsp;·&nbsp; **Subject:** Machine Learning &nbsp;·&nbsp; **Assignment 2**

**Live App:** https://telco-churn-ml-ecbf4eal4vgsh3xhzeuxqt.streamlit.app/
**GitHub:** https://github.com/SKPrince255/telco-churn-ml

---

## a. Problem Statement

Telecom providers spend far more acquiring a new customer than retaining an
existing one, which makes early churn detection directly valuable to the
business. This project trains and compares five classifiers to predict
whether a customer will churn in their next billing cycle, based on their
account history and the services they subscribe to, so that a retention team
could realistically prioritize outreach toward the customers most likely to
leave.

Two things shape how the models here are evaluated rather than just how
accurate they are on the surface. First, churners are a minority of the
dataset (roughly 1 in 4 customers), so a model that just predicts "No churn"
every time would already score ~73% accuracy without being useful — which is
why Precision, Recall, F1, AUC and MCC are reported for every model instead
of accuracy alone. Second, in a real retention scenario, missing a customer
who was about to leave is more expensive than wrongly flagging a loyal one
for a check-in call — so Recall carries more practical weight than Precision
for this specific problem.

## b. Dataset Description

The dataset used is the **Telco Customer Churn** dataset (IBM's public sample
data), sourced from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

After removing the non-predictive `customerID` column and ~11 rows with blank `TotalCharges` values, the working dataset has **7,032 rows** and **19 features** — well above the assignment's minimums of 500 rows / 12 features. The target column is `Churn`, encoded as `Yes`/`No`.

The 19 features split naturally into three groups. A handful describe who the
customer *is* — `gender`, `SeniorCitizen`, `Partner`, `Dependents`. A second
set describes their *account* — `tenure`, `Contract` type, `PaperlessBilling`,
`PaymentMethod`, `MonthlyCharges`, and `TotalCharges`. The remaining nine
columns describe *what they've subscribed to* — phone and internet service,
and add-ons like `OnlineSecurity`, `TechSupport`, `StreamingTV`, and
`StreamingMovies`. Numeric columns are standardized and categorical columns
are one-hot encoded, both handled inside each model's sklearn `Pipeline` so
preprocessing never has to be reimplemented separately in the Streamlit app.

## c. GitHub Repository Link

https://github.com/SKPrince255/telco-churn-ml

## d. Models Used

All five models below are trained on the identical 80/20 stratified
train/test split, so the comparison table that follows is apples-to-apples.

| Model | Category | Core Idea |
|---|---|---|
| Logistic Regression | Linear / probabilistic | Fits a linear decision boundary and passes it through a sigmoid to get a churn probability |
| Decision Tree | Rule-based | Recursively splits customers (e.g. by contract, then tenure) to separate the two classes |
| K-Nearest Neighbors (k=15) | Instance-based | Classifies a customer by majority vote among its 15 nearest neighbors in feature space |
| Gaussian Naive Bayes | Probabilistic (Bayesian) | Applies Bayes' theorem, assuming features are independent given the class |
| Random Forest (Ensemble) | Bagging ensemble | Averages 300 de-correlated trees trained on bootstrap samples to reduce variance over a single tree |

### Test-Set Results by Model

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8038 | 0.8359 | 0.6485 | 0.5722 | 0.6080 | 0.4795 |
| Decision Tree | 0.7690 | 0.7919 | 0.5679 | 0.5481 | 0.5578 | 0.4017 |
| K-Nearest Neighbors | 0.7733 | 0.8155 | 0.5741 | 0.5695 | 0.5718 | 0.4176 |
| Naive Bayes | 0.6823 | 0.8049 | 0.4472 | 0.8262 | 0.5803 | 0.4033 |
| Random Forest | 0.7946 | 0.8305 | 0.6403 | 0.5187 | 0.5731 | 0.4441 |

*Test set: 1,407 customers (20% stratified holdout), 26.6% actual churn rate.*

### Reading the Results

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.836) and MCC (0.480) of the five, with the most balanced Precision/Recall. Indicates churn is close to linearly separable on this feature set — contract type, tenure, and monthly charges behave as strong, roughly linear signals. |
| Decision Tree | Lowest AUC (0.792) and MCC (0.402) of all models. A single tree at this depth tends to fit noise in the training split rather than generalizable structure, without an ensemble's variance reduction. |
| kNN | Mid-table on every metric. Distance-based voting is diluted in a mostly one-hot encoded, high-dimensional space, where "nearness" carries less signal than on purely numeric features. |
| Naive Bayes | Lowest accuracy (0.682) but the highest Recall by a wide margin (0.826) — it over-flags churn, catching most real churners at the cost of many false alarms. This tracks with its independence assumption being violated, since the service add-on columns are strongly correlated with each other. |
| Random Forest | Second-best on Accuracy (0.795) and AUC (0.831), but its Recall (0.519) is the lowest of all five — it only flags churn when fairly confident, trading missed churners for fewer false alarms. |
| **Overall Winner** | **Logistic Regression**, on balance of AUC and MCC. If the retention team's real priority is *catching* every possible churner even at the cost of extra false positives, **Naive Bayes** is the more defensible pick despite its lower accuracy. |


