# Customer Churn Prediction | Machine Learning Portfolio Project

A reproducible, end-to-end **binary classification** project predicting subscription cancellations from customer behavior. Built with Python and scikit-learn.

> **Data integrity:** All 2,500 records are **synthetic**, generated for this portfolio demonstration. Model performance does not establish real-world effectiveness, and the synthetic target was generated using known assumptions.

## Business question

Which customers are at risk of cancelling, and what factors are associated with churn? A business could use a validated model to prioritize retention outreach, subject to fairness, privacy, and intervention-cost checks.

## Dataset

`data/synthetic_customers.csv` contains 2,500 simulated customers, with tenure, monthly charges, recent support tickets, contract type, autopay status, usage hours, and a binary churn label. `customer_id` is excluded from training. Data were generated with a fixed random seed (42).

## Methodology

1. **Split** into stratified training (80%) and held-out test (20%) partitions.
2. **Preprocess** numerical features with median imputation and scaling, and categorical features with most-frequent imputation and one-hot encoding. All preprocessing is fitted inside scikit-learn pipelines, preventing train/test leakage.
3. **Compare** Logistic Regression, Random Forest, and Histogram Gradient Boosting with stratified 5-fold cross-validation on the training partition.
4. **Select** the highest cross-validation ROC-AUC model and refit on the full training partition.
5. **Evaluate once** on the untouched test partition using ROC-AUC, average precision, classification report, and confusion matrix.
6. **Explain** the selected model using permutation importance. This describes predictive associations, not causal effects.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

Results are written to `outputs/`: `cross_validation.csv`, `metrics.json`, `roc_curve.png`, `confusion_matrix.png`, and `permutation_importance.csv`.

## Key technical skills

- Classification and business problem framing
- Data preprocessing with `ColumnTransformer` and `Pipeline`
- Stratified validation and model comparison
- ROC-AUC, average precision, precision/recall and confusion matrices
- Model interpretability via permutation importance
- Reproducible Python workflows

## Limitations / next steps

- **Synthetic data:** Results are illustrative and cannot be generalized to real customers.
- **Threshold:** The default 0.5 decision threshold is not tuned for retention costs. In a real deployment, select it using a separate validation procedure.
- **No temporal validation:** Real churn models should train on earlier periods and test on later periods to avoid look-ahead bias.
- **Ethics:** Validate subgroup performance and customer privacy before any operational use.
- **Next:** Add calibration, cost-sensitive decision analysis, drift monitoring, and a Streamlit dashboard.

## Repository structure

```text
customer-churn-ml/
├── data/synthetic_customers.csv
├── outputs/                 # Generated results and plots
├── train.py                 # Training, cross-validation, evaluation
├── requirements.txt
├── .gitignore
└── README.md
```