"""Train and evaluate reproducible churn classification models on synthetic data."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_customers.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA)
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]
num = ["tenure_months", "monthly_charges", "support_tickets_90d", "monthly_usage_hours"]
cat = ["contract_type", "autopay"]
preprocess = ColumnTransformer([
    ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
    ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, model in models.items():
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"})
    rows.append({"model": name, "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()), "cv_average_precision_mean": float(scores["test_average_precision"].mean())})
ranking = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
ranking.to_csv(OUT / "cross_validation.csv", index=False)
best_name = ranking.iloc[0]["model"]
best = Pipeline([("preprocess", preprocess), ("model", models[best_name])]).fit(X_train, y_train)
prob = best.predict_proba(X_test)[:, 1]
pred = (prob >= .5).astype(int)
metrics = {"selected_model": best_name, "test_roc_auc": float(roc_auc_score(y_test, prob)), "test_average_precision": float(average_precision_score(y_test, prob)), "test_classification_report": classification_report(y_test, pred, output_dict=True), "test_confusion_matrix": confusion_matrix(y_test, pred).tolist(), "dataset_rows": len(df), "churn_rate": float(y.mean())}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
fpr, tpr, _ = roc_curve(y_test, prob)
plt.figure(figsize=(6, 4)); plt.plot(fpr, tpr, label=f"{best_name} (AUC={metrics['test_roc_auc']:.3f})");plt.plot([0,1],[0,1],linestyle='--');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title('Held-out test ROC curve');plt.legend();plt.tight_layout();plt.savefig(OUT/'roc_curve.png',dpi=160);plt.close()
ConfusionMatrixDisplay.from_predictions(y_test,pred);plt.title('Held-out test confusion matrix');plt.tight_layout();plt.savefig(OUT/'confusion_matrix.png',dpi=160);plt.close()
# Permutation importance is model-agnostic and computed only on the held-out test set.
from sklearn.inspection import permutation_importance
importance = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
pd.DataFrame({"feature":X_test.columns,"importance_mean":importance.importances_mean,"importance_std":importance.importances_std}).sort_values("importance_mean",ascending=False).to_csv(OUT/'permutation_importance.csv',index=False)
print(ranking.to_string(index=False));print(json.dumps({k:v for k,v in metrics.items() if k not in ('test_classification_report','test_confusion_matrix')},indent=2))