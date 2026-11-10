import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

feature_path = BASE_DIR / "data" / "processed" / "court_cases_features.csv"
clean_path = BASE_DIR / "data" / "processed" / "court_cases_clean.csv"
model_path = BASE_DIR / "models" / "adjournment_risk_model.pkl"

output_dir = BASE_DIR / "outputs" / "adjournment_evaluation"
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

X = pd.read_csv(feature_path)
clean_data = pd.read_csv(clean_path, keep_default_na=False)

y = clean_data["adjourned"].astype(int)

# Remove ID
X = X.drop(columns=["case_id"])


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_data = joblib.load(model_path)

model = model_data["model"]


# --------------------------------------------------
# Same Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Adjournment", "Adjournment"],
        digits=4
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Adjournment", "Adjournment"]
)

disp.plot()
plt.title("Adjournment Risk - Confusion Matrix")
plt.tight_layout()

cm_path = output_dir / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

roc_auc = roc_auc_score(y_test, y_probability)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Adjournment Risk")
plt.legend()
plt.tight_layout()

roc_path = output_dir / "roc_curve.png"
plt.savefig(roc_path, dpi=300)
plt.close()


# --------------------------------------------------
# Final Output
# --------------------------------------------------

print("\nROC-AUC:", round(roc_auc, 4))

print("\nEvaluation files saved:")

print(cm_path)
print(roc_path)