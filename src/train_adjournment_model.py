import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

feature_path = BASE_DIR / "data" / "processed" / "court_cases_features.csv"
clean_path = BASE_DIR / "data" / "processed" / "court_cases_clean.csv"
model_path = BASE_DIR / "models" / "adjournment_risk_model.pkl"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

features = pd.read_csv(feature_path)
clean_data = pd.read_csv(clean_path, keep_default_na=False)

print("Feature dataset shape:", features.shape)
print("Target dataset shape:", clean_data.shape)


# --------------------------------------------------
# Prepare Target
# --------------------------------------------------

y = clean_data["adjourned"].astype(int)

# Remove ID because it has no predictive meaning
X = features.drop(columns=["case_id"])


print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# Calculate Class Weight
# --------------------------------------------------

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count


# --------------------------------------------------
# XGBoost Classification Model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)


# --------------------------------------------------
# Train Model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

cm = confusion_matrix(y_test, y_pred)


# --------------------------------------------------
# Display Results
# --------------------------------------------------

print("\n========== ADJOURNMENT RISK MODEL ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# Save Model
# --------------------------------------------------

model_data = {
    "model": model,
    "feature_columns": X.columns.tolist()
}

joblib.dump(model_data, model_path)

print("\nModel saved to:")
print(model_path)