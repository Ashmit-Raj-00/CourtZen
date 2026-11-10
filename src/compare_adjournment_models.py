import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

feature_path = BASE_DIR / "data" / "processed" / "court_cases_features.csv"
clean_path = BASE_DIR / "data" / "processed" / "court_cases_clean.csv"

output_dir = BASE_DIR / "outputs" / "adjournment_evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

comparison_path = output_dir / "model_comparison.csv"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

X = pd.read_csv(feature_path)
clean_data = pd.read_csv(clean_path, keep_default_na=False)

y = clean_data["adjourned"].astype(int)

# Remove case ID
X = X.drop(columns=["case_id"])


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


# --------------------------------------------------
# Class Weight
# --------------------------------------------------

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count


# --------------------------------------------------
# Define Models
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
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
}


# --------------------------------------------------
# Train & Evaluate
# --------------------------------------------------

results = []

print("\n========== MODEL COMPARISON ==========\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4)
    })


# --------------------------------------------------
# Results Table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))


# --------------------------------------------------
# Save Comparison
# --------------------------------------------------

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nComparison saved to:")
print(comparison_path)