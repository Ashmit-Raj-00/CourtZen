import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import joblib


# ============================================================
# COURTZEN - HEARING DURATION PREDICTION
# ============================================================

print("=" * 65)
print("COURTZEN - HEARING DURATION PREDICTION MODEL")
print("=" * 65)


# ============================================================
# 1. LOAD DATA
# ============================================================

data_path = "data/processed/court_cases_clean.csv"

df = pd.read_csv(
    data_path,
    keep_default_na=False
)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "case_type",
    "case_priority",
    "case_complexity",
    "case_age_days",
    "previous_hearings",
    "previous_adjournments",
    "adjournment_count_last_year",
    "judge_experience_years",
    "day_of_week",
    "month",
    "courtroom_available",
    "legal_representative_available"
]

target = "hearing_duration_minutes"

X = df[features]
y = df[target]


print("\nFeatures used:")
for feature in features:
    print(f"- {feature}")

print(f"\nTarget variable: {target}")


# ============================================================
# 3. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = [
    "case_type",
    "case_priority",
    "case_complexity",
    "day_of_week"
]

numerical_features = [
    "case_age_days",
    "previous_hearings",
    "previous_adjournments",
    "adjournment_count_last_year",
    "judge_experience_years",
    "month",
    "courtroom_available",
    "legal_representative_available"
]


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain/Test Split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 6. XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


# ============================================================
# 7. CREATE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 65)
print("MODEL PERFORMANCE")
print("=" * 65)

print(f"\nMAE  : {mae:.2f} minutes")
print(f"RMSE : {rmse:.2f} minutes")
print(f"R²   : {r2:.4f}")


# ============================================================
# 11. SHOW SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Duration": y_test.values,
    "Predicted Duration": np.round(y_pred, 2)
})

print("\nSample Predictions:")
print(results.head(10))


# ============================================================
# 12. SAVE MODEL
# ============================================================

model_path = "models/hearing_duration_model.pkl"

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved successfully:")
print(model_path)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 65)
print("HEARING DURATION MODEL COMPLETED SUCCESSFULLY")
print("=" * 65)