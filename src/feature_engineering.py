import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "court_cases_clean.csv"
output_path = BASE_DIR / "data" / "processed" / "court_cases_features.csv"

# Load cleaned dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 1. Derived Features
# --------------------------------------------------

# Average previous hearing load
df["avg_hearing_history"] = (
    df["previous_hearings"] / (df["case_age_days"] + 1)
)

# Recent adjournment intensity
df["adjournment_intensity"] = (
    df["adjournment_count_last_year"] / (df["case_age_days"] / 365 + 1)
)

# Resource availability score
df["resource_availability_score"] = (
    df["courtroom_available"].astype(int)
    + df["legal_representative_available"].astype(int)
)

# --------------------------------------------------
# 2. Remove target-related columns from features
# --------------------------------------------------

# These columns are targets / post-hearing information
target_columns = [
    "hearing_duration_minutes",
    "adjourned",
    "adjournment_reason"
]

feature_df = df.drop(columns=target_columns)

# --------------------------------------------------
# 3. One-Hot Encoding for categorical features
# --------------------------------------------------

categorical_columns = [
    "case_type",
    "case_priority",
    "case_complexity",
    "day_of_week",
    "month"
]

feature_df = pd.get_dummies(
    feature_df,
    columns=categorical_columns,
    drop_first=False
)

# --------------------------------------------------
# 4. Convert boolean columns to integers
# --------------------------------------------------

boolean_columns = feature_df.select_dtypes(include=["bool"]).columns

feature_df[boolean_columns] = feature_df[boolean_columns].astype(int)

# --------------------------------------------------
# 5. Save engineered features
# --------------------------------------------------

feature_df.to_csv(output_path, index=False)

print("Feature engineering completed.")
print("Final feature shape:", feature_df.shape)
print("Saved to:", output_path)

print("\nFeature columns:")
for column in feature_df.columns:
    print("-", column)