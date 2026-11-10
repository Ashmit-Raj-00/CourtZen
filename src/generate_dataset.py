import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of cases
N = 3000

# -----------------------------
# Basic case information
# -----------------------------

case_id = [f"C{100000 + i}" for i in range(N)]

case_type = np.random.choice(
    ["Civil", "Criminal", "Family", "Property", "Commercial"],
    size=N,
    p=[0.25, 0.25, 0.15, 0.20, 0.15]
)

case_priority = np.random.choice(
    ["Low", "Medium", "High"],
    size=N,
    p=[0.45, 0.40, 0.15]
)

case_complexity = np.random.choice(
    ["Low", "Medium", "High"],
    size=N,
    p=[0.40, 0.40, 0.20]
)

# -----------------------------
# Historical case information
# -----------------------------

case_age_days = np.random.randint(30, 2500, N)

previous_hearings = np.random.poisson(4, N)
previous_hearings = np.clip(previous_hearings, 0, 20)

previous_adjournments = np.random.poisson(1.5, N)
previous_adjournments = np.minimum(
    previous_adjournments,
    previous_hearings
)

adjournment_count_last_year = np.random.poisson(1, N)
adjournment_count_last_year = np.clip(
    adjournment_count_last_year, 0, 10
)

judge_experience_years = np.random.randint(2, 31, N)

day_of_week = np.random.choice(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    size=N
)

month = np.random.randint(1, 13, N)

# -----------------------------
# Scheduling/resource features
# -----------------------------

courtroom_available = np.random.choice(
    [0, 1],
    size=N,
    p=[0.10, 0.90]
)

legal_representative_available = np.random.choice(
    [0, 1],
    size=N,
    p=[0.08, 0.92]
)

# -----------------------------
# Hearing duration generation
# -----------------------------

complexity_effect = np.select(
    [
        case_complexity == "Low",
        case_complexity == "Medium",
        case_complexity == "High"
    ],
    [0, 15, 35]
)

type_effect = np.select(
    [
        case_type == "Civil",
        case_type == "Criminal",
        case_type == "Family",
        case_type == "Property",
        case_type == "Commercial"
    ],
    [5, 15, 10, 12, 20]
)

priority_effect = np.select(
    [
        case_priority == "Low",
        case_priority == "Medium",
        case_priority == "High"
    ],
    [0, 5, 10]
)

noise = np.random.normal(0, 8, N)

hearing_duration_minutes = (
    20
    + complexity_effect
    + type_effect
    + priority_effect
    + previous_hearings * 1.5
    + noise
)

hearing_duration_minutes = np.clip(
    hearing_duration_minutes,
    10,
    180
).round(0).astype(int)

# -----------------------------
# Adjournment probability
# -----------------------------

risk_score = (
    -2.5
    + previous_adjournments * 0.45
    + adjournment_count_last_year * 0.30
    + (case_age_days / 2500) * 0.7
    + (case_complexity == "High") * 0.35
    + (legal_representative_available == 0) * 1.0
    + (courtroom_available == 0) * 0.8
)

adjournment_probability = 1 / (
    1 + np.exp(-risk_score)
)

adjourned = np.random.binomial(
    1,
    adjournment_probability
)

# -----------------------------
# Adjournment reason
# -----------------------------

reasons = []

for value in adjourned:
    if value == 0:
        reasons.append("None")
    else:
        reasons.append(
            np.random.choice(
                [
                    "Legal Representative Unavailable",
                    "Insufficient Documentation",
                    "Witness Unavailable",
                    "Courtroom Unavailable",
                    "Other"
                ]
            )
        )

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "case_id": case_id,
    "case_type": case_type,
    "case_priority": case_priority,
    "case_complexity": case_complexity,
    "case_age_days": case_age_days,
    "previous_hearings": previous_hearings,
    "previous_adjournments": previous_adjournments,
    "adjournment_count_last_year": adjournment_count_last_year,
    "judge_experience_years": judge_experience_years,
    "day_of_week": day_of_week,
    "month": month,
    "courtroom_available": courtroom_available,
    "legal_representative_available": legal_representative_available,
    "hearing_duration_minutes": hearing_duration_minutes,
    "adjourned": adjourned,
    "adjournment_reason": reasons
})

# -----------------------------
# Save dataset
# -----------------------------

output_path = "data/raw/court_cases.csv"

df.to_csv(output_path, index=False)

print("CourtZen dataset created successfully!")
print(f"Total cases: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nAdjournment rate:")
print(f"{df['adjourned'].mean() * 100:.2f}%")