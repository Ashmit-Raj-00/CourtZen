import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# COURTZEN - EXPLORATORY DATA ANALYSIS
# ==========================================

# Load processed dataset
df = pd.read_csv(
    "data/processed/court_cases_clean.csv",
    keep_default_na=False
)

print("=" * 60)
print("COURTZEN - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ==========================================
# 1. BASIC STATISTICS
# ==========================================

print("\n1. BASIC STATISTICS")

print("\nTotal Cases:")
print(len(df))

print("\nAverage Hearing Duration:")
print(
    round(df["hearing_duration_minutes"].mean(), 2),
    "minutes"
)

print("\nMinimum Hearing Duration:")
print(
    df["hearing_duration_minutes"].min(),
    "minutes"
)

print("\nMaximum Hearing Duration:")
print(
    df["hearing_duration_minutes"].max(),
    "minutes"
)

print("\nOverall Adjournment Rate:")
print(
    round(df["adjourned"].mean() * 100, 2),
    "%"
)


# ==========================================
# 2. CASE TYPE DISTRIBUTION
# ==========================================

case_type_counts = df["case_type"].value_counts()

print("\n2. CASE TYPE DISTRIBUTION")
print(case_type_counts)

plt.figure(figsize=(8, 5))

case_type_counts.plot(kind="bar")

plt.title("Court Case Type Distribution")
plt.xlabel("Case Type")
plt.ylabel("Number of Cases")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "data/processed/case_type_distribution.png"
)

plt.show()


# ==========================================
# 3. AVERAGE HEARING DURATION BY CASE TYPE
# ==========================================

duration_by_type = (
    df.groupby("case_type")["hearing_duration_minutes"]
    .mean()
    .sort_values(ascending=False)
)

print("\n3. AVERAGE HEARING DURATION BY CASE TYPE")
print(duration_by_type.round(2))

plt.figure(figsize=(8, 5))

duration_by_type.plot(kind="bar")

plt.title("Average Hearing Duration by Case Type")
plt.xlabel("Case Type")
plt.ylabel("Average Hearing Duration (minutes)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "data/processed/hearing_duration_by_case_type.png"
)

plt.show()


# ==========================================
# 4. HEARING DURATION BY COMPLEXITY
# ==========================================

duration_by_complexity = (
    df.groupby("case_complexity")["hearing_duration_minutes"]
    .mean()
)

print("\n4. AVERAGE HEARING DURATION BY CASE COMPLEXITY")
print(duration_by_complexity.round(2))

plt.figure(figsize=(8, 5))

duration_by_complexity.plot(kind="bar")

plt.title("Average Hearing Duration by Case Complexity")
plt.xlabel("Case Complexity")
plt.ylabel("Average Hearing Duration (minutes)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "data/processed/duration_by_complexity.png"
)

plt.show()


# ==========================================
# 5. ADJOURNMENT RATE BY COMPLEXITY
# ==========================================

adjournment_by_complexity = (
    df.groupby("case_complexity")["adjourned"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\n5. ADJOURNMENT RATE BY CASE COMPLEXITY")
print(adjournment_by_complexity.round(2))

plt.figure(figsize=(8, 5))

adjournment_by_complexity.plot(kind="bar")

plt.title("Adjournment Rate by Case Complexity")
plt.xlabel("Case Complexity")
plt.ylabel("Adjournment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "data/processed/adjournment_by_complexity.png"
)

plt.show()


# ==========================================
# 6. ADJOURNMENT RATE VS PREVIOUS ADJOURNMENTS
# ==========================================

risk_by_previous_adjournments = (
    df.groupby("previous_adjournments")["adjourned"]
    .mean()
    .sort_index()
    * 100
)

print("\n6. ADJOURNMENT RATE VS PREVIOUS ADJOURNMENTS")
print(risk_by_previous_adjournments.round(2))

plt.figure(figsize=(9, 5))

risk_by_previous_adjournments.plot(kind="bar")

plt.title("Adjournment Rate vs Previous Adjournments")
plt.xlabel("Number of Previous Adjournments")
plt.ylabel("Adjournment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "data/processed/risk_vs_previous_adjournments.png"
)

plt.show()


# ==========================================
# 7. ADJOURNMENT REASONS
# ==========================================

reason_counts = (
    df[df["adjournment_reason"] != "None"]
    ["adjournment_reason"]
    .value_counts()
)

print("\n7. ADJOURNMENT REASONS")
print(reason_counts)

plt.figure(figsize=(9, 5))

reason_counts.plot(kind="bar")

plt.title("Major Reasons for Case Adjournment")
plt.xlabel("Adjournment Reason")
plt.ylabel("Number of Cases")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "data/processed/adjournment_reasons.png"
)

plt.show()


# ==========================================
# FINAL
# ==========================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated graphs:")
print("1. case_type_distribution.png")
print("2. hearing_duration_by_case_type.png")
print("3. duration_by_complexity.png")
print("4. adjournment_by_complexity.png")
print("5. risk_vs_previous_adjournments.png")
print("6. adjournment_reasons.png")