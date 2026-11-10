import pandas as pd
import numpy as np

# Load raw dataset
input_path = "data/raw/court_cases.csv"

df = pd.read_csv(input_path, keep_default_na=False)

print("=" * 60)
print("COURTZEN - DATA QUALITY REPORT")
print("=" * 60)

# Dataset shape
print("\n1. Dataset Shape")
print(df.shape)

# Column information
print("\n2. Column Information")
print(df.info())

# Missing values
print("\n3. Missing Values")
print(df.isnull().sum())

# Duplicate records
print("\n4. Duplicate Records")
print(df.duplicated().sum())

# Unique values
print("\n5. Unique Values")
print(df.nunique())

# Numerical statistics
print("\n6. Numerical Statistics")
print(df.describe())

# Categorical distributions
print("\n7. Case Type Distribution")
print(df["case_type"].value_counts())

print("\n8. Case Priority Distribution")
print(df["case_priority"].value_counts())

print("\n9. Case Complexity Distribution")
print(df["case_complexity"].value_counts())

print("\n10. Adjournment Distribution")
print(df["adjourned"].value_counts())

# Save processed copy
output_path = "data/processed/court_cases_clean.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("PREPROCESSING CHECK COMPLETE")
print(f"Clean dataset saved to: {output_path}")
print("=" * 60)