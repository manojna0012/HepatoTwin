import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

# =====================================================
# Paths
# =====================================================

INPUT_FILE = Path("data/processed/nhanes_merged.csv")
OUTPUT_FILE = Path("data/processed/nhanes_preprocessed.csv")

# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("NHANES PREPROCESSING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Original Shape: {df.shape}")

# =====================================================
# Features for Risk Model
# =====================================================

FEATURES = [
    "RIDAGEYR",      # Age
    "RIAGENDR",      # Sex
    "RIDRETH3",      # Race/Ethnicity
    "BMXBMI",        # BMI
    "BMXWAIST",      # Waist Circumference
    "LBXSATSI",      # ALT
    "LBXSASSI",      # AST
    "LBXGH",         # HbA1c
    "LBXTC",         # Total Cholesterol
    "LBDHDD",        # HDL
    "DR1TKCAL",      # Calories
    "DR1TPROT",      # Protein
    "DR1TCARB",      # Carbohydrates
    "DR1TTFAT",      # Total Fat
]

TARGET = "LUXCAPM"

# Keep only required columns
columns = FEATURES + [TARGET]
df = df[columns]

print(f"Columns Selected: {len(df.columns)}")

# =====================================================
# Remove Missing Target
# =====================================================

before = len(df)

df = df.dropna(subset=[TARGET])

after = len(df)

print(f"Removed {before-after} rows with missing CAP")
print(f"Remaining: {after}")

# =====================================================
# Missing Values Before Imputation
# =====================================================

print("\nMissing Values Before:")

print(df.isnull().sum())

# =====================================================
# Impute Numerical Features
# =====================================================

numeric_cols = [
    "RIDAGEYR",
    "BMXBMI",
    "BMXWAIST",
    "LBXSATSI",
    "LBXSASSI",
    "LBXGH",
    "LBXTC",
    "LBDHDD",
    "DR1TKCAL",
    "DR1TPROT",
    "DR1TCARB",
    "DR1TTFAT",
]

imputer = SimpleImputer(strategy="median")

df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# =====================================================
# Encode Categorical Variables
# =====================================================

df = pd.get_dummies(
    df,
    columns=[
        "RIAGENDR",
        "RIDRETH3"
    ],
    drop_first=True
)

# =====================================================
# Final Check
# =====================================================

print("\nMissing Values After:")

print(df.isnull().sum())

print("\nFinal Shape:")

print(df.shape)

# =====================================================
# Save
# =====================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\nPreprocessing Complete!")