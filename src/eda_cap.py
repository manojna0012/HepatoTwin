import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =====================================================
# Load merged dataset
# =====================================================

DATA = Path("data/processed/nhanes_merged.csv")

df = pd.read_csv(DATA)

print("=" * 60)
print("NHANES EXPLORATORY ANALYSIS")
print("=" * 60)

print(f"\nOriginal dataset shape: {df.shape}")


# =====================================================
# Keep only complete elastography examinations
# =====================================================

if "LUAXSTAT" not in df.columns:
    raise ValueError("LUAXSTAT column not found.")

df = df[df["LUAXSTAT"] == 1].copy()

print(f"After LUAXSTAT == 1 filter: {df.shape}")


# =====================================================
# Candidate features for risk model
# =====================================================

candidate_features = [
    "RIDAGEYR",      # Age
    "RIAGENDR",      # Sex
    "RIDRETH3",      # Race/Ethnicity
    "BMXBMI",        # BMI
    "BMXWAIST",      # Waist circumference
    "LBXSATSI",      # ALT
    "LBXSASSI",      # AST
    "LBXGH",         # HbA1c
    "LBXTC",         # Total cholesterol
    "LBDHDD",        # HDL
    "DR1TKCAL",      # Calories
    "DR1TPROT",      # Protein
    "DR1TCARB",      # Carbohydrates
    "DR1TTFAT",      # Total fat
    "LUXCAPM"        # CAP target
]

# Keep only columns that actually exist
candidate_features = [
    col for col in candidate_features
    if col in df.columns
]

print("\nCandidate Features:")
print(candidate_features)


# =====================================================
# Remove missing CAP for CAP-based analysis
# =====================================================

df_cap = df.dropna(subset=["LUXCAPM"]).copy()

print(f"\nParticipants with valid CAP: {len(df_cap)}")


# =====================================================
# Missing values
# =====================================================

print("\nMissing Values:")

missing = (
    df_cap[candidate_features]
    .isnull()
    .sum()
    .to_frame("Missing")
)

missing["Missing_%"] = (
    missing["Missing"] / len(df_cap) * 100
).round(2)

print(missing)


# =====================================================
# Summary statistics
# =====================================================

print("\nSummary Statistics:")

print(
    df_cap[candidate_features]
    .describe()
)


# =====================================================
# CAP Distribution
# =====================================================

cap = df_cap["LUXCAPM"]

print("\nCAP Summary:")
print(f"Participants with CAP: {len(cap)}")
print(f"Mean CAP: {cap.mean():.2f}")
print(f"Median CAP: {cap.median():.2f}")
print(f"Std CAP: {cap.std():.2f}")

pct_at_ceiling = (cap == 400).mean() * 100

print(
    f"Participants at CAP ceiling "
    f"(400 dB/m): {pct_at_ceiling:.2f}%"
)

print(
    "Note: CAP values are capped at 400 dB/m; "
    "values at the ceiling may be right-censored."
)

plt.figure(figsize=(8, 5))

sns.histplot(
    cap,
    bins=40
)

plt.title("Controlled Attenuation Parameter (CAP) Distribution")
plt.xlabel("CAP (dB/m)")
plt.ylabel("Participants")

plt.tight_layout()
plt.savefig("cap_distribution.png")
plt.close()


# =====================================================
# BMI Distribution
# =====================================================

if "BMXBMI" in df_cap.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df_cap["BMXBMI"].dropna(),
        bins=35
    )

    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Participants")

    plt.tight_layout()
    plt.savefig("bmi_distribution.png")
    plt.close()


# =====================================================
# Correlation Matrix
# =====================================================

# Exclude RIDRETH3 because it is a nominal category.
# Its numeric codes should not be interpreted as continuous.

corr_features = [
    feature
    for feature in candidate_features
    if feature != "RIDRETH3"
]

numeric = df_cap[corr_features].select_dtypes(
    include="number"
)

corr = numeric.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Candidate Numeric Features")

plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()


# =====================================================
# CAP vs BMI
# =====================================================

if (
    "BMXBMI" in df_cap.columns
    and "LUXCAPM" in df_cap.columns
):

    plot_df = df_cap[
        ["BMXBMI", "LUXCAPM"]
    ].dropna()

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        x="BMXBMI",
        y="LUXCAPM",
        data=plot_df,
        alpha=0.5
    )

    plt.title("CAP vs BMI")
    plt.xlabel("BMI")
    plt.ylabel("CAP (dB/m)")

    plt.tight_layout()
    plt.savefig("cap_vs_bmi.png")
    plt.close()


# =====================================================
# Save candidate feature list
# =====================================================

feature_df = pd.DataFrame({
    "Feature": candidate_features
})

feature_df.to_csv(
    "candidate_features.csv",
    index=False
)


# =====================================================
# Completed
# =====================================================

print("\nSaved:")
print("cap_distribution.png")
print("bmi_distribution.png")
print("correlation_matrix.png")
print("cap_vs_bmi.png")
print("candidate_features.csv")

print("\nEDA Completed")