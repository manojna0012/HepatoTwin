import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------------------------------
# Load merged dataset
# -------------------------------------------------

DATA = Path("data/processed/nhanes_merged.csv")

df = pd.read_csv(DATA)

print("="*60)
print("NHANES Exploratory Analysis")
print("="*60)

print("\nDataset shape:", df.shape)

# -------------------------------------------------
# Candidate variables
# -------------------------------------------------

candidate_features = [
    "RIDAGEYR",      # Age
    "RIAGENDR",      # Sex
    "RIDRETH3",      # Race
    "BMXBMI",        # BMI
    "BMXWAIST",      # Waist Circumference
    "LBXSATSI",      # ALT
    "LBXSASSI",      # AST
    "LBXGH",         # HbA1c
    "LBXTC",         # Total Cholesterol
    "LBDHDD",        # HDL (change if your file uses another name)
    "LBXTR",         # Triglycerides
    "DR1TKCAL",      # Calories
    "DR1TPROT",      # Protein
    "DR1TCARB",      # Carbohydrates
    "DR1TTFAT",      # Total Fat
    "LUXCAPM"        # CAP (Target)
]

candidate_features = [
    c for c in candidate_features
    if c in df.columns
]

print("\nCandidate Features")
print(candidate_features)

# -------------------------------------------------
# Missing values
# -------------------------------------------------

print("\nMissing Values")

missing = (
    df[candidate_features]
    .isnull()
    .sum()
    .to_frame("Missing")
)

missing["Missing_%"] = (
    missing["Missing"] / len(df) * 100
).round(2)

print(missing)

# -------------------------------------------------
# Summary statistics
# -------------------------------------------------

print("\nSummary Statistics")

print(
    df[candidate_features]
    .describe()
)

# -------------------------------------------------
# CAP Distribution
# -------------------------------------------------

cap = df["LUXCAPM"].dropna()

print("\nParticipants with CAP:", len(cap))
print("Mean CAP:", cap.mean())
print("Median CAP:", cap.median())
print("Std:", cap.std())

plt.figure(figsize=(8,5))

sns.histplot(cap, bins=40)

plt.title("Controlled Attenuation Parameter (CAP)")
plt.xlabel("CAP (dB/m)")
plt.ylabel("Participants")

plt.tight_layout()
plt.savefig("cap_distribution.png")

# -------------------------------------------------
# BMI Distribution
# -------------------------------------------------

if "BMXBMI" in df.columns:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["BMXBMI"].dropna(),
        bins=35
    )

    plt.title("BMI Distribution")

    plt.tight_layout()

    plt.savefig("bmi_distribution.png")

# -------------------------------------------------
# Correlation Matrix
# -------------------------------------------------

numeric = df[candidate_features].select_dtypes("number")

corr = numeric.corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("correlation_matrix.png")

# -------------------------------------------------
# CAP vs BMI
# -------------------------------------------------

if (
    "BMXBMI" in df.columns and
    "LUXCAPM" in df.columns
):

    plt.figure(figsize=(6,5))

    sns.scatterplot(
        x="BMXBMI",
        y="LUXCAPM",
        data=df
    )

    plt.tight_layout()

    plt.savefig("cap_vs_bmi.png")

# -------------------------------------------------
# Candidate feature list
# -------------------------------------------------

feature_df = pd.DataFrame({
    "Feature": candidate_features
})

feature_df.to_csv(
    "candidate_features.csv",
    index=False
)

print("\nSaved:")
print("cap_distribution.png")
print("bmi_distribution.png")
print("correlation_matrix.png")
print("cap_vs_bmi.png")
print("candidate_features.csv")

print("\nMeeting 2 Completed ✓")
import pandas as pd

demo = pd.read_sas("data/raw/P_DEMO.XPT")
bmx = pd.read_sas("data/raw/P_BMX.XPT")

print("DEMO columns:")
print(demo.columns.tolist())

print("\nBMX columns:")
print(bmx.columns.tolist())

print("\nFirst 5 SEQN in DEMO:")
print(demo["SEQN"].head())

print("\nFirst 5 SEQN in BMX:")
print(bmx["SEQN"].head())