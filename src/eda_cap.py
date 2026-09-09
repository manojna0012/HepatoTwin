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
for feature in candidate_features:
    print(" ", feature)


# =====================================================
# Remove missing CAP for CAP-based analysis
# =====================================================

df_cap = df.dropna(subset=["LUXCAPM"]).copy()

print(f"\nParticipants with valid CAP: {len(df_cap)}")


# =====================================================
# Missing values
# =====================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

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

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(
    df_cap[candidate_features]
    .describe()
)


# =====================================================
# CAP Distribution
# =====================================================

print("\n" + "=" * 60)
print("CAP DISTRIBUTION")
print("=" * 60)

cap = df_cap["LUXCAPM"]

print(f"Participants with CAP: {len(cap)}")
print(f"Mean CAP: {cap.mean():.2f}")
print(f"Median CAP: {cap.median():.2f}")
print(f"Std CAP: {cap.std():.2f}")
print(f"Minimum CAP: {cap.min():.2f}")
print(f"Maximum CAP: {cap.max():.2f}")

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
# Sex Distribution
# =====================================================

if "RIAGENDR" in df_cap.columns:

    print("\n" + "=" * 60)
    print("SEX DISTRIBUTION")
    print("=" * 60)

    sex_counts = (
        df_cap["RIAGENDR"]
        .value_counts()
        .sort_index()
    )

    male_count = sex_counts.get(1, 0)
    female_count = sex_counts.get(2, 0)

    print(f"Male: {male_count}")
    print(f"Female: {female_count}")

    print(f"Total: {male_count + female_count}")

    sex_labels = {
        1: "Male",
        2: "Female"
    }

    sex_plot = (
        df_cap["RIAGENDR"]
        .map(sex_labels)
        .value_counts()
        .reindex(["Male", "Female"])
    )

    plt.figure(figsize=(7, 5))

    sex_plot.plot(
        kind="bar"
    )

    plt.title("Sex Distribution")
    plt.xlabel("Sex")
    plt.ylabel("Number of Participants")

    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig("sex_distribution.png")
    plt.close()


# =====================================================
# Race/Ethnicity Distribution
# =====================================================

if "RIDRETH3" in df_cap.columns:

    print("\n" + "=" * 60)
    print("RACE/ETHNICITY DISTRIBUTION")
    print("=" * 60)

    race_labels = {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other Race / Multiracial"
    }

    race_plot = (
        df_cap["RIDRETH3"]
        .map(race_labels)
        .value_counts()
    )

    print("\nParticipant counts:")

    for race, count in race_plot.items():
        print(f"{race}: {count}")

    print(f"Total: {race_plot.sum()}")

    plt.figure(figsize=(10, 6))

    race_plot.plot(
        kind="bar"
    )

    plt.title("Race/Ethnicity Distribution")
    plt.xlabel("Race/Ethnicity")
    plt.ylabel("Number of Participants")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    plt.savefig("race_distribution.png")
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

print("\n" + "=" * 60)
print("SAVED FILES")
print("=" * 60)

print("cap_distribution.png")
print("bmi_distribution.png")
print("sex_distribution.png")
print("race_distribution.png")
print("correlation_matrix.png")
print("cap_vs_bmi.png")
print("candidate_features.csv")

print("\nEDA Completed")