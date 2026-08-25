import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_component(filename):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {filename}")

    df = pd.read_sas(path)

    # Make SEQN consistent
    df["SEQN"] = df["SEQN"].astype(int)

    print(f"{filename}: {df.shape}")

    return df


print("=" * 60)
print("NHANES DATA MERGE")
print("=" * 60)

# ---------------------------------------------------
# Load files
# ---------------------------------------------------

lux = load_component("P_LUX.XPT")
bio = load_component("P_BIOPRO.XPT")
ghb = load_component("P_GHB.XPT")
chol = load_component("P_TCHOL.XPT")
hdl = load_component("P_HDL.XPT")
trig = load_component("P_TRIGLY.XPT")
diet = load_component("P_DR1TOT.XPT")
demo = load_component("P_DEMO.XPT")
bmx = load_component("P_BMX.XPT")

# ---------------------------------------------------
# Keep only useful columns
# ---------------------------------------------------

demo = demo[
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
    ]
]

bmx = bmx[
    [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
        "BMXWT",
        "BMXHT",
    ]
]

# Diet variables

diet_cols = [
    "SEQN",
    "DR1TKCAL",
    "DR1TPROT",
    "DR1TCARB",
    "DR1TTFAT",
]

diet = diet[diet_cols]

# ---------------------------------------------------
# Merge
# ---------------------------------------------------

merged = lux.copy()

datasets = [
    ("BIOPRO", bio),
    ("GHB", ghb),
    ("TCHOL", chol),
    ("HDL", hdl),
    ("TRIGLY", trig),
    ("DIET", diet),
    ("DEMO", demo),
    ("BMX", bmx),
]

for name, df in datasets:

    merged = merged.merge(
        df,
        on="SEQN",
        how="left",
        suffixes=("", f"_{name}")
    )

    print(f"After {name}: {merged.shape}")

# ---------------------------------------------------
# Remove duplicated columns if any
# ---------------------------------------------------

merged = merged.loc[:, ~merged.columns.duplicated()]

# ---------------------------------------------------
# Check important variables
# ---------------------------------------------------

print("\n" + "=" * 60)
print("KEY VARIABLES")
print("=" * 60)

key_vars = [
    "LUXCAPM",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "BMXBMI",
    "BMXWAIST",
    "LBXGH",
    "LBXTC",
    "LBXTR",
]

for col in key_vars:

    if col in merged.columns:

        print(
            f"{col:12s}",
            merged[col].isna().sum(),
            "missing"
        )

# ---------------------------------------------------
# Dataset summary
# ---------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Shape:", merged.shape)

print(
    "Participants:",
    merged.SEQN.nunique()
)

print(
    "Duplicate SEQN:",
    merged.SEQN.duplicated().sum()
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

outfile = PROCESSED_DIR / "nhanes_merged.csv"

merged.to_csv(outfile, index=False)

print("\nSaved to")

print(outfile)