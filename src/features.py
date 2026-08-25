import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "nhanes_merged.csv"


# ============================================================
# LOAD ONE NHANES XPT FILE
# ============================================================

def load_component(filename):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Please make sure the file is inside data/raw/"
        )

    df = pd.read_sas(path)

    print(f"{filename}: {df.shape}")

    if "SEQN" not in df.columns:
        raise ValueError(f"{filename} does not contain SEQN.")

    print(f"  Unique SEQN: {df['SEQN'].nunique()}")
    print(f"  Duplicate SEQN: {df['SEQN'].duplicated().sum()}")

    return df


# ============================================================
# MERGE TWO DATAFRAMES
# ============================================================

def merge_component(base, other, name):
    before = len(base)

    base = base.merge(
        other,
        on="SEQN",
        how="left",
        suffixes=("", f"_{name}")
    )

    after = len(base)

    print(f"After {name}: {base.shape}")
    print(f"  Participants before: {before}")
    print(f"  Participants after:  {after}")

    return base


# ============================================================
# MAIN MERGE
# ============================================================

def load_and_merge():

    # --------------------------------------------------------
    # Existing files
    # --------------------------------------------------------

    lux = load_component("P_LUX.XPT")
    bio = load_component("P_BIOPRO.XPT")
    ghb = load_component("P_GHB.XPT")
    chol = load_component("P_TCHOL.XPT")
    hdl = load_component("P_HDL.XPT")
    trig = load_component("P_TRIGLY.XPT")
    diet = load_component("P_DR1TOT.XPT")

    # --------------------------------------------------------
    # Demographic and body-measure files
    # --------------------------------------------------------

    demo = load_component("DEMO_J.XPT")
    bmx = load_component("BMX_J.XPT")

    # --------------------------------------------------------
    # Start with LUX because CAP is our main outcome
    # --------------------------------------------------------

    df = lux.copy()

    print("\nStarting dataset:")
    print(df.shape)

    # --------------------------------------------------------
    # Merge each component using SEQN
    # LEFT JOIN preserves the LUX/CAP population.
    # --------------------------------------------------------

    df = merge_component(df, bio, "BIOPRO")
    df = merge_component(df, ghb, "GHB")
    df = merge_component(df, chol, "TCHOL")
    df = merge_component(df, hdl, "HDL")
    df = merge_component(df, trig, "TRIGLY")
    df = merge_component(df, diet, "DR1TOT")
    df = merge_component(df, demo, "DEMO")
    df = merge_component(df, bmx, "BMX")

    return df


# ============================================================
# CHECK IMPORTANT VARIABLES
# ============================================================

def check_key_columns(df):

    print("\n" + "=" * 60)
    print("KEY VARIABLE CHECK")
    print("=" * 60)

    key_columns = [
        # Participant ID
        "SEQN",

        # Liver steatosis
        "LUXCAPM",

        # Demographics
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",

        # Body measurements
        "BMXBMI",
        "BMXWAIST",

        # Laboratory variables
        "LBXGH",
        "LBXTC",
        "LBDHDL",
        "LBXTR",
    ]

    for column in key_columns:

        if column in df.columns:
            print(f"✓ {column}: present")
        else:
            print(f"✗ {column}: NOT FOUND")


# ============================================================
# MISSINGNESS CHECK
# ============================================================

def check_missingness(df):

    print("\n" + "=" * 60)
    print("MISSING VALUES IN KEY VARIABLES")
    print("=" * 60)

    key_columns = [
        "LUXCAPM",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "BMXBMI",
        "BMXWAIST",
        "LBXGH",
        "LBXTC",
        "LBDHDL",
        "LBXTR",
    ]

    existing = [
        col for col in key_columns
        if col in df.columns
    ]

    missing = df[existing].isnull().sum()

    result = pd.DataFrame({
        "Missing": missing,
        "Missing_%": (missing / len(df) * 100).round(2)
    })

    print(result)


# ============================================================
# FINAL DATASET CHECK
# ============================================================

def final_check(df):

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(f"Shape: {df.shape}")
    print(f"Unique participants: {df['SEQN'].nunique()}")
    print(f"Duplicate SEQN: {df['SEQN'].duplicated().sum()}")

    if df["SEQN"].duplicated().sum() != 0:
        raise ValueError("Duplicate SEQN values detected!")

    print("\nFirst 20 columns:")
    print(df.columns[:20].tolist())


# ============================================================
# SAVE
# ============================================================

def save_dataset(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NHANES DATA MERGE")
    print("=" * 60)

    df = load_and_merge()

    check_key_columns(df)

    check_missingness(df)

    final_check(df)

    save_dataset(df)