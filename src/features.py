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

    # Make SEQN consistent
    df["SEQN"] = df["SEQN"].astype(int)

    print(f"{filename}: {df.shape}")
    print(f"  Unique SEQN: {df['SEQN'].nunique()}")
    print(f"  Duplicate SEQN: {df['SEQN'].duplicated().sum()}")

    return df


# ============================================================
# MERGE
# ============================================================

def merge_component(base, other, name):

    before = len(base)

    merged = base.merge(
        other,
        on="SEQN",
        how="left"
    )

    print(f"After {name}: {merged.shape}")
    print(f"  Participants before: {before}")
    print(f"  Participants after:  {len(merged)}")

    return merged


# ============================================================
# LOAD + MERGE
# ============================================================

def load_and_merge():

    # --------------------------
    # Load datasets
    # --------------------------

    lux = load_component("P_LUX.XPT")
    bio = load_component("P_BIOPRO.XPT")
    ghb = load_component("P_GHB.XPT")
    chol = load_component("P_TCHOL.XPT")
    hdl = load_component("P_HDL.XPT")
    trig = load_component("P_TRIGLY.XPT")
    diet = load_component("P_DR1TOT.XPT")
    demo = load_component("P_DEMO.XPT")
    bmx = load_component("P_BMX.XPT")

    # --------------------------
    # Keep only required columns
    # --------------------------

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

    diet = diet[
        [
            "SEQN",
            "DR1TKCAL",
            "DR1TPROT",
            "DR1TCARB",
            "DR1TTFAT",
        ]
    ]

    # --------------------------
    # Merge
    # --------------------------

    df = lux.copy()

    print("\nStarting dataset:")
    print(df.shape)

    df = merge_component(df, bio, "BIOPRO")
    df = merge_component(df, ghb, "GHB")
    df = merge_component(df, chol, "TCHOL")
    df = merge_component(df, hdl, "HDL")
    df = merge_component(df, trig, "TRIGLY")
    df = merge_component(df, diet, "DIET")
    df = merge_component(df, demo, "DEMO")
    df = merge_component(df, bmx, "BMX")

    return df


# ============================================================
# CHECK KEY VARIABLES
# ============================================================

def check_key_columns(df):

    print("\n" + "=" * 60)
    print("KEY VARIABLE CHECK")
    print("=" * 60)

    key_columns = [
        "SEQN",
        "LUXCAPM",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "BMXBMI",
        "BMXWAIST",
        "LBXSATSI",
        "LBXSASSI",
        "LBXGH",
        "LBXTC",
        "LBDHDD",
        "LBXTR",
        "DR1TKCAL",
        "DR1TPROT",
        "DR1TCARB",
        "DR1TTFAT",
    ]

    for col in key_columns:
        if col in df.columns:
            print(f"✓ {col}")
        else:
            print(f"✗ {col}")


# ============================================================
# MISSING VALUES
# ============================================================

def check_missingness(df):

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    cols = [
        "LUXCAPM",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "BMXBMI",
        "BMXWAIST",
        "LBXSATSI",
        "LBXSASSI",
        "LBXGH",
        "LBXTC",
        "LBDHDD",
        "LBXTR",
        "DR1TKCAL",
        "DR1TPROT",
        "DR1TCARB",
        "DR1TTFAT",
    ]

    result = pd.DataFrame({
        "Missing": df[cols].isnull().sum(),
        "Missing %": (df[cols].isnull().mean() * 100).round(2)
    })

    print(result)


# ============================================================
# FINAL CHECK
# ============================================================

def final_check(df):

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print("Shape:", df.shape)
    print("Unique Participants:", df.SEQN.nunique())
    print("Duplicate SEQN:", df.SEQN.duplicated().sum())

    print("\nFirst 20 Columns:")
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
# MAIN
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