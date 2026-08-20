import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")


def load_component(filename):
    return pd.read_sas(RAW_DIR / filename)


def load_and_merge():
    lux = load_component("P_LUX.XPT")
    bio = load_component("P_BIOPRO.XPT")
    ghb = load_component("P_GHB.XPT")
    chol = load_component("P_TCHOL.XPT")
    hdl = load_component("P_HDL.XPT")
    trig = load_component("P_TRIGLY.XPT")
    diet = load_component("P_DR1TOT.XPT")

    dfs = [lux, bio, ghb, chol, hdl, trig, diet]

    df = dfs[0]

    for other in dfs[1:]:
        df = df.merge(other, on="SEQN", how="inner")

    return df


if __name__ == "__main__":
    df = load_and_merge()

    print(f"Merged shape: {df.shape}")
    print(f"Duplicate SEQN values: {df['SEQN'].duplicated().sum()}")

    df.to_csv("data/nhanes_merged.csv", index=False)

    print("\nFirst 20 columns:")
    print(df.columns[:20].tolist())

    print("\nMissing values in some key columns:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))