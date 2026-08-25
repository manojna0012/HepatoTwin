import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
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
    lux = pd.read_sas("data/raw/P_LUX.XPT")

    cap_cols = [c for c in lux.columns if "CAP" in c.upper()]

    print(cap_cols)
    print(lux.shape)
    # Inspect CAP (Controlled Attenuation Parameter)
    cap = lux["LUXCAPM"]

    print("\nCAP summary:")
    print(cap.describe())

    print("\nMissing CAP values:")
    print(cap.isna().sum())

    print("\nCAP percentiles:")
    print(cap.quantile([0, 0.25, 0.5, 0.75, 0.90, 0.95, 1.0]))


    cap = lux["LUXCAPM"].dropna()

    plt.hist(cap, bins=30)
    plt.xlabel("CAP (dB/m)")
    plt.ylabel("Number of participants")
    plt.title("NHANES CAP Distribution")
    plt.show()