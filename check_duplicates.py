import pandas as pd

files = [
    "P_LUX",
    "P_BIOPRO",
    "P_GHB",
    "P_TCHOL",
    "P_HDL",
    "P_TRIGLY",
    "P_DR1TOT",
    "P_DEMO",
    "P_BMX",
]

print("=" * 70)
print("CHECKING DUPLICATE SEQN VALUES")
print("=" * 70)

for name in files:

    df = pd.read_sas(f"data/raw/{name}.XPT")

    total_rows = len(df)
    unique_seqn = df["SEQN"].nunique()
    duplicates = df["SEQN"].duplicated().sum()

    print(
        f"{name:12} | "
        f"Rows: {total_rows:6} | "
        f"Unique SEQN: {unique_seqn:6} | "
        f"Duplicates: {duplicates}"
    )