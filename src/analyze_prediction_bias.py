import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

PRED_FILE = "data/results/final/test_predictions.csv"

df = pd.read_csv(PRED_FILE)

actual = df["Actual_CAP"].to_numpy()
predicted = df["Tuned XGBoost"].to_numpy()

residual = actual - predicted

print("=" * 60)
print("PREDICTION BIAS / REGRESSION-TO-THE-MEAN ANALYSIS")
print("=" * 60)

print(f"Actual mean CAP     = {actual.mean():.2f}")
print(f"Predicted mean CAP  = {predicted.mean():.2f}")
print(f"Mean residual       = {residual.mean():.2f}")

print(f"\nCorrelation(actual, predicted) = "
      f"{np.corrcoef(actual, predicted)[0,1]:.4f}")

print(f"Correlation(actual, residual)  = "
      f"{np.corrcoef(actual, residual)[0,1]:.4f}")

# Store values for grouping
df["actual"] = actual
df["predicted"] = predicted
df["residual"] = residual

# Divide actual CAP into quartiles
df["CAP_group"] = pd.qcut(
    df["actual"],
    q=4,
    duplicates="drop"
)

print("\n" + "=" * 60)
print("BIAS BY ACTUAL CAP QUARTILE")
print("=" * 60)

for group, g in df.groupby("CAP_group", observed=True):

    print(f"\n{group}")
    print(f"n = {len(g)}")
    print(f"Actual mean     = {g.actual.mean():.2f}")
    print(f"Predicted mean  = {g.predicted.mean():.2f}")
    print(f"Mean residual   = {g.residual.mean():.2f}")
    print(f"MAE             = "
          f"{mean_absolute_error(g.actual, g.predicted):.2f}")
    print(f"R²              = "
          f"{r2_score(g.actual, g.predicted):.4f}")

print("\n" + "=" * 60)
print("EXTREME CAP BIAS")
print("=" * 60)

low = df[df["actual"] <= 180]
high = df[df["actual"] >= 350]

print("\nLOW CAP (<=180)")
print(f"n = {len(low)}")
print(f"Actual mean    = {low.actual.mean():.2f}")
print(f"Predicted mean = {low.predicted.mean():.2f}")
print(f"Mean residual  = {low.residual.mean():.2f}")

print("\nHIGH CAP (>=350)")
print(f"n = {len(high)}")
print(f"Actual mean    = {high.actual.mean():.2f}")
print(f"Predicted mean = {high.predicted.mean():.2f}")
print(f"Mean residual  = {high.residual.mean():.2f}")