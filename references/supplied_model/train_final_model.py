"""
HepatoTwin - final patient-state model.

Predicts Controlled Attenuation Parameter (LUXCAPM, dB/m) from NHANES
anthropometric, biochemistry and 24h dietary-recall features.

Key differences from baseline_xgboost.py:
  * reads nhanes_merged.csv directly - imputation happens INSIDE the
    pipeline, so test-fold medians never leak into training
  * expanded biochemistry feature set + engineered clinical ratios
  * hyperparameters tuned for this sample size (depth 4, lr 0.02)
  * repeated CV instead of a single split, so fold-to-fold sd is visible
  * reports AUROC at clinical steatosis thresholds alongside MAE/RMSE/R2
  * saves the FITTED PIPELINE (imputer + encoder + model) as one artifact,
    and writes out the held-out SEQNs so demo.py can't cheat

Usage:
    python train_final_model.py                # 5-fold CV, 1 repeat
    python train_final_model.py --repeats 3    # slower, tighter sd
    python train_final_model.py --no-cv        # skip CV, just fit + test
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/nhanes_merged.csv")
OUTPUT_DIR = Path("data/results")

TARGET = "LUXCAPM"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Device measurement range. CAP is censored at both ends.
CAP_MIN = 100.0
CAP_MAX = 400.0

# Clinical steatosis grade cut-offs (dB/m), used for the AUROC report.
CAP_THRESHOLDS = {"S1+": 248, "S2+": 268, "S3": 280}


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------
# NOTE: no LU* column may ever appear below. LUXCPIQR is the IQR of the
# CAP measurement itself and LUXSMED / LUXSIQR are liver stiffness from
# the same elastography exam - any of them is target leakage.

BASE_NUMERIC = [
    # Anthropometric
    "RIDAGEYR",
    "BMXBMI",
    "BMXWAIST",
    # Liver enzymes
    "LBXSATSI",   # ALT
    "LBXSASSI",   # AST
    # Glycaemic
    "LBXGH",      # HbA1c
    # Lipids
    "LBXTC",
    "LBDHDD",
    # Diet
    "DR1TKCAL",
    "DR1TPROT",
    "DR1TCARB",
    "DR1TTFAT",
]

EXTRA_NUMERIC = [
    "BMXWT",
    "BMXHT",
    "LBXSGTSI",   # GGT - strong steatosis marker
    "LBXSTR",     # serum triglycerides (7.8% missing, unlike LBXTR at 54%)
    "LBXSGL",     # glucose
    "LBXSUA",     # uric acid
    "LBXSAL",     # albumin
    "LBXSTP",     # total protein
    "LBXSGB",     # globulin
    "LBXSAPSI",   # alkaline phosphatase
    "LBXSTB",     # total bilirubin
    "LBXSCR",     # creatinine
    "LBXSBU",     # urea nitrogen
    "LBXSCA",
    "LBXSPH",
    "LBXSKSI",
    "LBXSNASI",
    "LBXSCLSI",
    "LBXSC3SI",
    "LBXSIR",
    "LBXSLDSI",   # LDH
    "LBXSCK",
]

ENGINEERED = [
    "WHtR",
    "AST_ALT",
    "TG_HDL",
    "TC_HDL",
    "HSI",
    "FLI",
    "pct_prot",
    "pct_carb",
    "pct_fat",
    "kcal_per_kg",
]

CATEGORICAL = ["RIAGENDR", "RIDRETH3"]

NUMERIC = BASE_NUMERIC + EXTRA_NUMERIC + ENGINEERED
FEATURES = NUMERIC + CATEGORICAL


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
# Baseline used depth 6 / lr 0.05, which overfits ~7,200 training rows.

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=3.0,
    reg_alpha=0.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """Add clinical ratios and energy-normalised diet features.

    Must be importable by demo.py so that a single patient row goes
    through exactly the same transformation as the training data.
    """
    d = df.copy()

    # Central adiposity, scale-free
    d["WHtR"] = d["BMXWAIST"] / d["BMXHT"]

    # Liver enzyme and lipid ratios
    d["AST_ALT"] = d["LBXSASSI"] / d["LBXSATSI"]
    d["TG_HDL"] = d["LBXSTR"] / d["LBDHDD"]
    d["TC_HDL"] = d["LBXTC"] / d["LBDHDD"]

    # Hepatic Steatosis Index (Lee et al.)
    d["HSI"] = (
        8.0 * (d["LBXSATSI"] / d["LBXSASSI"])
        + d["BMXBMI"]
        + 2.0 * (d["RIAGENDR"] == 2)
    )

    # Fatty Liver Index linear predictor (Bedogni et al.), TG in mg/dL
    z = (
        0.953 * np.log(d["LBXSTR"].clip(lower=1))
        + 0.139 * d["BMXBMI"]
        + 0.718 * np.log(d["LBXSGTSI"].clip(lower=1))
        + 0.053 * d["BMXWAIST"]
        - 15.745
    )
    d["FLI"] = 100.0 * np.exp(z) / (1.0 + np.exp(z))

    # Macronutrients as percent of energy. Absolute grams are collinear
    # with DR1TKCAL, so the model otherwise sees "ate a lot" four times.
    kcal = d["DR1TKCAL"].replace(0, np.nan)
    d["pct_prot"] = 4.0 * d["DR1TPROT"] / kcal
    d["pct_carb"] = 4.0 * d["DR1TCARB"] / kcal
    d["pct_fat"] = 9.0 * d["DR1TTFAT"] / kcal
    d["kcal_per_kg"] = kcal / d["BMXWT"]

    return d


# ============================================================
# DATA
# ============================================================

def load_cohort(input_file=INPUT_FILE):
    """Eligible participants: valid elastography exam and a CAP value."""
    df = pd.read_csv(input_file)
    n_raw = len(df)

    df = df[df["LUAXSTAT"] == 1].copy()
    n_exam = len(df)

    df = df.dropna(subset=[TARGET]).copy()

    print(f"Raw rows                : {n_raw}")
    print(f"After LUAXSTAT == 1     : {n_exam}")
    print(f"After CAP present       : {len(df)}")

    missing = [c for c in FEATURES if c not in engineer_features(df).columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return engineer_features(df)


def build_pipeline(params=None):
    """Imputation and encoding live inside the pipeline, not upstream."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ]),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(**(params or XGB_PARAMS))),
    ])


# ============================================================
# EVALUATION
# ============================================================

def run_cv(X, y, repeats):
    cv = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE)

    scores = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=5,
    )

    mae, rmse, r2 = -scores["test_mae"], -scores["test_rmse"], scores["test_r2"]

    print(f"\n5-fold x {repeats} repeat(s), {len(mae)} fits")
    print(f"MAE  : {mae.mean():.2f}  (sd {mae.std():.2f}) dB/m")
    print(f"RMSE : {rmse.mean():.2f}  (sd {rmse.std():.2f}) dB/m")
    print(f"R2   : {r2.mean():.3f} (sd {r2.std():.3f})")
    print(
        "\nFold-to-fold sd is the yardstick: any model comparison closer "
        f"than ~{2 * r2.std():.3f} R2 is noise, not a result."
    )

    return {"mae": mae.mean(), "mae_sd": mae.std(), "rmse": rmse.mean(),
            "rmse_sd": rmse.std(), "r2": r2.mean(), "r2_sd": r2.std()}


def report_test(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), CAP_MIN, CAP_MAX)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET")
    print("=" * 60)
    print(f"n     : {len(y_true)}")
    print(f"MAE   : {mae:.2f} dB/m")
    print(f"RMSE  : {rmse:.2f} dB/m")
    print(f"R2    : {r2:.3f}")

    # Shrinkage toward the mean is the model's main limitation.
    print(f"\nActual sd    : {y_true.std():.1f} dB/m")
    print(f"Predicted sd : {y_pred.std():.1f} dB/m")
    print("Predictions are compressed toward the mean; most error is in the tails.")

    # Error by decile of true CAP
    deciles = pd.qcut(y_true, 10, duplicates="drop")
    by_decile = (
        pd.DataFrame({"abs_err": np.abs(y_true - y_pred), "bin": deciles})
        .groupby("bin", observed=True)["abs_err"]
        .agg(["mean", "count"])
        .round(1)
    )
    print("\nMAE by decile of true CAP:")
    print(by_decile.to_string())

    # Censored at the device ceiling - irreducible error
    at_ceiling = y_true >= CAP_MAX - 1
    if at_ceiling.any():
        print(
            f"\n{at_ceiling.sum()} participants sit at the {CAP_MAX:.0f} dB/m device "
            f"ceiling (censored target), MAE {mean_absolute_error(y_true[at_ceiling], y_pred[at_ceiling]):.1f}. "
            "No tuning fixes those."
        )

    # Threshold performance: the honest headline for this project
    print("\nAs a steatosis-grade classifier (regression output as score):")
    auroc = {}
    for label, thr in CAP_THRESHOLDS.items():
        pos = y_true >= thr
        if pos.sum() < 10 or (~pos).sum() < 10:
            continue
        auroc[label] = roc_auc_score(pos, y_pred)
        print(f"  {label:4s} (CAP >= {thr}) : AUROC {auroc[label]:.3f}  prevalence {pos.mean():.2f}")

    thr = CAP_THRESHOLDS["S2+"]
    tn, fp, fn, tp = confusion_matrix(y_true >= thr, y_pred >= thr).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print(f"\nConfusion matrix at CAP >= {thr} dB/m (predicted vs actual):")
    print(f"  TP {tp:5d}   FP {fp:5d}")
    print(f"  FN {fn:5d}   TN {tn:5d}")
    print(f"  sensitivity {sens:.3f}   specificity {spec:.3f}")

    metrics = {"n_test": len(y_true), "mae": mae, "rmse": rmse, "r2": r2,
               "sensitivity_s2": sens, "specificity_s2": spec}
    metrics.update({f"auroc_{k}": v for k, v in auroc.items()})
    return metrics


def feature_importance(pipeline):
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    names = NUMERIC + list(encoder.get_feature_names_out(CATEGORICAL))
    imp = pd.DataFrame({
        "Feature": names,
        "Importance": pipeline.named_steps["model"].feature_importances_,
    })
    return imp.sort_values("Importance", ascending=False).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=1,
                        help="CV repeats of 5-fold. More = tighter sd, slower.")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation, only fit and test.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HEPATOTWIN FINAL PATIENT-STATE MODEL")
    print("=" * 60)

    df = load_cohort(args.input)
    X = df[FEATURES]
    y = df[TARGET]

    print(f"\nFeatures: {len(FEATURES)} "
          f"({len(BASE_NUMERIC)} base + {len(EXTRA_NUMERIC)} extra labs "
          f"+ {len(ENGINEERED)} engineered + {len(CATEGORICAL)} categorical)")
    print(f"Target  : {TARGET}  (mean {y.mean():.1f}, sd {y.std():.1f} dB/m)")

    # Split once, stratified on CAP quintile so the tails aren't lopsided.
    strata = pd.qcut(y, 5, labels=False, duplicates="drop")
    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
    )

    X_train, y_train = X.loc[idx_train], y.loc[idx_train]
    X_test, y_test = X.loc[idx_test], y.loc[idx_test]
    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")

    cv_metrics = None
    if not args.no_cv:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION (training set only)")
        print("=" * 60)
        cv_metrics = run_cv(X_train, y_train, args.repeats)

    print("\nFitting final pipeline on the training set...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    test_metrics = report_test(y_test, y_pred)

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------
    model_file = args.outdir / "final_patient_state_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURES,
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "target": TARGET,
            "cap_range": (CAP_MIN, CAP_MAX),
            "cap_thresholds": CAP_THRESHOLDS,
            "test_seqn": df.loc[idx_test, "SEQN"].astype(int).tolist(),
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
        },
        model_file,
    )

    imp = feature_importance(pipeline)
    imp.to_csv(args.outdir / "final_feature_importance.csv", index=False)

    pd.DataFrame({
        "SEQN": df.loc[idx_test, "SEQN"].astype(int).values,
        "Actual_CAP": y_test.values,
        "Predicted_CAP": np.clip(y_pred, CAP_MIN, CAP_MAX),
    }).to_csv(args.outdir / "final_test_predictions.csv", index=False)

    rows = [{"metric": k, "value": v} for k, v in test_metrics.items()]
    if cv_metrics:
        rows += [{"metric": f"cv_{k}", "value": v} for k, v in cv_metrics.items()]
    pd.DataFrame(rows).to_csv(args.outdir / "final_metrics.csv", index=False)

    print("\nTop 12 features:")
    print(imp.head(12).to_string(index=False))

    print("\nSaved:")
    for p in ["final_patient_state_pipeline.joblib", "final_feature_importance.csv",
              "final_test_predictions.csv", "final_metrics.csv"]:
        print(f"  {args.outdir / p}")


if __name__ == "__main__":
    main()
