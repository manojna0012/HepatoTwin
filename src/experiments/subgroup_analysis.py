"""Where the CAP regression R² ceiling comes from: age groups and cohort filters.

All results are 5-fold OOF on the pooled 2017-2023 cohort with the combined
model inputs (see combined_model.py). Three questions:

1. Per age group: is the model worse for older patients, or is R² lower
   because CAP varies less within the group? Reports CAP SD, RMSE, MAE, R² and
   ROC-AUC for CAP >= 248.
2. Cohort filters (exam reliability, age, CAP ceiling): which change R², and is
   the best one stable across CV seeds?
3. Can the 60+ group be improved (2017-2020-only history and medication items,
   a 60+-only model, upweighting, deeper trees)?

Findings (2026-10-16): 60+ R² 0.405 with RMSE 46.0 vs 42-44 dB/m for younger
adults; reliable exams (IQR < 40) and age 12-59 reach R² 0.584 (stable across
seeds and age cutoffs 50-65); no 60+ variant beat 0.406. Run:

    .venv/Scripts/python.exe -m src.experiments.subgroup_analysis
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.train_final_model import ROOT, TARGET
from src.cap_features import CAPFeatures, RAW_FEATURES
from src.experiments.combined_model import features, load_pooled, xgb
from src.experiments.extended_nhanes_features import (
    BLOOD_PRESSURE, BODY, CBC, DEMOGRAPHIC, FASTING, INFLAMMATION, QUESTIONNAIRE, RX,
    add_derived, build_extended, metrics)

OUTPUT_DIR = ROOT / 'data' / 'results' / 'experiments' / 'subgroup_analysis'
TUNED = dict(n_estimators=1500, learning_rate=0.01, colsample_bytree=0.4)
AGE_GROUPS = {'12-17': (12, 18), '18-39': (18, 40), '40-59': (40, 60), '60+': (60, 200)}


def oof(X, y, seed=42, weight=None, train_mask=None, **params):
    prediction = np.full(len(y), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        if train_mask is not None:
            tr, va = tr[train_mask[tr]], va[train_mask[va]]
        model = xgb(**(TUNED | params))
        model.fit(X.iloc[tr], y[tr], sample_weight=None if weight is None else weight[tr])
        prediction[va] = model.predict(X.iloc[va])
    return prediction


def with_2017_2020_items(data, X):
    """Append items only asked in 2017-2020 (NaN for 2021-2023 participants)."""
    old = build_extended()
    old = old.loc[old.LUAXSTAT.eq(1) & old[TARGET].notna()].reset_index(drop=True)
    X_old = add_derived(pd.concat([CAPFeatures().fit_transform(old[RAW_FEATURES]), old[
        BODY + DEMOGRAPHIC + FASTING + CBC + INFLAMMATION + BLOOD_PRESSURE + QUESTIONNAIRE + RX]], axis=1))
    extra = [c for c in X_old.columns if c not in X.columns]
    lookup = pd.concat([old[['SEQN']], X_old[extra]], axis=1)
    return pd.concat([X, data[['SEQN']].merge(lookup, on='SEQN', how='left')[extra]], axis=1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_pooled()
    X, y = features(data), data[TARGET].to_numpy(dtype=float)
    age, iqr = data.RIDAGEYR.to_numpy(), data.LUXCPIQR.to_numpy()

    # 1. Age groups.
    base = oof(X, y)
    rows = []
    for name, (lo, hi) in ({'all': (0, 200)} | AGE_GROUPS).items():
        m = (age >= lo) & (age < hi)
        rows.append(dict(group=name, n=int(m.sum()), cap_sd=float(y[m].std()), **metrics(y[m], base[m]),
                         roc_auc_cap248=roc_auc_score(y[m] >= 248, base[m])))
    by_age = pd.DataFrame(rows)
    by_age.to_csv(OUTPUT_DIR / 'metrics_by_age.csv', index=False)
    print(by_age.to_string(index=False, float_format=lambda v: f'{v:.3f}'), flush=True)

    # 2. Cohort filters, each trained and scored inside the filtered cohort.
    filters = {
        'all exams': np.ones(len(y), bool),
        'IQR<40': iqr < 40,
        'IQR<30': iqr < 30,
        'IQR<20': iqr < 20,
        'exclude CAP=400 ceiling': y < 400,
        'adults 18+': age >= 18,
        'age 12-59': age < 60,
        'IQR<40 & age<50': (iqr < 40) & (age < 50),
        'IQR<40 & age<60': (iqr < 40) & (age < 60),
        'IQR<40 & age<65': (iqr < 40) & (age < 65),
        'IQR<40 & age 60+': (iqr < 40) & (age >= 60),
    }
    rows = []
    for name, m in filters.items():
        Xm, ym = X[m].reset_index(drop=True), y[m]
        scores = [metrics(ym, oof(Xm, ym, seed=s))['r2'] for s in (42, 1, 7)]
        rows.append(dict(cohort=name, n=int(m.sum()), cap_sd=float(ym.std()),
                         r2_mean=float(np.mean(scores)), r2_min=min(scores), r2_max=max(scores)))
        print(f'{name:26s} n={m.sum():5d} R2={np.mean(scores):.4f} '
              f'({min(scores):.4f}-{max(scores):.4f})', flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'cohort_filters.csv', index=False)

    # 3. Attempts to improve the 60+ group.
    older = age >= 60
    X_more = with_2017_2020_items(data, X)
    variants = {
        'pooled model (current inputs)': oof(X, y),
        '+ 2017-2020 history/medication items': oof(X_more, y),
        '60+ only model': oof(X_more, y, train_mask=older),
        '60+ weighted 2x': oof(X_more, y, weight=np.where(older, 2.0, 1.0)),
        'deeper trees': oof(X_more, y, max_depth=6, min_child_weight=20, n_estimators=2000),
    }
    rows = []
    for name, prediction in variants.items():
        rows.append(dict(variant=name, **{f'{k}_60plus': v for k, v in
                                          metrics(y[older], prediction[older]).items()}))
        print(f'{name:40s} 60+ R2={rows[-1]["r2_60plus"]:.4f} RMSE={rows[-1]["rmse_60plus"]:.2f}',
              flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'older_adult_variants.csv', index=False)
    print(f'\nResults saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
