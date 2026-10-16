"""Test two new data sources for CAP regression R².

1. More participants: the NHANES August 2021-August 2023 cycle (suffix _L) has
   the same FibroScan CAP exam and variable names, adding ~6,300 eligible people
   to the 9,021 in 2017-March 2020. Pooled 5-fold OOF on the combined cohort;
   R² is also reported on the original 9,021 so it compares with the 0.5135
   baseline.
2. DXA body composition (2017-2018 only, ages 8-59): visceral, subcutaneous,
   android and gynoid fat from DXXAG_J and whole-body fat from DXX_J. Scored on
   the participants who have a scan, against the baseline on the same people.

Files go in data/raw_2021/ (from .../Public/2021/DataFiles/<NAME>.xpt) and
data/raw_extra/ (DXXAG_J, DXX_J from .../Public/2017/DataFiles/). Run:

    .venv/Scripts/python.exe -m src.experiments.more_data_experiment
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.train_final_model import ROOT, TARGET
from src.cap_features import CAPFeatures, RAW_FEATURES
from src.experiments.extended_nhanes_features import SEED, metrics, make_model, oof_predict

RAW_2021 = ROOT / 'data' / 'raw_2021'
RAW_2017 = ROOT / 'data' / 'raw_2017'
MERGED = ROOT / 'data' / 'processed' / 'nhanes_merged.csv'
OUTPUT_DIR = ROOT / 'data' / 'results' / 'experiments' / 'more_data'
CORE_FILES = ['LUX', 'DEMO', 'BMX', 'BIOPRO', 'GHB', 'HDL', 'TCHOL', 'DR1TOT']
DXA = ['DXXVFATM', 'DXXSATM', 'DXXTATM', 'DXXANFM', 'DXXGYFM', 'DXXAGRAT',
       'DXXAPFAT', 'DXXGPFAT', 'DXDTOPF', 'DXDTOFAT', 'DXDTOLE']


def eligible(frame):
    frame = frame.loc[frame.LUAXSTAT.eq(1) & frame[TARGET].notna()].reset_index(drop=True)
    if frame.SEQN.duplicated().any():
        raise ValueError('Duplicate SEQN values in eligible cohort.')
    return frame


def load_cycle(folder, suffix):
    data = None
    for name in CORE_FILES:
        part = pd.read_sas(folder / f'{name}_{suffix}.xpt')
        if data is not None:
            # Survey weights repeat across lab files; keep the first copy.
            part = part[['SEQN'] + [c for c in part if c not in data]]
        data = part if data is None else data.merge(part, on='SEQN', how='left', validate='one_to_one')
    return eligible(data)


def load_dxa():
    agg = pd.read_sas(RAW_2017 / 'DXXAG_J.xpt')
    whole = pd.read_sas(RAW_2017 / 'DXX_J.xpt')
    agg = agg.loc[agg.DXXAGST.eq(1)]
    whole = whole.loc[whole.DXAEXSTS.eq(1)]
    dxa = agg.merge(whole, on='SEQN', how='outer')
    return dxa[['SEQN'] + DXA]


def report(rows, name, y, prediction, mask=None):
    mask = np.ones(len(y), bool) if mask is None else mask
    rows.append(dict(model=name, participants=int(mask.sum()), **metrics(y[mask], prediction[mask])))
    print(f'{name:45s} n={mask.sum():5d} R2={rows[-1]["r2"]:.4f} RMSE={rows[-1]["rmse"]:.2f}', flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    old = eligible(pd.read_csv(MERGED))
    new = load_cycle(RAW_2021, 'L')
    old['cycle_2021'], new['cycle_2021'] = 0.0, 1.0
    print(f'2017-2020: {len(old)}  2021-2023: {len(new)}', flush=True)

    # 1. Original cohort only, reproducing the compare_catboost.py folds.
    X_old = CAPFeatures().fit_transform(old[RAW_FEATURES])
    y_old = old[TARGET].astype(float).to_numpy()
    folds_old = list(KFold(5, shuffle=True, random_state=SEED).split(X_old))
    baseline = oof_predict(X_old, y_old, folds_old, make_model)
    report(rows, 'baseline (2017-2020 only)', y_old, baseline)

    # Same validation folds on the old cohort; every 2021-2023 participant is added to training.
    both = pd.concat([old, new], ignore_index=True)
    X_both = CAPFeatures().fit_transform(both[RAW_FEATURES])
    X_both['cycle_2021'] = both.cycle_2021.to_numpy()
    y_both = both[TARGET].astype(float).to_numpy()
    new_idx = np.arange(len(old), len(both))
    folds_aug = [(np.concatenate([tr, new_idx]), va) for tr, va in folds_old]
    augmented = np.full(len(both), np.nan)
    for tr, va in folds_aug:
        model = make_model()
        model.fit(X_both.iloc[tr], y_both[tr])
        augmented[va] = model.predict(X_both.iloc[va])
    report(rows, 'train + 2021-2023, scored on 2017-2020', y_old, augmented[:len(old)])

    # Fully pooled cohort, 5-fold OOF over everyone.
    folds_all = list(KFold(5, shuffle=True, random_state=SEED).split(X_both))
    pooled = oof_predict(X_both, y_both, folds_all, make_model)
    report(rows, 'pooled 2017-2023, scored on all', y_both, pooled)
    pooled_tuned = oof_predict(X_both, y_both, folds_all, lambda: make_model(
        n_estimators=1500, learning_rate=0.01, colsample_bytree=0.5))
    report(rows, 'pooled 2017-2023 tuned, scored on all', y_both, pooled_tuned)

    # 2. DXA inside the 2017-2018 cycle; NaN for people without a scan.
    cycle = load_cycle(RAW_2017, 'J')
    X_j = CAPFeatures().fit_transform(cycle[RAW_FEATURES])
    y_j = cycle[TARGET].astype(float).to_numpy()
    folds_j = list(KFold(5, shuffle=True, random_state=SEED).split(X_j))
    dxa = cycle[['SEQN']].merge(load_dxa(), on='SEQN', how='left', validate='one_to_one')
    has_dxa = dxa[DXA].notna().any(axis=1).to_numpy()
    X_dxa = pd.concat([X_j, dxa[DXA]], axis=1)
    X_dxa['VAT_per_height'] = X_dxa.DXXVFATM / X_dxa.BMXHT
    baseline_j = oof_predict(X_j, y_j, folds_j, make_model)
    with_dxa = oof_predict(X_dxa, y_j, folds_j, make_model)
    report(rows, '2017-2018 baseline, scored on DXA subset', y_j, baseline_j, has_dxa)
    report(rows, '2017-2018 +DXA, scored on DXA subset', y_j, with_dxa, has_dxa)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'summary_metrics.csv', index=False)
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(dict(target=TARGET, seed=SEED, core_files=CORE_FILES, dxa_columns=DXA,
                       participants_2017_2020=int(len(old)), participants_2021_2023=int(len(new)),
                       participants_with_dxa=int(has_dxa.sum()),
                       note='Internal CV only. Hyperparameters match compare_catboost.py.'), f, indent=2)
    print(f'\nResults saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
