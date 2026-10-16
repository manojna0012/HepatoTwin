"""Test whether additional NHANES components raise CAP regression R².

Every model-side experiment so far (XGBoost, CatBoost, MLP, stacking, feature
selection, calibration) plateaus near R² 0.51 on the same inputs, which points
to an information ceiling rather than a modelling one. This experiment adds
NHANES 2017-March 2020 components the pipeline never loaded (fasting insulin
and glucose, CBC, hs-CRP, blood pressure, extra body measures, medical,
weight-history and lifestyle questionnaires, prescriptions) and measures the
gain group by group on the same pooled 5-fold OOF protocol as
compare_catboost.py (9,021 participants, KFold seed 42, baseline R² 0.5135).

Excluded on purpose: every elastography field (LUX*) and self-reported liver
conditions (MCQ160L, MCQ170L, MCQ510*), which would act as label proxies.

Download the extra files into data/raw_extra/ first, from
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/<NAME>.xpt for
NAME in P_INS P_GLU P_CBC P_HSCRP P_FERTIN P_BPXO P_RXQ_RX and QUESTION_FILES.
Run from the project root:

    .venv/Scripts/python.exe -m src.experiments.extended_nhanes_features

Result (5-fold OOF): baseline R² 0.5135, all 128 inputs tuned 0.5240,
ROC-AUC for CAP >= 248 0.865 -> 0.869.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.train_final_model import ROOT, TARGET
from src.cap_features import CAPFeatures, RAW_FEATURES

RAW = ROOT / 'data' / 'raw'
EXTRA = ROOT / 'data' / 'raw_extra'
MERGED = ROOT / 'data' / 'processed' / 'nhanes_merged.csv'
EXTENDED = ROOT / 'data' / 'processed' / 'nhanes_extended.csv'
OUTPUT_DIR = ROOT / 'data' / 'results' / 'experiments' / 'extended_nhanes_features'
SEED = 42
FOLDS = 5

# NHANES "refused" / "don't know" codes. Single-digit items use 7/9; items whose
# valid answers reach 7 or 9 (counts, hours, frequency scales) use wider codes.
WIDE_MISSING = {
    'ALQ121': {77, 99}, 'ALQ130': {777, 999}, 'PAD680': {7777, 9999},
    'WHD050': {7777, 9999}, 'WHD110': {7777, 9999}, 'WHD140': {7777, 9999},
    'SLD012': set(), 'DMDBORN4': {77, 99},
}


def clean_missing_codes(data, columns):
    for column in columns:
        codes = WIDE_MISSING.get(column, {7, 9})
        data[column] = data[column].where(~data[column].isin(codes))
    return data

BODY = ['BMXHIP', 'BMXARMC', 'BMXARML', 'BMXLEG']
DEMOGRAPHIC = ['INDFMPIR', 'DMDEDUC2', 'DMDBORN4']
FASTING = ['LBXIN', 'LBXGLU']
CBC = ['LBXWBCSI', 'LBXLYPCT', 'LBXNEPCT', 'LBXMOPCT', 'LBXHGB', 'LBXHCT',
       'LBXMCVSI', 'LBXRDW', 'LBXPLTSI', 'LBXMPSI']
INFLAMMATION = ['LBXHSCRP', 'LBXFER']
BLOOD_PRESSURE = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3', 'BPXODI1', 'BPXODI2', 'BPXODI3', 'BPXOPLS1']
QUESTIONNAIRE = [
    'DIQ010', 'DIQ160', 'DIQ050', 'DIQ070',          # diabetes, prediabetes, insulin, pills
    'BPQ020', 'BPQ080', 'BPQ090D',                   # hypertension, cholesterol, statin advice
    'MCQ080', 'MCQ366A', 'MCQ366B', 'MCQ366C', 'MCQ366D',
    'MCQ371A', 'MCQ371B', 'MCQ371C', 'MCQ371D',      # overweight told / lifestyle advice
    'WHQ030', 'WHQ070', 'WHD050', 'WHD110', 'WHD140',  # weight perception and history
    'HUQ010', 'SLD012', 'SLQ030', 'SLQ040', 'SLQ050',
    'SMQ020', 'SMQ040', 'ALQ111', 'ALQ121', 'ALQ130', 'ALQ151',
    'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665', 'PAD680',
]
QUESTION_FILES = ['P_DIQ', 'P_BPQ', 'P_MCQ', 'P_WHQ', 'P_HUQ', 'P_SLQ', 'P_SMQ', 'P_ALQ', 'P_PAQ']

RX_PATTERNS = {
    'rx_statin': 'STATIN',
    'rx_metformin': 'METFORMIN',
    'rx_insulin': 'INSULIN',
    'rx_glp1_sglt2': 'GLUTIDE|GLIFLOZIN',
    'rx_sulfonylurea': 'GLIPIZIDE|GLIMEPIRIDE|GLYBURIDE',
    'rx_antihypertensive': 'PRIL$|SARTAN|AMLODIPINE|HYDROCHLOROTHIAZIDE|CHLORTHALIDONE|OLOL$',
    'rx_antipsychotic': 'QUETIAPINE|OLANZAPINE|RISPERIDONE|ARIPIPRAZOLE',
    'rx_steroid': 'PREDNISONE|METHYLPREDNISOLONE|DEXAMETHASONE',
}
RX = ['rx_count'] + list(RX_PATTERNS)

DERIVED = ['WHR', 'HOMA_IR', 'TyG', 'mean_SBP', 'mean_DBP', 'weight_change_1y', 'weight_from_max']


def read_xpt(name, folder=EXTRA, columns=None):
    frame = pd.read_sas(folder / f'{name}.xpt')
    keep = ['SEQN'] + [c for c in (columns or frame.columns) if c != 'SEQN' and c in frame]
    return frame[keep]


def prescription_features():
    rx = read_xpt('P_RXQ_RX', columns=['RXDUSE', 'RXDDRUG'])
    drugs = rx.RXDDRUG.fillna(b'').map(
        lambda v: v.decode('latin-1') if isinstance(v, bytes) else str(v)).str.upper()
    per_person = pd.DataFrame({'SEQN': rx.SEQN})
    for name, pattern in RX_PATTERNS.items():
        per_person[name] = drugs.str.split(';').map(
            lambda items: any(pd.Series(items).str.strip().str.contains(pattern, regex=True)))
    out = per_person.groupby('SEQN').max().astype(float)
    counted = drugs.str.len().gt(0) & rx.RXDUSE.eq(1)
    out['rx_count'] = counted.groupby(rx.SEQN).sum().astype(float)
    # Answered "no prescriptions" (RXDUSE == 2) is a real zero, not missing.
    return out.reset_index()


def build_extended():
    base = pd.read_csv(MERGED)
    parts = [
        read_xpt('P_BMX', RAW, BODY),
        read_xpt('P_DEMO', RAW, DEMOGRAPHIC),
        read_xpt('P_INS', columns=['LBXIN']),
        read_xpt('P_GLU', columns=['LBXGLU']),
        read_xpt('P_CBC', columns=CBC),
        read_xpt('P_HSCRP', columns=['LBXHSCRP']),
        read_xpt('P_FERTIN', columns=['LBXFER']),
        read_xpt('P_BPXO', columns=BLOOD_PRESSURE),
        prescription_features(),
    ]
    for name in QUESTION_FILES:
        parts.append(read_xpt(name, columns=QUESTIONNAIRE))
    data = base
    for part in parts:
        data = data.merge(part, on='SEQN', how='left', validate='one_to_one')
    return clean_missing_codes(data, QUESTIONNAIRE + ['DMDEDUC2', 'DMDBORN4'])


def add_derived(X):
    X = X.copy()
    X['WHR'] = X.BMXWAIST / X.BMXHIP
    X['HOMA_IR'] = X.LBXIN * X.LBXGLU / 405
    # Uses fasting glucose when available, otherwise the standard biochemistry glucose.
    glucose = X.LBXGLU.fillna(X.LBXSGL)
    X['TyG'] = np.log(X.LBXSTR.where(X.LBXSTR > 0) * glucose.where(glucose > 0) / 2)
    X['mean_SBP'] = X[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
    X['mean_DBP'] = X[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1)
    # Self-reported weight is in pounds; BMXWT is measured kg.
    X['weight_change_1y'] = X.BMXWT - X.WHD050 * 0.4536
    X['weight_from_max'] = X.BMXWT - X.WHD140 * 0.4536
    return X.replace([np.inf, -np.inf], np.nan)


FEATURE_SETS = {
    'baseline_36': [],
    '+body': BODY + ['WHR'],
    '+fasting_labs': FASTING + ['HOMA_IR', 'TyG'],
    '+cbc_inflammation': CBC + INFLAMMATION,
    '+blood_pressure': BLOOD_PRESSURE + ['mean_SBP', 'mean_DBP'],
    '+questionnaire': QUESTIONNAIRE + DEMOGRAPHIC + ['weight_change_1y', 'weight_from_max'],
    '+prescriptions': RX,
}


def make_model(n_estimators=600, learning_rate=0.02, colsample_bytree=0.6, **extra):
    # Same hyperparameters as the compare_catboost.py baseline; NaN handled natively.
    return XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators,
                        learning_rate=learning_rate, max_depth=4, min_child_weight=8,
                        subsample=0.8, colsample_bytree=colsample_bytree, reg_lambda=3,
                        reg_alpha=0.5, random_state=SEED, n_jobs=-1, **extra)


def metrics(y, prediction):
    return dict(r2=float(r2_score(y, prediction)),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                mae=float(mean_absolute_error(y, prediction)))


def oof_predict(X, y, folds, model_factory):
    prediction = np.full(len(y), np.nan)
    for train_idx, valid_idx in folds:
        model = model_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        prediction[valid_idx] = model.predict(X.iloc[valid_idx])
    return prediction


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_extended()
    data.to_csv(EXTENDED, index=False)
    cohort = data.loc[data.LUAXSTAT.eq(1) & data[TARGET].notna()].reset_index(drop=True)
    if cohort.SEQN.duplicated().any():
        raise ValueError('Duplicate SEQN values in eligible cohort.')
    y = cohort[TARGET].astype(float).to_numpy()
    engineered = CAPFeatures().fit_transform(cohort[RAW_FEATURES])
    extra_columns = (BODY + DEMOGRAPHIC + FASTING + CBC + INFLAMMATION
                     + BLOOD_PRESSURE + QUESTIONNAIRE + RX)
    X_all = add_derived(pd.concat([engineered, cohort[extra_columns]], axis=1))
    folds = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(X_all))
    print(f'Eligible participants: {len(cohort)}', flush=True)

    columns = list(engineered.columns)
    rows, predictions = [], {'SEQN': cohort.SEQN, 'actual_CAP': y}
    for name, added in FEATURE_SETS.items():
        columns = columns + [c for c in added if c not in columns]
        prediction = oof_predict(X_all[columns], y, folds, make_model)
        predictions[name] = prediction
        rows.append(dict(model=name, n_features=len(columns), **metrics(y, prediction)))
        print(f'{name:22s} features={len(columns):3d} R2={rows[-1]["r2"]:.4f} '
              f'RMSE={rows[-1]["rmse"]:.2f}', flush=True)

    # Wider feature set: lower column sampling, more trees at a lower learning rate.
    name = 'all_tuned'
    prediction = oof_predict(X_all[columns], y, folds, lambda: make_model(
        n_estimators=1500, learning_rate=0.01, colsample_bytree=0.4))
    predictions[name] = prediction
    rows.append(dict(model=name, n_features=len(columns), **metrics(y, prediction)))
    print(f'{name:22s} features={len(columns):3d} R2={rows[-1]["r2"]:.4f} '
          f'RMSE={rows[-1]["rmse"]:.2f}', flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / 'summary_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / 'oof_predictions.csv', index=False)
    final = make_model(n_estimators=1500, learning_rate=0.01, colsample_bytree=0.4)
    final.fit(X_all[columns], y)
    pd.Series(final.get_booster().get_score(importance_type='gain'), name='gain') \
        .sort_values(ascending=False).to_csv(OUTPUT_DIR / 'feature_importance_gain.csv')
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(dict(
            target=TARGET, participants=int(len(cohort)), folds=FOLDS, seed=SEED,
            evaluation='Pooled 5-fold OOF on the same folds as compare_catboost.py',
            feature_sets={k: v for k, v in FEATURE_SETS.items()}, derived=DERIVED,
            excluded='All LUX elastography fields; self-reported liver conditions',
            note='Feature sets are cumulative. Internal CV only, not external validation.',
        ), f, indent=2)
    print(f'\nResults saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
