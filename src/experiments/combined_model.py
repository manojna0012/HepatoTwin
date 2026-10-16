"""Combined CAP regression: more participants, more inputs, exam-quality handling.

Combines the three levers that each moved R² in earlier experiments:

1. Participants from both FibroScan cycles: 2017-March 2020 (P_ files) and
   August 2021-August 2023 (_L files), about 15,300 eligible people.
2. Extra NHANES inputs present in both cycles: hip/arm circumference, fasting
   insulin and glucose (HOMA-IR, TyG), CBC, hs-CRP, ferritin, blood pressure,
   diabetes/hypertension/lifestyle questionnaire items.
3. Exam reliability: CAP IQR >= 40 dB/m marks a less reliable exam. Candidates
   either train on every exam, train only on reliable exams, or down-weight
   unreliable ones.

Design: a 20% test set (stratified by cycle, seed 42) is set aside first.
Candidates are compared by 5-fold CV on the remaining 80% only; the winner is
refitted on all training rows and scored once on the test set. Test R² is
reported for the full test set and for its reliable-exam subset, alongside the
original 36-input baseline trained on the same rows.

Elastography fields (LUX*) and self-reported liver conditions are never inputs.
Needs the files from extended_nhanes_features.py and more_data_experiment.py,
plus the _L questionnaire files in data/raw_2021/. Run from the project root:

    .venv/Scripts/python.exe -m src.experiments.combined_model
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from src.train_final_model import ROOT, TARGET
from src.cap_features import CAPFeatures, RAW_FEATURES
from src.experiments.extended_nhanes_features import (
    BLOOD_PRESSURE, BODY, CBC, DEMOGRAPHIC, FASTING, SEED, clean_missing_codes,
    add_derived, build_extended, metrics)
from src.experiments.more_data_experiment import RAW_2021, eligible, load_cycle

OUTPUT_DIR = ROOT / 'data' / 'results' / 'experiments' / 'combined_model'
RELIABLE_IQR = 40

# Questionnaire items asked with the same codes in both cycles.
QUESTIONNAIRE_COMMON = [
    'DIQ010', 'DIQ160', 'DIQ050', 'DIQ070', 'BPQ020', 'BPQ080',
    'WHQ070', 'WHD050', 'HUQ010', 'SLD012', 'SMQ020', 'SMQ040',
    'ALQ111', 'ALQ121', 'ALQ130', 'ALQ151', 'PAD680',
]
COMMON_EXTRA = (BODY + DEMOGRAPHIC + FASTING + CBC + ['LBXHSCRP', 'LBXFER']
                + BLOOD_PRESSURE + QUESTIONNAIRE_COMMON)
EXTRA_FILES_2021 = ['INS', 'GLU', 'CBC', 'HSCRP', 'FERTIN', 'BPXO', 'DIQ', 'BPQ',
                    'WHQ', 'HUQ', 'SLQ', 'SMQ', 'ALQ', 'PAQ']
DERIVED = ['WHR', 'HOMA_IR', 'TyG', 'mean_SBP', 'mean_DBP', 'weight_change_1y']


def load_2021_extended():
    data = load_cycle(RAW_2021, 'L')
    for name in EXTRA_FILES_2021:
        part = pd.read_sas(RAW_2021 / f'{name}_L.xpt')
        keep = [c for c in part if c in COMMON_EXTRA and c not in data]
        data = data.merge(part[['SEQN'] + keep], on='SEQN', how='left', validate='one_to_one')
    return clean_missing_codes(data, QUESTIONNAIRE_COMMON + ['DMDEDUC2', 'DMDBORN4'])


def load_pooled():
    old = eligible(build_extended())
    new = load_2021_extended()
    old['cycle_2021'], new['cycle_2021'] = 0.0, 1.0
    keep = ['SEQN', 'cycle_2021', TARGET, 'LUXCPIQR'] + RAW_FEATURES + COMMON_EXTRA
    pooled = pd.concat([old[keep], new[keep]], ignore_index=True)
    # 2021-2023 SEQN values continue past the 2017-2020 range, so IDs stay unique.
    if pooled.SEQN.duplicated().any():
        raise ValueError('Duplicate SEQN across cycles.')
    return pooled


def features(data, extended=True):
    X = CAPFeatures().fit_transform(data[RAW_FEATURES])
    if not extended:
        return X
    X = pd.concat([X, data[COMMON_EXTRA].reset_index(drop=True)], axis=1)
    X['WHD140'] = np.nan  # add_derived expects it; not asked in 2021-2023
    X = add_derived(X).drop(columns=['WHD140', 'weight_from_max'])
    X['cycle_2021'] = data.cycle_2021.to_numpy()
    return X


def xgb(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=8,
        colsample_bytree=0.6):
    return XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators,
                        learning_rate=learning_rate, max_depth=max_depth,
                        min_child_weight=min_child_weight, subsample=0.8,
                        colsample_bytree=colsample_bytree, reg_lambda=3, reg_alpha=0.5,
                        random_state=SEED, n_jobs=-1)


CONFIGS = {
    'xgb_baseline_params': {},
    'xgb_slow_wide': dict(n_estimators=1500, learning_rate=0.01, colsample_bytree=0.4),
    'xgb_deeper': dict(n_estimators=2000, learning_rate=0.01, max_depth=5,
                       min_child_weight=20, colsample_bytree=0.4),
}


def training_rows(iqr, mode):
    """Sample weights per training mode; zero weight drops a row."""
    if mode == 'all_exams':
        return np.ones(len(iqr))
    if mode == 'reliable_only':
        return (iqr < RELIABLE_IQR).astype(float)
    if mode == 'quality_weighted':
        # Full weight below the threshold, shrinking as IQR grows; missing IQR keeps weight 1.
        return np.clip(RELIABLE_IQR / np.fmax(iqr, RELIABLE_IQR), 0.25, 1).fillna(1).to_numpy()
    raise ValueError(mode)


def fit(config, X, y, weight):
    keep = weight > 0
    model = xgb(**CONFIGS[config])
    model.fit(X[keep], y[keep], sample_weight=weight[keep])
    return model


def score(y, prediction, reliable):
    full, rel = metrics(y, prediction), metrics(y[reliable], prediction[reliable])
    return dict(r2_all=full['r2'], rmse_all=full['rmse'], mae_all=full['mae'],
                r2_reliable=rel['r2'], rmse_reliable=rel['rmse'])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_pooled()
    y = data[TARGET].astype(float).to_numpy()
    reliable = (data.LUXCPIQR < RELIABLE_IQR).to_numpy()
    train, test = train_test_split(np.arange(len(data)), test_size=0.2,
                                   random_state=SEED, stratify=data.cycle_2021)
    X_full = features(data)
    print(f'Pooled cohort: {len(data)} (2021-2023: {int(data.cycle_2021.sum())}); '
          f'train {len(train)}, test {len(test)}; reliable exams {reliable.mean():.1%}', flush=True)

    # Model selection: 5-fold CV inside the training rows only.
    X_tr, y_tr, rel_tr = X_full.iloc[train].reset_index(drop=True), y[train], reliable[train]
    iqr_tr = data.LUXCPIQR.iloc[train].reset_index(drop=True)
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(X_tr))
    cv_rows = []
    for mode in ['all_exams', 'reliable_only', 'quality_weighted']:
        weight = training_rows(iqr_tr, mode)
        for config in CONFIGS:
            oof = np.full(len(y_tr), np.nan)
            for tr, va in folds:
                oof[va] = fit(config, X_tr.iloc[tr], y_tr[tr], weight[tr]).predict(X_tr.iloc[va])
            cv_rows.append(dict(training=mode, config=config, **score(y_tr, oof, rel_tr)))
            print(f'CV {mode:17s} {config:20s} R2 all={cv_rows[-1]["r2_all"]:.4f} '
                  f'reliable={cv_rows[-1]["r2_reliable"]:.4f}', flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUTPUT_DIR / 'cv_selection.csv', index=False)

    # Two winners, each chosen by CV only: best on every exam, best on reliable exams.
    best_all = cv.loc[cv.r2_all.idxmax()]
    best_rel = cv.loc[cv.r2_reliable.idxmax()]

    # Test-set comparison, every model trained on the same training rows.
    test_rows, test_pred = [], {'SEQN': data.SEQN.iloc[test].to_numpy(), 'actual_CAP': y[test],
                                'reliable_exam': reliable[test]}
    old_train = train[data.cycle_2021.iloc[train].eq(0).to_numpy()]
    X_base = features(data, extended=False)
    candidates = [
        ('baseline_36_inputs_2017_2020_only', X_base, old_train, None, 'xgb_baseline_params'),
        ('baseline_36_inputs_pooled', X_base, train, None, 'xgb_baseline_params'),
        ('selected_for_all_exams', X_full, train, best_all.training, best_all.config),
        ('selected_for_reliable_exams', X_full, train, best_rel.training, best_rel.config),
    ]
    fitted = {}
    for name, X, rows, mode, config in candidates:
        weight = np.ones(len(rows)) if mode is None else training_rows(
            data.LUXCPIQR.iloc[rows].reset_index(drop=True), mode)
        model = fit(config, X.iloc[rows].reset_index(drop=True), y[rows], weight)
        fitted[name] = (model, list(X.columns), mode, config)
        prediction = model.predict(X.iloc[test])
        test_pred[name] = prediction
        old_test = data.cycle_2021.iloc[test].eq(0).to_numpy()
        test_rows.append(dict(model=name, training=mode or 'all_exams', config=config,
                              n_features=X.shape[1], **score(y[test], prediction, reliable[test]),
                              r2_2017_2020_test=metrics(y[test][old_test], prediction[old_test])['r2']))
        r = test_rows[-1]
        print(f'TEST {name:36s} R2 all={r["r2_all"]:.4f} reliable={r["r2_reliable"]:.4f} '
              f'2017-2020={r["r2_2017_2020_test"]:.4f}', flush=True)

    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / 'test_metrics.csv', index=False)
    pd.DataFrame(test_pred).to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
    for name in ['selected_for_all_exams', 'selected_for_reliable_exams']:
        model, columns, mode, config = fitted[name]
        joblib.dump(dict(model=model, feature_columns=columns, training=mode, config=config,
                         raw_inputs=RAW_FEATURES + COMMON_EXTRA + ['cycle_2021'],
                         build_features='src.experiments.combined_model.features'),
                    OUTPUT_DIR / f'{name}.joblib')
        pd.Series(model.get_booster().get_score(importance_type='gain'), name='gain') \
            .sort_values(ascending=False).to_csv(OUTPUT_DIR / f'{name}_importance.csv')
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(dict(
            target=TARGET, participants=int(len(data)), train=int(len(train)), test=int(len(test)),
            participants_2021_2023=int(data.cycle_2021.sum()), reliable_iqr_below=RELIABLE_IQR,
            extra_inputs=COMMON_EXTRA, derived=DERIVED, configs=CONFIGS,
            selected_for_all_exams=best_all[['training', 'config']].to_dict(),
            selected_for_reliable_exams=best_rel[['training', 'config']].to_dict(),
            note=('Selection used training-set CV only; the test set was scored once. '
                  'Internal validation only. Reliable-exam R² describes a quality-filtered '
                  'subset and must be reported alongside full-test R².'),
        ), f, indent=2)
    print(f'\nResults saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
