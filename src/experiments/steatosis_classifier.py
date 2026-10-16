"""Calibrated steatosis classifier on the pooled 2017-2023 cohort.

The proposal (§5.1, §12.1) evaluates Layer 2 with ROC-AUC, F1, Brier score and
calibration. This trains XGBoost classifiers for CAP >= 248 dB/m (any
steatosis, S1+) and CAP >= 280 dB/m (S3-range; cutoffs follow the existing
classification experiment and are provisional). Probabilities are calibrated
with isotonic regression fitted on 5-fold out-of-fold training predictions.

Same 80/20 test split as combined_model.py (stratified by cycle, seed 42). The
decision threshold maximises F1 on training out-of-fold predictions only. Test
metrics are reported overall and by age group, for the original 46 inputs and
the extended inputs. Run from the project root:

    .venv/Scripts/python.exe -m src.experiments.steatosis_classifier
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.train_final_model import ROOT, TARGET
from src.experiments.combined_model import SEED, features, load_pooled

OUTPUT_DIR = ROOT / 'data' / 'results' / 'experiments' / 'steatosis_classifier'
CUTOFFS = {'S1plus_cap248': 248, 'cap280': 280}
AGE_GROUPS = {'12-17': (12, 18), '18-39': (18, 40), '40-59': (40, 60), '60+': (60, 200)}


def classifier():
    return XGBClassifier(objective='binary:logistic', n_estimators=1500, learning_rate=0.01,
                         max_depth=4, min_child_weight=8, subsample=0.8, colsample_bytree=0.4,
                         reg_lambda=3, reg_alpha=0.5, eval_metric='logloss',
                         random_state=SEED, n_jobs=-1)


def expected_calibration_error(label, probability, bins=10):
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, probability, side='right') - 1, 0, bins - 1)
    return float(sum(abs(label[index == b].mean() - probability[index == b].mean()) * (index == b).mean()
                     for b in range(bins) if (index == b).any()))


def best_f1_threshold(label, probability):
    precision, recall, threshold = precision_recall_curve(label, probability)
    f1 = 2 * precision * recall / np.fmax(precision + recall, 1e-12)
    return float(threshold[np.argmax(f1[:-1])])


def evaluate(label, probability, threshold):
    predicted = probability >= threshold
    return dict(roc_auc=roc_auc_score(label, probability),
                pr_auc=average_precision_score(label, probability),
                brier=brier_score_loss(label, probability),
                ece=expected_calibration_error(label, probability),
                f1=f1_score(label, predicted),
                balanced_accuracy=balanced_accuracy_score(label, predicted),
                prevalence=float(label.mean()))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_pooled()
    train, test = train_test_split(np.arange(len(data)), test_size=0.2,
                                   random_state=SEED, stratify=data.cycle_2021)
    age = data.RIDAGEYR.to_numpy()
    inputs = {'base_46_inputs': features(data, extended=False), 'extended_inputs': features(data)}
    rows, calibration_rows = [], []
    for target_name, cutoff in CUTOFFS.items():
        label = (data[TARGET].to_numpy() >= cutoff).astype(int)
        for input_name, X in inputs.items():
            X_tr, y_tr = X.iloc[train].reset_index(drop=True), label[train]
            oof = np.zeros(len(train))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tr, y_tr):
                oof[va] = classifier().fit(X_tr.iloc[tr], y_tr[tr]).predict_proba(X_tr.iloc[va])[:, 1]
            calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof, y_tr)
            threshold = best_f1_threshold(y_tr, calibrator.predict(oof))
            model = classifier().fit(X_tr, y_tr)
            raw = model.predict_proba(X.iloc[test])[:, 1]
            calibrated = calibrator.predict(raw)
            y_te, age_te = label[test], age[test]
            for kind, probability in [('uncalibrated', raw), ('isotonic', calibrated)]:
                cut = threshold if kind == 'isotonic' else best_f1_threshold(y_tr, oof)
                groups = {'all': np.ones(len(test), bool)} | {
                    g: (age_te >= lo) & (age_te < hi) for g, (lo, hi) in AGE_GROUPS.items()}
                for group, mask in groups.items():
                    rows.append(dict(target=target_name, inputs=input_name, probabilities=kind,
                                     group=group, n=int(mask.sum()), threshold=cut,
                                     **evaluate(y_te[mask], probability[mask], cut)))
            bins = pd.qcut(calibrated, 10, duplicates='drop')
            curve = pd.DataFrame({'bin': bins, 'p': calibrated, 'y': y_te}).groupby('bin', observed=True)
            calibration_rows += [dict(target=target_name, inputs=input_name, mean_predicted=g.p.mean(),
                                      observed_rate=g.y.mean(), n=len(g)) for _, g in curve]
            all_row = [r for r in rows if r['target'] == target_name and r['inputs'] == input_name
                       and r['probabilities'] == 'isotonic' and r['group'] == 'all'][0]
            print(f'{target_name:14s} {input_name:16s} AUC={all_row["roc_auc"]:.4f} '
                  f'Brier={all_row["brier"]:.4f} ECE={all_row["ece"]:.4f} F1={all_row["f1"]:.4f}',
                  flush=True)
            if input_name == 'extended_inputs':
                joblib.dump(dict(model=model, calibrator=calibrator, threshold=threshold,
                                 cutoff=cutoff, feature_columns=list(X.columns),
                                 build_features='src.experiments.combined_model.features'),
                            OUTPUT_DIR / f'{target_name}_classifier.joblib')

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / 'test_metrics.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(OUTPUT_DIR / 'calibration_curve.csv', index=False)
    print('\nIsotonic, extended inputs, by age group:')
    view = results.query("probabilities == 'isotonic' and inputs == 'extended_inputs'")
    print(view[['target', 'group', 'n', 'roc_auc', 'brier', 'f1', 'balanced_accuracy']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(dict(cutoffs=CUTOFFS, cutoff_status='Provisional, matching the classification experiment',
                       train=int(len(train)), test=int(len(test)), calibration='Isotonic on 5-fold OOF',
                       threshold='Max F1 on training OOF', note='Internal validation only.'), f, indent=2)
    print(f'\nResults saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
