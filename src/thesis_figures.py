"""Thesis figures and clinical-score benchmark for the combined model.

Uses the saved artifacts and the test split from train_combined_model.py:

1. ROC curves for the three steatosis classifiers (CAP >= 248, 280, 302).
2. Benchmark against the Fatty Liver Index (Bedogni et al., BMC
   Gastroenterol 2006) and Hepatic Steatosis Index (Lee et al., Dig Liver
   Dis 2010) on the same test participants, with paired bootstrap intervals
   for the AUC difference.
3. Calibration (reliability) plots with 95% intervals per bin.
4. SHAP summary for the CAP >= 248 classifier.
5. Learning curve: test R² and AUC against training-set size.

FLI uses non-fasting serum triglycerides (LBXSTR), as fasting values exist for
only part of the cohort. Run after train_combined_model.py and
validate_combined_model.py:

    .venv/Scripts/python.exe -m src.thesis_figures
"""
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.metrics import r2_score, roc_auc_score, roc_curve

from src.train_final_model import ROOT, TARGET
from src.combined_features import COMBINED_FEATURES
from src.build_combined_dataset import OUTPUT as DATA_FILE
from src.train_combined_model import (SEED, classifier_pipeline, exam_weights, fit_weighted,
                                      regressor_pipeline)

MODEL_DIR = ROOT / 'data' / 'results' / 'combined'
OUTPUT_DIR = MODEL_DIR / 'figures'
CUTOFFS = {'cap248': 248, 'cap280': 280, 'cap302': 302}
BOOTSTRAPS = 1000
LEARNING_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
LABELS = {
    'BMXWAIST': 'Waist circumference', 'BMXBMI': 'BMI', 'BMXWT': 'Body weight', 'BMXHT': 'Height',
    'BMXHIP': 'Hip circumference', 'BMXARMC': 'Arm circumference', 'BMXARML': 'Upper arm length',
    'BMXLEG': 'Upper leg length', 'WHR': 'Waist-to-hip ratio', 'WHtR': 'Waist-to-height ratio',
    'LBXGH': 'HbA1c', 'LBXSGL': 'Glucose (serum)', 'LBXGLU': 'Fasting glucose', 'LBXIN': 'Fasting insulin',
    'HOMA_IR': 'HOMA-IR', 'TyG': 'Triglyceride-glucose index', 'LBXSTR': 'Triglycerides',
    'LBDHDD': 'HDL cholesterol', 'LBXTC': 'Total cholesterol', 'LBXSATSI': 'ALT', 'LBXSASSI': 'AST',
    'LBXSGTSI': 'GGT', 'LBXSUA': 'Uric acid', 'LBXSCR': 'Creatinine', 'RIDAGEYR': 'Age',
    'RIAGENDR': 'Sex', 'RIDRETH3': 'Race/ethnicity', 'LBXHSCRP': 'hs-CRP', 'LBXFER': 'Ferritin',
    'TG_HDL': 'Triglyceride/HDL ratio', 'AST_ALT': 'AST/ALT ratio', 'FLI_proxy': 'FLI (derived)', 'HSI_partial': 'HSI (derived, partial)',
    'LBXWBCSI': 'White cell count', 'LBXPLTSI': 'Platelets', 'LBXSAL': 'Albumin', 'LBXSPH': 'Phosphorus',
    'mean_SBP': 'Systolic blood pressure', 'mean_DBP': 'Diastolic blood pressure',
    'DIQ010': 'Diagnosed diabetes', 'LBXSKSI': 'Potassium', 'LBXSCK': 'Creatine kinase',
    'LBXSBU': 'Blood urea nitrogen', 'LBXSIR': 'Iron', 'LBXSTP': 'Total protein', 'LBXSGB': 'Globulin',
}


def label(name):
    base = name.split('__')[-1]
    for code, text in LABELS.items():
        if base == code:
            return text
        if base.startswith(code + '_'):
            return f'{text} = {base[len(code) + 1:]}'
    return base


def fatty_liver_index(data):
    logit = (0.953 * np.log(data.LBXSTR) + 0.139 * data.BMXBMI + 0.718 * np.log(data.LBXSGTSI)
             + 0.053 * data.BMXWAIST - 15.745)
    return 100 / (1 + np.exp(-logit))


def hepatic_steatosis_index(data):
    diabetes = data.DIQ010.eq(1) | data.LBXGH.ge(6.5)
    return (8 * data.LBXSATSI / data.LBXSASSI + data.BMXBMI
            + 2 * data.RIAGENDR.eq(2) + 2 * diabetes)


def paired_auc_difference(label_, a, b, rng):
    n = len(label_)
    draws = []
    for _ in range(BOOTSTRAPS):
        i = rng.integers(0, n, n)
        if label_[i].min() == label_[i].max():
            continue
        draws.append(roc_auc_score(label_[i], a[i]) - roc_auc_score(label_[i], b[i]))
    return np.percentile(draws, [2.5, 97.5])


def auc_interval(label_, score, rng):
    n = len(label_)
    draws = [roc_auc_score(label_[i], score[i]) for i in (rng.integers(0, n, n) for _ in range(BOOTSTRAPS))]
    return np.percentile(draws, [2.5, 97.5])


def style(ax, title):
    ax.set_title(title, fontsize=11, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 10, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    rng = np.random.default_rng(SEED)
    data = pd.read_csv(DATA_FILE)
    manifest = pd.read_csv(MODEL_DIR / 'split_manifest.csv')
    test_mask = manifest.partition.eq('test').to_numpy()
    train, test = data[~test_mask].reset_index(drop=True), data[test_mask].reset_index(drop=True)
    X_test, y_test = test[COMBINED_FEATURES], test[TARGET].to_numpy(dtype=float)

    classifiers = {name: joblib.load(MODEL_DIR / f'steatosis_{name}.joblib') for name in ('cap248', 'cap280')}
    classifiers['cap302'] = joblib.load(MODEL_DIR / 'validation' / 'steatosis_cap302.joblib')
    probability = {name: c.predict_proba(X_test) for name, c in classifiers.items()}
    labels = {name: (y_test >= cutoff).astype(int) for name, cutoff in CUTOFFS.items()}

    # 1-2. ROC curves and the clinical-score benchmark.
    complete = test[['LBXSTR', 'LBXSGTSI', 'BMXWAIST', 'BMXBMI', 'LBXSATSI', 'LBXSASSI']].notna().all(axis=1)
    complete &= test.LBXSTR.gt(0) & test.LBXSGTSI.gt(0) & test.LBXSASSI.gt(0)
    complete = complete.to_numpy()
    scores = {'HepatoTwin model': None, 'Fatty Liver Index': fatty_liver_index(test).to_numpy(),
              'Hepatic Steatosis Index': hepatic_steatosis_index(test).to_numpy()}
    rows = []
    for name, cutoff in CUTOFFS.items():
        y, model = labels[name][complete], probability[name][complete]
        low, high = auc_interval(y, model, rng)
        rows.append(dict(target=f'CAP >= {cutoff}', score='HepatoTwin model', n=int(complete.sum()),
                         auc=roc_auc_score(y, model), ci_low=low, ci_high=high))
        for score_name in ('Fatty Liver Index', 'Hepatic Steatosis Index'):
            other = scores[score_name][complete]
            low, high = auc_interval(y, other, rng)
            d_low, d_high = paired_auc_difference(y, model, other, rng)
            rows.append(dict(target=f'CAP >= {cutoff}', score=score_name, n=int(complete.sum()),
                             auc=roc_auc_score(y, other), ci_low=low, ci_high=high,
                             model_minus_score=roc_auc_score(y, model) - roc_auc_score(y, other),
                             difference_ci_low=d_low, difference_ci_high=d_high))
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(OUTPUT_DIR / 'clinical_score_benchmark.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for name, cutoff in CUTOFFS.items():
        fpr, tpr, _ = roc_curve(labels[name], probability[name])
        axes[0].plot(fpr, tpr, lw=1.6, label=f'CAP ≥ {cutoff} (AUC {roc_auc_score(labels[name], probability[name]):.3f})')
    y = labels['cap248'][complete]
    for score_name, colour in [('HepatoTwin model', 'C0'), ('Fatty Liver Index', 'C3'),
                               ('Hepatic Steatosis Index', 'C2')]:
        s = probability['cap248'][complete] if score_name == 'HepatoTwin model' else scores[score_name][complete]
        fpr, tpr, _ = roc_curve(y, s)
        axes[1].plot(fpr, tpr, lw=1.6, color=colour, label=f'{score_name} (AUC {roc_auc_score(y, s):.3f})')
    for ax, title in zip(axes, ['(a) Steatosis classifiers, test set (n = 3,060)',
                                f'(b) CAP ≥ 248 against clinical scores (n = {complete.sum():,})']):
        ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=0.8)
        ax.set_xlabel('False positive rate (1 − specificity)')
        ax.set_ylabel('True positive rate (sensitivity)')
        ax.legend(loc='lower right', fontsize=8.5, frameon=False)
        style(ax, title)
    fig.savefig(OUTPUT_DIR / 'fig_roc_curves.png')
    plt.close(fig)

    # 3. Calibration plots.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
    for col, name in enumerate(['cap248', 'cap280']):
        p, y = probability[name], labels[name]
        bins = pd.qcut(p, 10, duplicates='drop')
        grouped = pd.DataFrame({'bin': bins, 'p': p, 'y': y}).groupby('bin', observed=True)
        mean_p, rate, n = grouped.p.mean(), grouped.y.mean(), grouped.y.size()
        err = 1.96 * np.sqrt(rate * (1 - rate) / n)
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=0.8, label='Perfect calibration')
        ax.errorbar(mean_p, rate, yerr=err, fmt='o-', ms=4, lw=1.4, capsize=2.5, label='Isotonic-calibrated model')
        ax.set_ylabel('Observed proportion')
        ax.legend(loc='upper left', fontsize=8.5, frameon=False)
        style(ax, f'({"ab"[col]}) P(CAP ≥ {CUTOFFS[name]} dB/m)')
        axes[1, col].hist(p, bins=30, color='C0', alpha=0.7)
        axes[1, col].set_xlabel('Predicted probability')
        axes[1, col].set_ylabel('Participants')
        axes[1, col].spines[['top', 'right']].set_visible(False)
    fig.savefig(OUTPUT_DIR / 'fig_calibration.png')
    plt.close(fig)

    # 4. SHAP for the CAP >= 248 classifier (log-odds of the uncalibrated model;
    # isotonic calibration is monotone, so the feature ranking is unchanged).
    pipeline = classifiers['cap248'].pipeline
    transformed = pipeline[:-1].transform(X_test)
    transformed = pd.DataFrame(transformed, columns=pipeline[:-1].get_feature_names_out()) \
        if not isinstance(transformed, pd.DataFrame) else transformed
    # XGBoost's built-in TreeSHAP; shap 0.49's TreeExplainer cannot read XGBoost 3 models.
    contributions = pipeline[-1].get_booster().predict(
        xgboost.DMatrix(transformed, missing=np.nan), pred_contribs=True)
    explanation = shap.Explanation(values=contributions[:, :-1], base_values=contributions[:, -1],
                                   data=transformed.to_numpy(dtype=float),
                                   feature_names=[label(c) for c in transformed.columns])
    importance = pd.Series(np.abs(explanation.values).mean(axis=0), index=transformed.columns) \
        .sort_values(ascending=False)
    importance.rename('mean_abs_shap_log_odds').to_csv(OUTPUT_DIR / 'shap_importance_cap248.csv')
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False, plot_size=(8, 5.6))
    plt.gca().set_xlabel('SHAP value (effect on log-odds of CAP ≥ 248 dB/m)')
    plt.savefig(OUTPUT_DIR / 'fig_shap_cap248.png')
    plt.close('all')

    # 5. Learning curve with the selected settings.
    metadata = json.loads((MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))
    selected = metadata['selected']
    X_train, y_train = train[COMBINED_FEATURES], train[TARGET].to_numpy(dtype=float)
    iqr = train.LUXCPIQR.to_numpy(dtype=float)
    order = rng.permutation(len(train))
    curve = []
    for fraction in LEARNING_FRACTIONS:
        rows_ = order[:int(round(fraction * len(train)))]
        regressor = fit_weighted(regressor_pipeline(selected['config']), X_train.iloc[rows_], y_train[rows_],
                                 exam_weights(iqr[rows_], selected['exam_weighting']))
        classifier = classifier_pipeline().fit(X_train.iloc[rows_], (y_train[rows_] >= 248).astype(int))
        curve.append(dict(fraction=fraction, n_train=len(rows_),
                          test_r2=r2_score(y_test, regressor.predict(X_test)),
                          test_auc_cap248=roc_auc_score(labels['cap248'], classifier.predict_proba(X_test)[:, 1])))
        print(f'learning curve n={len(rows_)}: R2={curve[-1]["test_r2"]:.3f} AUC={curve[-1]["test_auc_cap248"]:.3f}',
              flush=True)
    curve = pd.DataFrame(curve)
    curve.to_csv(OUTPUT_DIR / 'learning_curve.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, column, title, ylabel in [(axes[0], 'test_r2', '(a) CAP regression', 'Test R²'),
                                      (axes[1], 'test_auc_cap248', '(b) CAP ≥ 248 classification', 'Test ROC-AUC')]:
        ax.plot(curve.n_train, curve[column], 'o-', lw=1.6)
        for x, v in zip(curve.n_train, curve[column]):
            ax.annotate(f'{v:.3f}', (x, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.5)
        ax.set_xlabel('Training participants')
        ax.set_ylabel(ylabel)
        low, high = curve[column].min(), curve[column].max()
        ax.set_ylim(low - 0.06, high + 0.04)
        style(ax, title)
    fig.savefig(OUTPUT_DIR / 'fig_learning_curve.png')
    plt.close(fig)

    print('\nClinical-score benchmark:')
    print(benchmark.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nTop SHAP features:')
    print(importance.head(10).rename(index=label).to_string(float_format=lambda v: f'{v:.3f}'))
    print(f'\nSaved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
