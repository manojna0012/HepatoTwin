"""External and long-term validation of the Layer 2 steatosis model on NHANES III (Proposal section 17.9).

Data: src/nhanes3.py maps NHANES III (1988-1994) adults aged 20-59 with a gradable ultrasound onto the
model's inputs (51 of 81; the rest are missing, which XGBoost handles) and links the public-use mortality
file (follow-up to 31 December 2019).

1. Discrimination. The model's chances of CAP >= 280 and CAP >= 248 are compared with the ultrasound grade
   (moderate or severe steatosis against normal or mild, the NCHS-recommended recode). AUC with a bootstrap
   95% interval, beside two established scores computed from the same record: the Fatty Liver Index (FLI;
   Bedogni 2006; needs GGT, so on the subset with GGT) and the Hepatic Steatosis Index (HSI; Lee 2010), and
   BMI and waist alone. A different reference standard (ultrasound, not CAP) 30 years earlier, so this tests
   whether the model ranks people correctly, not whether it reproduces CAP values or prevalence.
2. Subgroups: AUC by sex and by race and ethnicity.
3. Mortality. Participants are split into quarters of the chance of CAP >= 280; deaths per 1,000
   person-years and rate ratios adjusted for age and sex (Poisson regression with log person-years as
   offset, fitted by iteratively reweighted least squares; Wald 95% intervals) for all-cause, cardiovascular (heart disease
   or stroke, NCHS leading causes 1 and 5) and cancer (2) deaths. The ultrasound grade itself is analysed
   the same way as a reference. The public file has no liver-death category, and smoking, alcohol and
   income are not mapped, so the ratios are adjusted for age and sex only.

Unweighted: the results describe this sample, not the US population.

    .venv/Scripts/python.exe -m src.nhanes3_validation
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.predict_cap import MODEL_FILE, predict_patients, predict_steatosis
from src.train_final_model import ROOT

DATA = ROOT / 'data' / 'processed' / 'nhanes3_mapped.csv'
OUTPUT = ROOT / 'data' / 'results' / 'nhanes3'
MODELS = MODEL_FILE.parent
BOOT = 1000
SEED = 20261108
CAUSES = {'all-cause': None, 'cardiovascular': (1, 5), 'cancer': (2,)}
RACE_NAMES = {1: 'Mexican American', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black', 7: 'Other'}


def fatty_liver_index(d):
    """Bedogni et al. 2006: triglycerides mg/dL, BMI, GGT U/L, waist cm."""
    logit = 0.953 * np.log(d.LBXSTR) + 0.139 * d.BMXBMI + 0.718 * np.log(d.LBXSGTSI) + 0.053 * d.BMXWAIST - 15.745
    return 100 / (1 + np.exp(-logit))


def hepatic_steatosis_index(d):
    """Lee et al. 2010: 8 x ALT/AST + BMI + 2 if female + 2 if diabetes (here HbA1c >= 6.5% or fasting
    glucose >= 126 mg/dL, as NHANES III's diagnosis question is not mapped)."""
    diabetes = (d.LBXGH >= 6.5) | (d.LBXGLU >= 126)
    return 8 * d.LBXSATSI / d.LBXSASSI + d.BMXBMI + 2 * (d.RIAGENDR == 2) + 2 * diabetes


def auc_ci(y, score, rng, boot=BOOT):
    keep = score.notna() & y.notna()
    y, score = y[keep].to_numpy(), score[keep].to_numpy()
    point = roc_auc_score(y, score)
    draws = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            draws.append(roc_auc_score(y[i], score[i]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {'n': int(len(y)), 'auc': round(float(point), 3), 'ci95': [round(float(low), 3), round(float(high), 3)]}


def model_chances(data):
    model = __import__('joblib').load(MODEL_FILE)
    X = data.reindex(columns=list(model.feature_names_in_))
    out = pd.DataFrame(index=data.index)
    out['chance_cap280'] = predict_steatosis(X, MODELS / 'steatosis_cap280.joblib').to_numpy()
    out['chance_cap248'] = predict_steatosis(X, MODELS / 'steatosis_cap248.joblib').to_numpy()
    out['predicted_cap'] = predict_patients(X).to_numpy()
    return out


def discrimination(d, rng):
    y = d.steatosis_moderate_severe
    scores = {'model: chance of CAP >= 280': d.chance_cap280, 'model: chance of CAP >= 248': d.chance_cap248,
              'model: estimated CAP': d.predicted_cap, 'Hepatic Steatosis Index': d.hsi, 'BMI': d.BMXBMI,
              'waist': d.BMXWAIST}
    out = {name: auc_ci(y, s, rng) for name, s in scores.items()}
    has_ggt = d.fli.notna()
    out['with GGT (FLI subset)'] = {'Fatty Liver Index': auc_ci(y[has_ggt], d.fli[has_ggt], rng),
                                    'model: chance of CAP >= 280': auc_ci(y[has_ggt], d.chance_cap280[has_ggt], rng)}
    return out


def subgroups(d, rng):
    out = {}
    for label, column, names in (('sex', 'RIAGENDR', {1: 'men', 2: 'women'}), ('race and ethnicity', 'RIDRETH3', RACE_NAMES)):
        out[label] = {}
        for code, name in names.items():
            g = d[d[column] == code]
            if g.steatosis_moderate_severe.nunique() == 2 and len(g) >= 100:
                out[label][name] = {**auc_ci(g.steatosis_moderate_severe, g.chance_cap280, rng, boot=300),
                                    'moderate_severe_share': round(float(g.steatosis_moderate_severe.mean()), 3)}
    return out


def calibration(d):
    """Mean chance against the ultrasound share by tenth of the chance (different reference standard)."""
    tenth = pd.qcut(d.chance_cap280, 10, labels=False)
    return [{'tenth': int(t) + 1, 'mean_chance_cap280': round(float(g.chance_cap280.mean()), 3),
             'moderate_severe_share': round(float(g.steatosis_moderate_severe.mean()), 3), 'n': int(len(g))}
            for t, g in d.groupby(tenth)]


def died_of(d, cause):
    codes = CAUSES[cause]
    dead = d.MORTSTAT == 1
    return dead if codes is None else dead & d.UCOD_LEADING.isin(codes)


def poisson_fit(X, events, years, iterations=50):
    """Poisson regression with log person-years as offset, by iteratively reweighted least squares.
    Returns coefficients and their standard errors (inverse Fisher information)."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    y, offset = np.asarray(events, dtype=float), np.log(np.asarray(years, dtype=float))
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / np.exp(offset).sum())
    for _ in range(iterations):
        mu = np.exp(X @ beta + offset)
        info = X.T @ (X * mu[:, None])
        step = np.linalg.solve(info, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * np.exp(X @ beta + offset)[:, None]))))
    return beta[1:], se[1:]


def rate_ratios(d, groups, events):
    """Age- and sex-adjusted rate ratios of each group against the first, with Wald 95% intervals."""
    levels = sorted(groups.unique())
    X = pd.DataFrame({'age_per_10y': (d.RIDAGEYR - 40) / 10, 'female': (d.RIAGENDR == 2).astype(float)})
    for level in levels[1:]:
        X[f'g{level}'] = (groups == level).astype(float)
    beta, se = poisson_fit(X, events, d.PERMTH_EXM / 12)
    return {str(level): {'rate_ratio': round(float(np.exp(b)), 2),
                         'ci95': [round(float(np.exp(b - 1.96 * e)), 2), round(float(np.exp(b + 1.96 * e)), 2)]}
            for level, b, e in zip(levels[1:], beta[2:], se[2:])}


def mortality(d):
    d = d[(d.ELIGSTAT == 1) & d.MORTSTAT.isin([0, 1]) & (d.PERMTH_EXM > 0)].reset_index(drop=True)
    quarter = pd.qcut(d.chance_cap280, 4, labels=False) + 1
    out = {'participants': int(len(d)), 'median_follow_up_years': round(float(d.PERMTH_EXM.median() / 12), 1),
           'deaths': int((d.MORTSTAT == 1).sum()),
           'quarter_bounds_chance_cap280': [round(float(x), 3) for x in d.chance_cap280.quantile([0, .25, .5, .75, 1])]}
    for cause in CAUSES:
        events = died_of(d, cause).astype(float)
        table = []
        for q in range(1, 5):
            g = quarter == q
            years = float(d.PERMTH_EXM[g].sum() / 12)
            table.append({'quarter': q, 'deaths': int(events[g].sum()), 'person_years': round(years),
                          'per_1000_person_years': round(1000 * float(events[g].sum()) / years, 2)})
        out[cause] = {'by_quarter': table,
                      'adjusted_rate_ratio_vs_lowest_quarter': rate_ratios(d, quarter, events),
                      'ultrasound_moderate_severe_vs_not': rate_ratios(d, d.steatosis_moderate_severe, events)['1']}
    return out


def main():
    rng = np.random.default_rng(SEED)
    data = pd.read_csv(DATA)
    d = pd.concat([data, model_chances(data)], axis=1)
    d['fli'], d['hsi'] = fatty_liver_index(d), hepatic_steatosis_index(d)
    summary = {'participants': int(len(d)), 'moderate_severe_share': round(float(d.steatosis_moderate_severe.mean()), 3),
               'mean_chance_cap280': round(float(d.chance_cap280.mean()), 3),
               'discrimination': discrimination(d, rng), 'subgroups': subgroups(d, rng), 'calibration_by_tenth': calibration(d),
               'mortality': mortality(d)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    d[['SEQN', 'steatosis_moderate_severe', 'chance_cap280', 'chance_cap248', 'predicted_cap', 'fli', 'hsi']].to_csv(
        OUTPUT / 'predictions.csv', index=False)
    (OUTPUT / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
