"""Fibrosis risk (FIB-4) and the referral pathway (Proposal section 17.2).

FIB-4 = age x AST / (platelets x sqrt(ALT)) (Sterling et al., Hepatology 2006), with age in years,
AST and ALT in U/L and platelets in 10^9/L (NHANES LBXPLTSI, 1000 cells/uL). Categories follow the
EASL-EASD-EASO 2024 pathway: below 1.3 low risk, 1.3-2.67 indeterminate (further testing), above
2.67 high risk. FIB-4 was developed in adults, so patients aged 12-17 get no score. The guideline's
higher lower cut-off of 2.0 above age 65 is not needed: ages 60+ are outside the model's scope.
FIB-4 has low accuracy under age 35 (AASLD 2023 Practice Guidance, Figure 2), so a low score at 18-34
carries a note that it does not fully rule out fibrosis and that elastography would help.

A measured FibroScan liver stiffness (optional input LUXSMED, kPa, with LUXSIQRM, IQR/median %) replaces
FIB-4 for adults when the reading is reliable (IQR/median <= 30%, or not given): below 8 kPa low risk,
8 to below 12 kPa indeterminate, 12 kPa or more high risk (EASL-EASD-EASO 2024). An unreliable reading
falls back to FIB-4 with a note. The adult cut-offs are not applied under 18.

Validation: FIB-4 against liver stiffness measured in the same FibroScan exam, with >= 8 kPa as the
transient-elastography threshold the guideline uses to rule out advanced fibrosis and >= 12 kPa to
rule it in. FIB-4 has no fitted parameters, so every in-scope adult with a reliable stiffness reading
is used, and the held-out test partition is reported separately.

    .venv/Scripts/python.exe -m src.fibrosis
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.clinical_data import RELIABLE_STIFFNESS_IQR_RATIO, load_extras, reliable_stiffness
from src.train_final_model import ROOT

LOW, HIGH = 1.3, 2.67
ADULT_AGE = 18
FIB4_RELIABLE_AGE = 35
STIFFNESS_CUTOFFS = (8.0, 12.0)
OUTPUT = ROOT / 'data' / 'results' / 'fibrosis'
DATA_FILE = ROOT / 'data' / 'processed' / 'nhanes_combined.csv'
MANIFEST_FILE = ROOT / 'data' / 'results' / 'combined' / 'split_manifest.csv'

REFERRAL = {
    'high': ('High fibrosis risk (FIB-4 above 2.67). Please seek a clinical assessment of your liver '
             'before making diet changes; recipe suggestions are shown only after that advice.'),
    'indeterminate': ('Indeterminate fibrosis risk (FIB-4 1.3 to 2.67). The guideline recommends a second '
                      'test, such as liver elastography, to rule out advanced fibrosis.'),
    'low': 'Low fibrosis risk (FIB-4 below 1.3). Routine follow-up is enough.',
    'not_applicable': ('No fibrosis score is given for people under 18, because FIB-4 was developed and '
                       'validated in adults.'),
    'unknown': 'Fibrosis risk could not be calculated because age, AST, ALT or platelets are missing.',
}
STIFFNESS_REFERRAL = {
    'high': ('High fibrosis risk (FibroScan liver stiffness 12 kPa or more). Please seek a clinical assessment '
             'of your liver before making diet changes; recipe suggestions are shown only after that advice.'),
    'indeterminate': ('Indeterminate fibrosis risk (FibroScan liver stiffness 8 to 12 kPa). Please discuss further '
                      'assessment with your clinician.'),
    'low': 'Low fibrosis risk, confirmed by FibroScan (liver stiffness below 8 kPa). Routine follow-up is enough.',
}
YOUNG_NOTE = ('FIB-4 is less reliable under age 35, so a low score does not fully rule out liver scarring. '
              'If you have other risk factors, ask your clinician about a FibroScan.')
UNRELIABLE_NOTE = ('The FibroScan reading was not reliable (spread above 30% of the median), so FIB-4 was used '
                   'instead.')


def fib4(age, ast, alt, platelets):
    """FIB-4 index; NaN when an input is missing or not positive."""
    age, ast, alt, platelets = (np.asarray(x, dtype=float) for x in (age, ast, alt, platelets))
    valid = (ast > 0) & (alt > 0) & (platelets > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        value = age * ast / (platelets * np.sqrt(alt))
    return np.where(valid, value, np.nan)


def category(score, age):
    if age is None or age != age:
        return 'unknown'
    if age < ADULT_AGE:
        return 'not_applicable'
    if score is None or score != score:
        return 'unknown'
    return 'low' if score < LOW else ('high' if score > HIGH else 'indeterminate')


def _value(patient, name):
    try:
        value = float(patient.get(name))
    except (TypeError, ValueError):
        return None
    return None if value != value else value


def stiffness_category(kpa):
    return 'low' if kpa < STIFFNESS_CUTOFFS[0] else ('high' if kpa >= STIFFNESS_CUTOFFS[1] else 'indeterminate')


def assess(patient):
    """Fibrosis risk, its category and the referral message for one patient (raw NHANES-coded values).

    Uses a reliable measured FibroScan stiffness when the patient has one, otherwise FIB-4.
    """
    age = _value(patient, 'RIDAGEYR')
    stiffness, spread = _value(patient, 'LUXSMED'), _value(patient, 'LUXSIQRM')
    adult = age is not None and age >= ADULT_AGE
    reliable = spread is None or spread <= RELIABLE_STIFFNESS_IQR_RATIO
    if adult and stiffness is not None and reliable:
        cat = stiffness_category(stiffness)
        return {'method': 'fibroscan', 'liver_stiffness_kpa': round(stiffness, 1), 'fib4': None, 'category': cat,
                'refer_before_recipes': cat == 'high', 'message': STIFFNESS_REFERRAL[cat], 'note': None,
                'source': 'EASL-EASD-EASO 2024 liver stiffness cut-offs 8 and 12 kPa'}
    score = float(fib4(age, patient.get('LBXSASSI'), patient.get('LBXSATSI'), patient.get('LBXPLTSI')))
    cat = category(score, age)
    notes = []
    if adult and stiffness is not None:
        notes.append(UNRELIABLE_NOTE)
    if cat == 'low' and age < FIB4_RELIABLE_AGE:
        notes.append(YOUNG_NOTE)
    return {'method': 'fib4', 'liver_stiffness_kpa': None,
            'fib4': None if cat in ('not_applicable', 'unknown') else round(score, 2), 'category': cat,
            'refer_before_recipes': cat == 'high', 'message': REFERRAL[cat], 'note': ' '.join(notes) or None,
            'source': 'Sterling et al. 2006; EASL-EASD-EASO 2024 cut-offs 1.3 and 2.67; AASLD 2023 (age < 35)'}


def diagnostic_rows(score, stiffness, group):
    rows = []
    for cutoff in STIFFNESS_CUTOFFS:
        y = stiffness >= cutoff
        base = dict(group=group, n=len(y), stiffness_cutoff_kpa=cutoff, prevalence=float(y.mean()),
                    auc=float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else np.nan)
        for name, positive in (('fib4_ge_1.3', score >= LOW), ('fib4_gt_2.67', score > HIGH)):
            tp, fp = int((positive & y).sum()), int((positive & ~y).sum())
            fn, tn = int((~positive & y).sum()), int((~positive & ~y).sum())
            rows.append(base | dict(rule=name, sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                                    ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1)))
    return rows


def validation_cohort():
    data = pd.read_csv(DATA_FILE).merge(load_extras(), on='SEQN', how='left', validate='one_to_one')
    partition = pd.read_csv(MANIFEST_FILE).set_index('SEQN').partition
    data['partition'] = data.SEQN.map(partition)
    data['fib4'] = fib4(data.RIDAGEYR, data.LBXSASSI, data.LBXSATSI, data.LBXPLTSI)
    keep = (data.RIDAGEYR >= ADULT_AGE) & (data.RIDAGEYR < 60) & reliable_stiffness(data) & data.fib4.notna()
    return data[keep].reset_index(drop=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = validation_cohort()
    rows = []
    for group, part in (('ages 18-59, reliable stiffness', data), ('test partition only', data[data.partition.eq('test')])):
        rows += diagnostic_rows(part.fib4.to_numpy(), part.LUXSMED.to_numpy(), group)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / 'fib4_validation.csv', index=False)
    shares = data.fib4.apply(lambda s: category(s, 30)).value_counts(normalize=True).round(4).to_dict()
    (OUTPUT / 'fib4_summary.json').write_text(json.dumps({'n': len(data), 'category_shares': shares}, indent=2))
    print(f'{len(data)} adults aged 18-59 with a reliable stiffness reading; categories {shares}')
    print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
