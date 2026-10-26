"""Steatotic liver disease subtype and safety guardrails (Proposal section 17.3).

Subtype (Rinella et al., 2023 multisociety nomenclature): steatotic liver disease (SLD) with at least
one cardiometabolic risk factor (CMRF) is MASLD. MASLD with an average alcohol intake of 140-350 g a
week (women) or 210-420 g (men) is MetALD; above that range the label is alcohol-related liver disease
(ALD). SLD with no CMRF is cryptogenic, and the nomenclature advises further testing ("possible MASLD").

Adult CMRF (any one): BMI >= 25 kg/m2 (>= 23 for Asian patients) or waist > 94 cm (men) / > 80 cm
(women); fasting glucose >= 100 mg/dL, HbA1c >= 5.7%, diagnosed diabetes or glucose-lowering
treatment; blood pressure >= 130/85 mmHg or diagnosed hypertension; triglycerides >= 150 mg/dL;
HDL <= 40 mg/dL (men) / <= 50 mg/dL (women). Adolescent CMRF (ages 12-17), checked against the
nomenclature's cardiometabolic-criteria figure: BMI >= 85th percentile for age and sex (CDC 2000
charts); the adult glucose criteria plus a random serum glucose >= 200 mg/dL; blood pressure
>= 130/85 mmHg from age 13 (>= 130/80 at age 12); triglycerides >= 150 mg/dL (the figure's 100 mg/dL
applies under age 10); HDL <= 40 mg/dL. Two adolescent alternatives are not implemented, which can only
miss criteria, never add them: waist > 95th percentile (no reference percentiles in the project) and,
at age 12, blood pressure >= 95th percentile when that is below 130/80 (needs the height-specific
AAP 2017 tables).

NHANES limits: blood-pressure and cholesterol treatment are not in the dataset, so "diagnosed" stands
in for "treated"; triglycerides are the non-fasting biochemistry value (LBXSTR); other causes of
steatosis (viral hepatitis, medicines) are not assessed. Labels support the twin's explanations and
are not diagnoses.

Alcohol: NHANES ALQ121 (how often in the past 12 months) times ALQ130 (drinks on a drinking day),
at 14 g of alcohol per US standard drink (NIAAA).

Guardrails: patients under 18, with alcohol intake in the MetALD range or above, taking
glucose-lowering medicine, or pregnant receive a "discuss with a clinician" notice; a high FIB-4
(src/fibrosis.py) asks for clinical assessment before any recipes.
"""
import numpy as np
import pandas as pd

from src.simulation.growth import GROWTH_DIR, lms_at, z_score

GRAMS_PER_DRINK = 14
ADULT_AGE = 18
ASIAN = 6
MALE = 1
# ALQ121 codes -> drinking days per week (midpoint of each answer).
DAYS_PER_WEEK = {0: 0.0, 1: 7.0, 2: 5.5, 3: 3.5, 4: 2.0, 5: 1.0, 6: 2.5 / 4.345, 7: 1 / 4.345,
                 8: 9 / 52.18, 9: 4.5 / 52.18, 10: 1.5 / 52.18}
METALD_G_PER_WEEK = {'female': (140, 350), 'male': (210, 420)}
P85_Z = 1.0364   # z-score of the 85th percentile

NOTICES = {
    'under_18': ('You are under 18. Adult dietary guidance may not suit you; please discuss any diet change '
                 'with a doctor or dietitian.'),
    'alcohol': ('Your reported alcohol intake is in a range that adds liver risk. Please discuss your liver '
                'health and alcohol with a clinician; diet changes alone are not enough.'),
    'glucose_lowering_medicine': ('You take medicine that lowers blood glucose. Changing your diet can change '
                                  'how much you need, so please discuss diet changes with your clinician first.'),
    'pregnant': 'You are pregnant. Please discuss any diet change with your midwife or doctor.',
}


def _num(row, name):
    try:
        value = float(row.get(name))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def alcohol_g_per_week(row):
    """Average grams of alcohol a week; 0 for lifelong abstainers; None when unknown."""
    if _num(row, 'ALQ111') == 2:
        return 0.0
    freq, drinks = _num(row, 'ALQ121'), _num(row, 'ALQ130')
    if freq is None or int(freq) not in DAYS_PER_WEEK:
        return None
    days = DAYS_PER_WEEK[int(freq)]
    if days == 0:
        return 0.0
    if drinks is None or drinks >= 777:
        return None
    return days * drinks * GRAMS_PER_DRINK


def alcohol_band(row):
    grams = alcohol_g_per_week(row)
    if grams is None:
        return 'unknown', None
    sex = 'male' if _num(row, 'RIAGENDR') == MALE else 'female'
    low, high = METALD_G_PER_WEEK[sex]
    band = 'below_metald' if grams < low else ('metald_range' if grams <= high else 'above_metald')
    return band, round(grams, 1)


class BmiForAge:
    def __init__(self):
        table = pd.read_csv(GROWTH_DIR / 'bmiagerev.csv', usecols=['Sex', 'Agemos', 'L', 'M', 'S'])
        self.tables = {sex: g.sort_values('Agemos') for sex, g in table.groupby('Sex')}

    def at_or_above_85th(self, bmi, sex, age_years):
        z = z_score(bmi, *lms_at(self.tables, sex, 12 * age_years + 6))
        return bool(z >= P85_Z)


_BMI_FOR_AGE = None


def _bmi_for_age():
    global _BMI_FOR_AGE
    if _BMI_FOR_AGE is None:
        _BMI_FOR_AGE = BmiForAge()
    return _BMI_FOR_AGE


def cardiometabolic_criteria(row):
    """Which of the five CMRF are met (True/False, None when no data)."""
    age, male = _num(row, 'RIDAGEYR'), _num(row, 'RIAGENDR') == MALE
    bmi, waist = _num(row, 'BMXBMI'), _num(row, 'BMXWAIST')
    adult = age is None or age >= ADULT_AGE

    def any_known(*values):
        known = [v for v in values if v is not None]
        return any(known) if known else None

    if adult:
        bmi_cut = 23 if _num(row, 'RIDRETH3') == ASIAN else 25
        body = any_known(None if bmi is None else bmi >= bmi_cut,
                         None if waist is None else waist > (94 if male else 80))
    else:
        body = None if bmi is None else _bmi_for_age().at_or_above_85th(bmi, 1 if male else 2, age)
    glucose = _num(row, 'LBXGLU')
    random_glucose = _num(row, 'LBXSGL')
    glycaemia = any_known(None if glucose is None else glucose >= 100,
                          None if adult or random_glucose is None else random_glucose >= 200,
                          None if _num(row, 'LBXGH') is None else _num(row, 'LBXGH') >= 5.7,
                          _num(row, 'DIQ010') == 1 if _num(row, 'DIQ010') is not None else None,
                          glucose_lowering_medicine(row) or None)
    systolic = [v for v in (_num(row, f'BPXOSY{i}') for i in (1, 2, 3)) if v is not None]
    diastolic = [v for v in (_num(row, f'BPXODI{i}') for i in (1, 2, 3)) if v is not None]
    dia_cut = 80 if (not adult and age is not None and age < 13) else 85
    pressure = any_known(None if not systolic else np.mean(systolic) >= 130,
                         None if not diastolic else np.mean(diastolic) >= dia_cut,
                         _num(row, 'BPQ020') == 1 if _num(row, 'BPQ020') is not None else None)
    tg = _num(row, 'LBXSTR')
    triglycerides = None if tg is None else tg >= 150
    hdl = _num(row, 'LBDHDD')
    hdl_cut = 40 if (male or not adult) else 50
    low_hdl = None if hdl is None else hdl <= hdl_cut
    return {'body_size': body, 'glycaemia': glycaemia, 'blood_pressure': pressure,
            'triglycerides': triglycerides, 'low_hdl': low_hdl}


def glucose_lowering_medicine(row):
    return _num(row, 'DIQ050') == 1 or _num(row, 'DIQ070') == 1


def liver_label(row, steatosis):
    """Subtype label for a patient whose steatosis status is `steatosis` (True/False/None)."""
    criteria = cardiometabolic_criteria(row)
    met = [k for k, v in criteria.items() if v]
    band, grams = alcohol_band(row)
    if not steatosis:
        label = 'no_steatosis_indicated' if steatosis is not None else 'unknown'
    elif band == 'above_metald':
        label = 'ALD'
    elif not met:
        label = 'cryptogenic_SLD'
    elif band == 'metald_range':
        label = 'MetALD'
    else:
        label = 'MASLD'
    return {'label': label, 'criteria_met': met, 'criteria': criteria, 'alcohol_g_per_week': grams,
            'alcohol_band': band,
            'note': ('Research label from the 2023 nomenclature; other causes of steatosis were not assessed, '
                     'so this is not a diagnosis.')}


def guardrails(row, fibrosis=None):
    """Notices to show before recommendations; `block_recipes` when clinical assessment comes first."""
    notices = []
    age = _num(row, 'RIDAGEYR')
    if age is not None and age < ADULT_AGE:
        notices.append('under_18')
    if alcohol_band(row)[0] in ('metald_range', 'above_metald'):
        notices.append('alcohol')
    if glucose_lowering_medicine(row):
        notices.append('glucose_lowering_medicine')
    if _num(row, 'RIDEXPRG') == 1:
        notices.append('pregnant')
    out = [{'id': n, 'message': NOTICES[n]} for n in notices]
    block = bool(fibrosis and fibrosis.get('refer_before_recipes'))
    if block:
        out.insert(0, {'id': 'fibrosis_referral', 'message': fibrosis['message']})
    return {'notices': out, 'block_recipes': block}


def main():
    """Label distribution in the in-scope cohort, using measured CAP >= 248 dB/m as steatosis."""
    from src.clinical_data import load_extras
    from src.train_final_model import ROOT
    out = ROOT / 'data' / 'results' / 'clinical'
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data' / 'processed' / 'nhanes_combined.csv').merge(load_extras(), on='SEQN', how='left')
    data = data[(data.RIDAGEYR < 60) & (data.LUXCPIQR < 40)]
    rows = []
    for _, row in data.iterrows():
        label = liver_label(row, bool(row.LUXCAPM >= 248))
        notes = guardrails(row)
        rows.append({'SEQN': int(row.SEQN), 'adult': row.RIDAGEYR >= ADULT_AGE, 'label': label['label'],
                     'n_criteria': len(label['criteria_met']), 'alcohol_band': label['alcohol_band'],
                     **{f'notice_{n["id"]}': True for n in notes['notices']}})
    table = pd.DataFrame(rows)
    notice_cols = [c for c in table if c.startswith('notice_')]
    table[notice_cols] = table[notice_cols].astype('boolean').fillna(False).astype(bool)
    table.to_csv(out / 'liver_labels.csv', index=False)
    summary = table.groupby('adult').label.value_counts(normalize=True).round(4).rename('share').reset_index()
    summary.to_csv(out / 'liver_label_summary.csv', index=False)
    print(f'{len(table)} in-scope participants')
    print(summary.to_string(index=False))
    print(table.filter(like='notice_').mean().round(4).to_string())


if __name__ == '__main__':
    main()
