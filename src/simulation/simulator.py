"""True monthly trajectories of virtual patients (the simulator's hidden truth).

For each month-0 patient and diet scenario the simulator generates months 0-12:

1. Adherence: a person-level pattern (steady, slipping, regaining or dropping out) scales the
   scenario's prescribed change in energy intake.
2. Weight: Hall et al. (2011): each sustained 100 kJ/day changes weight by about 1 kg in the end,
   approached as a single exponential fitted to "half in a year, 95% in three years". Each person's
   eventual change and speed are scaled by their body fat (Hall's web appendix; body fat from BMI,
   Deurenberg 1991): people with more fat lose more in the end, but more slowly.
3. Growth: patients aged 12-17 also grow along their own CDC weight and height centiles.
4. Body and blood markers: waist, hip and arm change per kg; HbA1c, glucose, insulin, HDL,
   triglycerides and blood pressure change by a share of the patient's own baseline per 1% weight
   change, with Look AHEAD values (Wing 2011) for patients with diabetes and pooled trials in adults
   without diabetes (Appuhamy 2014) otherwise; CRP (Selvin 2007), uric acid (Yokose 2020) and the
   white-cell count and its
   lymphocyte and neutrophil shares (Dixon 2006) change per kg; other blood counts and ferritin stay
   at month 0 (CBC-STABLE in the parameter table). Each person responds
   more or less strongly than the published average, with responses correlated across markers. A
   change never pushes a value past the cohort's 5th or 95th percentile (unless it started there).
5. Liver: baseline liver fat (MRI-PDFF) is read from baseline CAP (Wang 2020 cut-offs). Liver fat
   changes with the weight lost (Patel 2015, Lim 2011) and falls further while the patient is eating
   less (Lim 2011's first week), with its own person-level spread; CAP follows the liver fat, and
   ALT, AST and GGT follow the change in liver fat (Koutoukidis 2019 with Wong 2013).

Growth is not energy restriction: markers and liver fat respond to the diet-driven weight change
only, while body size and BMI use the total. A person's biology and adherence are drawn once per
person (seeded by SEQN), so scenarios for the same person differ only in their diet. This module
never loads the Layer 2 model.
"""
import numpy as np
import pandas as pd

from src.simulation.growth import Growth
from src.simulation.parameters import KJ_PER_KCAL, Calibration, Evidence, body_fat_kg, personal_weight_response

MONTHS = 12
PATTERNS = ('steady', 'slipping', 'regaining', 'dropping_out')
GLUCOSE = ('LBXGLU', 'LBXSGL')
SYSTOLIC = ('BPXOSY1', 'BPXOSY2', 'BPXOSY3')
INTAKE = ('DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT')
ENZYMES = ('LBXSATSI', 'LBXSASSI', 'LBXSGTSI')
COUNTS = ('LBXWBCSI', 'LBXLYPCT', 'LBXNEPCT')
CIRCUMFERENCES = ('BMXWAIST', 'BMXHIP', 'BMXARMC')
RESPONSES = ('LBXGH', 'glucose', 'LBXIN', 'LBDHDD', 'LBXSTR', 'sbp', 'LBXHSCRP', 'LBXWBCSI', 'LBXSUA') + ENZYMES
BOUNDED = ('LBXGH', 'LBXIN', 'LBDHDD', 'LBXSTR', 'LBXHSCRP', 'LBXSUA') + ENZYMES + GLUCOSE + SYSTOLIC + COUNTS
CAP_RANGE = (100, 400)   # FibroScan reports CAP from 100 to 400 dB/m
LB_PER_KG = 1 / 0.4536
DIABETES_HBA1C = 6.5     # % (NGSP): diagnostic threshold, used with the self-report question DIQ010


def marker_bounds(cohort, low=0.05, high=0.95):
    """Plausible range of each changing marker: the cohort's 5th and 95th percentiles."""
    return {col: (cohort[col].quantile(low), cohort[col].quantile(high)) for col in BOUNDED}


def bounded(baseline, change, lo, hi):
    """Apply a change without crossing the bound it moves towards (unless already beyond it)."""
    if not np.isfinite(baseline):
        return np.nan
    if change < 0:
        return max(baseline + change, min(baseline, lo))
    return min(baseline + change, max(baseline, hi))


def has_diabetes(row):
    return row.get('DIQ010') == 1 or (pd.notna(row.get('LBXGH')) and row['LBXGH'] >= DIABETES_HBA1C)


def person_traits(seqn, seed, cal):
    """Adherence pattern and response multipliers of one person, fixed across scenarios."""
    rng = np.random.default_rng([seed, int(seqn)])
    shares = np.array([cal.pattern_shares[p] for p in PATTERNS], dtype=float)
    pattern = PATTERNS[rng.choice(len(PATTERNS), p=shares / shares.sum())]
    stop = {'regaining': int(rng.integers(cal.regain_month_range[0], cal.regain_month_range[1] + 1)),
            'dropping_out': int(rng.integers(cal.dropout_month_range[0], cal.dropout_month_range[1] + 1))}

    def lognormal(sd, z):
        return float(np.exp(sd * z - sd ** 2 / 2))   # mean 1

    shared = rng.standard_normal()
    rho = cal.shared_response_corr
    markers = {m: lognormal(cal.marker_response_sd, rho * shared + np.sqrt(1 - rho ** 2) * rng.standard_normal())
               for m in RESPONSES}
    return {'pattern': pattern,
            'adherence0': float(np.clip(rng.normal(cal.adherence_mean, cal.adherence_sd), 0, 1)),
            'stop_month': stop.get(pattern),
            'weight_response': lognormal(cal.weight_response_sd, rng.standard_normal()),
            'liver_response': lognormal(cal.liver_response_sd, rng.standard_normal()),
            'markers': markers}


def adherence_path(traits, cal):
    """Adherence during each month 0..MONTHS-1 (the month after each visit)."""
    t = np.arange(MONTHS, dtype=float)
    a0, pattern = traits['adherence0'], traits['pattern']
    if pattern == 'slipping':
        return a0 * np.exp(-t / cal.slip_months)
    if pattern in ('regaining', 'dropping_out'):
        return np.where(t < traits['stop_month'], a0, 0.0)
    return np.full(MONTHS, a0)


def weight_path(energy_change_kcal, adherence, response, ev, tau_months=None):
    """Diet-driven weight change (kg) at months 0..MONTHS from Hall's rule, adherence varying by month.
    `response` scales the eventual change; `tau_months` is the person's time constant."""
    decay = np.exp(-1 / (ev.weight_tau_months if tau_months is None else tau_months))
    change = np.zeros(MONTHS + 1)
    for month in range(MONTHS):
        eventual = ev.kg_per_100kj * energy_change_kcal * adherence[month] * KJ_PER_KCAL / 100 * response
        change[month + 1] = eventual + (change[month] - eventual) * decay
    return change


def baseline_pdff(cap, ev, cal):
    """Liver fat (% PDFF) implied by a CAP reading, on the line through Wang et al.'s cut-offs."""
    return max(cal.min_pdff, ev.anchor_pdff + (cap - ev.anchor_cap) / ev.cap_per_pdff_point)


def simulate_patient(row, scenario, ev, cal, bounds, seed=0, growth=None):
    """Months 0..MONTHS of one patient under one scenario: a DataFrame of true values."""
    traits = person_traits(row.SEQN, seed, cal)
    # A scenario's prescribed change in energy intake (kcal/day): a number, or a rule of the patient
    # (for example a fixed intake target such as Look AHEAD's 1,200-1,800 kcal/day).
    change = scenario['energy_change_kcal']
    energy = float(change(row) if callable(change) else change)
    adherence = adherence_path(traits, cal) if energy else np.zeros(MONTHS)
    months = np.arange(MONTHS + 1)
    w0, h0, age, male = float(row.BMXWT), float(row.BMXHT), float(row.RIDAGEYR), row.RIAGENDR == 1
    fat = body_fat_kg(w0, float(row.BMXBMI), age, male)
    eventual, tau = personal_weight_response(fat, ev)
    diet_kg = weight_path(energy, adherence, traits['weight_response'] * eventual, ev, tau)
    diet_pct = 100 * diet_kg / w0
    growth = Growth() if growth is None else growth
    grow_kg = growth.weight_path(w0, row.RIAGENDR, age, months)
    grow_cm = growth.height_path(h0, row.RIAGENDR, age, months)
    total_kg = diet_kg + grow_kg
    diabetes = has_diabetes(row)
    slopes = ev.per_pct_diabetes if diabetes else ev.per_pct_no_diabetes
    pdff0 = baseline_pdff(float(row.LUXCAPM), ev, cal)
    whd050_kg = float(row.WHD050) / LB_PER_KG if pd.notna(row.WHD050) else np.nan
    usual_kcal = float(row.DR1TKCAL)
    m = traits['markers']

    rows = []
    for month in months:
        cut = -energy * adherence[month - 1] if month else 0.0          # kcal/day eaten less last month
        out = {'SEQN': int(row.SEQN), 'scenario': scenario['name'], 'month': int(month),
               'pattern': traits['pattern'], 'adherence': float(adherence[month - 1]) if month else 0.0,
               'dropped_out': traits['pattern'] == 'dropping_out' and month > traits['stop_month'],
               'BMXWT': w0 + total_kg[month], 'BMXHT': h0 + grow_cm[month]}
        out['BMXBMI'] = out['BMXWT'] / (out['BMXHT'] / 100) ** 2
        for col in CIRCUMFERENCES:
            out[col] = float(row[col]) + ev.circumference_cm_per_kg[col] * total_kg[month]

        def relative(col, key):
            return bounded(float(row[col]), float(row[col]) * slopes[key] / 100 * diet_pct[month] * m[key],
                           *bounds[col])
        for col in ('LBXGH', 'LBXIN', 'LBDHDD', 'LBXSTR'):
            out[col] = relative(col, col)
        for col in GLUCOSE:
            out[col] = relative(col, 'glucose')
        for col in SYSTOLIC:
            out[col] = relative(col, 'sbp')
        out['LBXHSCRP'] = bounded(float(row.LBXHSCRP), ev.crp_per_kg * diet_kg[month] * m['LBXHSCRP'],
                                  *bounds['LBXHSCRP'])
        out['LBXSUA'] = bounded(float(row.LBXSUA), ev.urate_per_kg[bool(diabetes)] * diet_kg[month] * m['LBXSUA'],
                                *bounds['LBXSUA'])
        for col in COUNTS:
            response = m['LBXWBCSI'] if col == 'LBXWBCSI' else 1.0
            out[col] = bounded(float(row[col]), float(row[col]) * ev.counts_pct_per_kg[col] / 100 * diet_kg[month]
                               * response, *bounds[col])
        # Total cholesterol = LDL + HDL + TG/5 (Friedewald); LDL does not change with weight (Wing 2011).
        out['LBXTC'] = float(row.LBXTC) + (out['LBDHDD'] - float(row.LBDHDD)) \
            + (out['LBXSTR'] - float(row.LBXSTR)) / 5

        # Liver fat: weight-driven change plus an extra fall while eating less (Lim's first week).
        deficit = max(cut, 0.0) / usual_kcal if usual_kcal > 0 else 0.0
        relative_fat = (ev.liver_fat_pct_per_pct / 100 * diet_pct[month]
                        - ev.liver_fat_pct_per_deficit / 100 * min(deficit, 1.0)) * traits['liver_response']
        pdff = pdff0 * (1 + float(np.clip(relative_fat, -cal.max_liver_fat_fall, 1.0)))
        out['true_pdff'] = pdff
        out['LUXCAPM'] = float(np.clip(float(row.LUXCAPM) + (pdff - pdff0) * ev.cap_per_pdff_point, *CAP_RANGE))
        for col in ENZYMES:
            out[col] = bounded(float(row[col]), ev.enzymes_per_pdff_point[col] * (pdff - pdff0) * m[col],
                               *bounds[col])

        # Intake during the month before this visit; macronutrients keep their usual proportions.
        kcal = usual_kcal - cut
        scale = kcal / usual_kcal if usual_kcal > 0 else np.nan
        out |= {col: float(row[col]) * scale for col in INTAKE}
        # Weight a year before this visit, in pounds as NHANES records it: before month 12 it lies
        # between the reported weight a year before month 0 and the month-0 weight.
        out['WHD050'] = (w0 + total_kg[month - 12] if month >= 12
                         else whd050_kg + (w0 - whd050_kg) * month / 12) * LB_PER_KG
        rows.append(out)
    frame = pd.DataFrame(rows)
    traits_row = {'SEQN': int(row.SEQN), 'pattern': traits['pattern'], 'adherence0': traits['adherence0'],
                  'stop_month': traits['stop_month'], 'weight_response': traits['weight_response'],
                  'liver_response': traits['liver_response'], 'baseline_pdff': pdff0, 'body_fat_kg': fat,
                  'eventual_scale': eventual, 'weight_tau_months': tau, 'diabetes': bool(diabetes),
                  'growing': age < 18, **{f'response_{k}': v for k, v in m.items()}}
    return frame, traits_row


def simulate_cohort(cohort, scenarios, seed=0, ev=None, cal=None):
    """True trajectories for every patient and scenario, plus one row of traits per patient."""
    ev = Evidence.from_table() if ev is None else ev
    cal = Calibration() if cal is None else cal
    bounds = marker_bounds(cohort)
    growth = Growth()
    frames, traits = [], {}
    for _, row in cohort.iterrows():
        for scenario in scenarios:
            frame, person = simulate_patient(row, scenario, ev, cal, bounds, seed, growth)
            frames.append(frame)
            traits[person['SEQN']] = person
    return pd.concat(frames, ignore_index=True), pd.DataFrame(list(traits.values()))
