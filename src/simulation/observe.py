"""Observation layer: what the twin sees of each virtual patient (Proposal section 17.14).

The simulator's truth is never shown to the twin. This module turns it into the measurements a
real patient would produce, using the OB- rows of data/evidence/simulator_parameters.csv:

1. Schedule (OB-SCHEDULE): weight and food log every month; blood tests, blood pressure and body
   circumferences every 3 months; CAP at months 0, 6 and 12.
2. Laboratory noise: each blood test varies within a person from day to day (OB-CVI-*, EFLM
   database) and the laboratory adds its own error, half the within-person CV (OB-CVA, Fraser's
   specification). The two combine as sqrt(CVI^2 + CVA^2) and act multiplicatively.
3. CAP noise (OB-CAP-OP): two operators agree with a concordance of 0.82, so the measurement error
   is set to the SD that gives this concordance across the cohort's baseline CAP:
   SD_error = SD_cohort * sqrt(1 / 0.82 - 1). OB-CAP-CHANGE (38 dB/m) is used to check the result.
4. Food logs (OB-ENERGY-M/W, OB-PROTEIN): each person under-reports a fixed share of their true
   intake, drawn once from the OPEN study's range for their sex. Carbohydrate and fat are taken to
   be under-reported like energy, which OPEN did not measure. The NHANES recall at month 0 is
   itself such a report, so the true intake is recall / (1 - share) and a later log reports
   (true intake - cut) * (1 - share): the log shows a smaller cut than the patient made.
5. Missed visits (OB-MISSING): a visit is missed with a small probability, which rises when the
   patient's diet-driven weight rose since the last month (growth in teenagers does not count); a
   patient who drops out reports nothing afterwards. The two probabilities are tuned so that the
   number of monthly visits missed in the first year matches Look AHEAD (OB-MISS-LA, Neiberg 2012).

Month 0 is the patient's real NHANES examination, so its values are passed through unchanged.
Measurements with no cited noise (blood pressure, CRP, uric acid, blood counts, circumferences,
weight) are recorded at their scheduled visits without added noise. A person's noise draws are
seeded by SEQN and do not depend on the scenario, so scenarios differ only in what truly happened.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulation.parameters import load_table, value
from src.simulation.simulator import CAP_RANGE, CIRCUMFERENCES, COUNTS, GLUCOSE, INTAKE, MONTHS, SYSTOLIC

LAB_CVI = {'LBXSATSI': 'OB-CVI-ALT', 'LBXSASSI': 'OB-CVI-AST', 'LBXSGTSI': 'OB-CVI-GGT', 'LBXGLU': 'OB-CVI-GLU',
           'LBXSGL': 'OB-CVI-GLU', 'LBXGH': 'OB-CVI-HBA1C', 'LBXIN': 'OB-CVI-INS', 'LBXSTR': 'OB-CVI-TG',
           'LBDHDD': 'OB-CVI-HDL'}
BLOODS = tuple(LAB_CVI) + ('LBXTC', 'LBXHSCRP', 'LBXSUA') + COUNTS
CLINIC = SYSTOLIC + CIRCUMFERENCES
MONTHLY = ('BMXWT', 'BMXBMI', 'WHD050') + INTAKE
ENERGY_LIKE = ('DR1TKCAL', 'DR1TCARB', 'DR1TTFAT')


@dataclass(frozen=True)
class ObservationEvidence:
    """Literature values of the observation layer, each traceable to a param_id."""
    total_cv: dict          # % CV of each laboratory column, biological and analytical combined
    cap_concordance: float  # OB-CAP-OP
    cap_change: float       # OB-CAP-CHANGE (dB/m), for checking
    energy_share: dict      # (low, high) % under-reported, by RIAGENDR (1 men, 2 women)
    protein_share: tuple    # (low, high) % under-reported

    @classmethod
    def from_table(cls, table=None):
        table = load_table() if table is None else table
        analytical = value(table, 'OB-CVA')
        total = {col: value(table, pid) * np.sqrt(1 + analytical ** 2) for col, pid in LAB_CVI.items()}

        def bounds(pid):
            return float(table.loc[pid, 'low']), float(table.loc[pid, 'high'])
        return cls(total_cv=total, cap_concordance=value(table, 'OB-CAP-OP'),
                   cap_change=value(table, 'OB-CAP-CHANGE'),
                   energy_share={1: bounds('OB-ENERGY-M'), 2: bounds('OB-ENERGY-W')},
                   protein_share=bounds('OB-PROTEIN'))


@dataclass(frozen=True)
class ObservationDesign:
    """Schedule and missed visits (OB-SCHEDULE, OB-MISSING). The two chances of missing a visit were
    fitted on the tuned diet scenarios to the first-year missed-visit rates of Look AHEAD (OB-MISS-LA):
    70% vs 69% miss none, 18% vs 14% miss one, 5% vs 9% miss 2-3, 2% vs 4% miss 4-6, 5% vs 3% miss
    more than 6. Real misses come in runs more than these independent ones do."""
    blood_months: tuple = (0, 3, 6, 9, 12)
    cap_months: tuple = (0, 6, 12)
    miss_probability: float = 0.005       # chance of missing a visit when doing well
    miss_probability_gaining: float = 0.04  # when the patient gained weight since the last month


def cap_error_sd(baseline_cap, concordance):
    """Measurement SD that makes two independent readings agree with this concordance."""
    return float(np.std(baseline_cap, ddof=1) * np.sqrt(1 / concordance - 1))


def multiplicative(truth, cv_percent, z):
    """Truth times a log-normal error with mean 1 and this CV."""
    sigma = np.sqrt(np.log(1 + (cv_percent / 100) ** 2))
    return truth * np.exp(sigma * z - sigma ** 2 / 2)


def person_reporting(seqn, sex, seed, oev):
    """A person's under-reported shares (fractions), fixed across months and scenarios."""
    rng = np.random.default_rng([seed, 1, int(seqn)])
    energy = rng.uniform(*oev.energy_share[1 if sex == 1 else 2]) / 100
    return {'energy': float(energy), 'protein': float(rng.uniform(*oev.protein_share) / 100)}


def person_draws(seqn, seed):
    """A person's laboratory, CAP and attendance draws, the same in every scenario."""
    rng = np.random.default_rng([seed, 2, int(seqn)])
    lab_z = rng.standard_normal((MONTHS + 1, len(LAB_CVI)))
    cap_z = rng.standard_normal(MONTHS + 1)
    return lab_z, cap_z, rng.uniform(size=MONTHS + 1)


def attendance(dropped_out, diet_weight_kg, miss_u, design):
    """Whether each visit 0..MONTHS was attended; month 0 always is."""
    gaining = np.r_[False, np.diff(diet_weight_kg) > 0]
    risk = np.where(gaining, design.miss_probability_gaining, design.miss_probability)
    attended = (miss_u >= risk) & ~np.asarray(dropped_out, dtype=bool)
    attended[0] = True
    return attended


def missed_visit_shares(truth, seed, design):
    """Share of patient-scenarios missing 0, 1, 2-3, 4-6 and more than 6 of the 12 monthly visits."""
    missed = []
    for (seqn, _), path in truth.groupby(['SEQN', 'scenario'], sort=False):
        path = path.sort_values('month')
        attended = attendance(path.dropped_out.to_numpy(), path.diet_weight_kg.to_numpy(),
                              person_draws(seqn, seed)[2], design)
        missed.append(int((~attended[1:]).sum()))
    counts = np.bincount(missed, minlength=MONTHS + 1) / len(missed)
    return {'0': counts[0], '1': counts[1], '2-3': counts[2:4].sum(), '4-6': counts[4:7].sum(), '>6': counts[7:].sum()}


def observe_patient(truth, baseline, oev, design, cap_sd, seed=0):
    """Observed months 0..MONTHS of one patient under one scenario (NaN = not measured)."""
    seqn = int(baseline.SEQN)
    lab_z, cap_z, miss_u = person_draws(seqn, seed)
    shares = person_reporting(seqn, baseline.RIAGENDR, seed, oev)
    truth = truth.sort_values('month').reset_index(drop=True)
    visits = attendance(truth.dropped_out.to_numpy(), truth.diet_weight_kg.to_numpy(), miss_u, design)

    rows = []
    for month, t in truth.iterrows():
        out = {'SEQN': seqn, 'scenario': t.scenario, 'month': int(month)}
        attended = bool(visits[month])
        out['attended'] = attended
        if not attended:
            rows.append(out)
            continue
        if month == 0:
            out |= {col: float(baseline[col]) for col in MONTHLY + BLOODS + CLINIC + ('LUXCAPM',)}
            rows.append(out)
            continue
        out |= {col: float(t[col]) for col in ('BMXWT', 'BMXBMI', 'WHD050')}
        for col in INTAKE:
            share = shares['protein'] if col == 'DR1TPROT' else shares['energy']
            recall = float(baseline[col])
            # True intake is recall / (1 - share) minus the cut; the log reports (1 - share) of it.
            out[col] = recall - (recall - float(t[col])) * (1 - share)
        if month in design.blood_months:
            for i, col in enumerate(LAB_CVI):
                out[col] = multiplicative(float(t[col]), oev.total_cv[col], lab_z[month, i])
            out |= {col: float(t[col]) for col in BLOODS if col not in LAB_CVI}
            out |= {col: float(t[col]) for col in CLINIC}
        if month in design.cap_months:
            out['LUXCAPM'] = float(np.clip(t.LUXCAPM + cap_sd * cap_z[month], *CAP_RANGE))
        rows.append(out)
    frame = pd.DataFrame(rows)
    columns = ['SEQN', 'scenario', 'month', 'attended', *MONTHLY, *BLOODS, *CLINIC, 'LUXCAPM']
    return frame.reindex(columns=columns), {'SEQN': seqn, 'energy_underreport': shares['energy'],
                                            'protein_underreport': shares['protein']}


def observe_cohort(truth, cohort, seed=0, oev=None, design=None):
    """Observations for every patient and scenario, and each person's reporting shares."""
    oev = ObservationEvidence.from_table() if oev is None else oev
    design = ObservationDesign() if design is None else design
    cap_sd = cap_error_sd(cohort.LUXCAPM, oev.cap_concordance)
    baselines = cohort.set_index('SEQN', drop=False)
    frames, people = [], {}
    for (seqn, _), path in truth.groupby(['SEQN', 'scenario'], sort=False):
        frame, person = observe_patient(path, baselines.loc[seqn], oev, design, cap_sd, seed)
        frames.append(frame)
        people[seqn] = person
    return pd.concat(frames, ignore_index=True), pd.DataFrame(list(people.values())), cap_sd
