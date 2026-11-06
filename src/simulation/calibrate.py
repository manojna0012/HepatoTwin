"""Calibrate the simulator's untested settings so that trial-like virtual patients reproduce the
published weight outcomes (Weekly Plan, meeting 11).

Each trial is re-created from real NHANES participants who meet its entry criteria, drawn from the
training partition so that the test patients stay unseen until the evaluation. Each follows the
trial's own prescription (RX- rows of the parameter table):

- Look AHEAD intensive lifestyle: an intake goal of 1,350 kcal/day below 114 kg and 1,650 above,
  against the calories that maintain the person's weight. Those maintenance calories come from the
  DPP: its calorie goal for the person's weight band plus that band's 500-1,000 kcal/day deficit.
- Look AHEAD usual care: no prescription, only the natural weight drift.
- DPP lifestyle: the 500-1,000 kcal/day deficit of the person's weight band.
- Vilar-Gomez: 750 kcal/day below the person's daily energy need.

Each programme also prescribed walking (RX-EX-LA 175, RX-EX-DPP 150, RX-EX-VG 90 rising to 200
minutes a week), built up over the first 6 months; its energy cost is (MET - 1) x weight x hours
with moderate-pace walking at 3.8 MET (EX-MET). Look AHEAD also replaced meals with shakes in weeks
3-19 (RX-LA-MR, an extra deficit sized to the cited 3 kg). Walking and meal replacements are
followed as closely as the diet: the same adherence scales the whole prescription.

The settings shared by all trials (adherence patterns, how adherence varies and fades, how strongly
people respond, natural drift) are tuned together; the starting adherence is tuned per trial,
because an intensive programme with meal replacements is followed more closely than routine
clinical advice. During each programme's intensive-contact phase (DPP's 24-week one-to-one core
curriculum, Look AHEAD's 6 months of weekly sessions; RX-CONTACT-*) a 'slipping' participant keeps
full effort and only eases off afterwards; without this the early loss was too slow (DPP's 24-week
goal 37% against 50%). HepatoTwin's own scenarios use the Vilar-Gomez adherence: routine advice to
patients with fatty liver, the setting the twin is meant for. The share of patients who drop out
is not tuned but set from Vilar-Gomez's loss to follow-up (CAL-DROPOUT): stopping the diet looks
the same in weight whether the patient keeps reporting (regaining) or not (dropping out), so the
weight targets alone cannot tell the two apart.

Mean weight results also cannot tell a plateau caused by most people quitting from one caused by
most people easing off. The share who stop the diet completely (regaining plus dropping out) is
therefore capped (--quit-cap) and the tuning repeated for several caps; the cap is chosen by how
little Look AHEAD patients regained within the first year (LA-Y1-REGAIN-T1, Neiberg 2012), a
target used only for this choice. Vilar-Gomez is fitted on its Table 2 weight-loss groups, which
agree with its 3.8% mean and 30% losing >= 5%, and on the shape (not the size) of the completers'
weight course in Supplementary Table 2, whose 4.6 kg does not agree with Table 2.

The tuning runs the simulator's weight model for every person at once (Cohort) and searches the
bounded settings with differential evolution, then polishes with Nelder-Mead. Only the targets
marked 'calibrate' are used for tuning. The held-back targets (Look AHEAD year 8,
DPP at about 3 years) are checked once afterwards, extending each adherence pattern beyond the
first year as it was (steady stays steady, a slipping patient keeps slipping, a stopped patient
stays stopped).

    .venv/Scripts/python.exe -m src.simulation.calibrate --quit-cap 0.2
"""
import argparse
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from src.simulation.cohort import DATA_FILE, MANIFEST_FILE
from src.simulation.parameters import KJ_PER_KCAL, ROOT, TARGET_FILE, Calibration, Evidence, body_fat_kg, \
    load_table, personal_weight_response
from src.simulation.simulator import drift_path, has_diabetes, lognormal, pattern_of, slipping_adherence, trait_draws

OUTPUT = ROOT / 'data' / 'results' / 'simulation' / 'calibration'
SEED = 42
LB_PER_KG = 1 / 0.4536
TRIALS = ('look_ahead', 'dpp', 'vilar_gomez')
PATTERNS = ('steady', 'slipping', 'regaining', 'dropping_out')
# Spread used to weigh each kind of target in the loss: about one "acceptable miss".
SCALE = {'% of initial weight': 1.0, '% of participants': 5.0, 'kg': 0.5, 'fraction': 0.05}
VG_WEEKS = (8, 16, 24, 32, 40, 52)
VG_GROUPS = {'VG-GRP-LT5': (-np.inf, 5), 'VG-GRP-5TO7': (5, 7), 'VG-GRP-7TO10': (7, 10), 'VG-GRP-GE10': (10, np.inf)}


def numbers(text):
    return [float(part) for part in str(text).split(';')]


EXERCISE_RAMP_MONTHS = 6
MEAL_REPLACEMENT_MONTHS = (2 * 7 / 30.4375, 19 * 7 / 30.4375)   # weeks 3-19


def walking_kcal(kg, months, goal_minutes, start_minutes, met):
    """Extra energy used by walking (kcal/day, negative) in each month, the weekly minutes rising
    linearly from start to goal over the first EXERCISE_RAMP_MONTHS (taken at each month's middle)."""
    middle = np.arange(months) + 0.5
    minutes = start_minutes + (goal_minutes - start_minutes) * np.minimum(middle / EXERCISE_RAMP_MONTHS, 1)
    return -(met - 1) * kg * minutes / 60 / 7


def in_window(months, window):
    """Share of each month 0..months-1 that falls inside the window (start, end), in months."""
    start = np.arange(months)
    return np.clip(np.minimum(start + 1, window[1]) - np.maximum(start, window[0]), 0, 1)


def prescriptions(table):
    """Energy change (kcal/day) in each month of each trial arm: a function of weight (kg) and the
    number of months, returning one value per month."""
    goal = numbers(table.loc['RX-DPP-GOAL', 'value'])
    deficit = numbers(table.loc['RX-DPP-DEFICIT', 'value'])
    la_goal = numbers(table.loc['RX-LA-GOAL', 'value'])
    vg = float(table.loc['RX-VG-DEFICIT', 'value'])
    met = float(table.loc['EX-MET', 'value'])
    walk = {arm: (float(table.loc[pid, 'value']), float(table.loc[pid, 'low']) if arm == 'vilar_gomez' else 0.0)
            for arm, pid in (('look_ahead', 'RX-EX-LA'), ('dpp', 'RX-EX-DPP'), ('vilar_gomez', 'RX-EX-VG'))}
    meal_replacement = -float(table.loc['RX-LA-MR', 'value'])

    def band(kg):
        return int(np.searchsorted([175, 220, 250], kg * LB_PER_KG, side='right'))

    def maintenance(kg):
        return goal[band(kg)] + deficit[band(kg)]

    def programme(arm, diet):
        def energy(kg, months):
            out = diet(kg) + walking_kcal(kg, months, *walk[arm], met)
            if arm == 'look_ahead':
                out = out + meal_replacement * in_window(months, MEAL_REPLACEMENT_MONTHS)
            return out
        return energy
    return {'look_ahead': programme('look_ahead', lambda kg: (la_goal[0] if kg < 114 else la_goal[1]) - maintenance(kg)),
            'usual_care': lambda kg, months: np.zeros(months),
            'dpp': programme('dpp', lambda kg: -deficit[band(kg)]),
            'vilar_gomez': programme('vilar_gomez', lambda kg: -vg)}


def trial_cohorts(data_file=DATA_FILE, manifest_file=MANIFEST_FILE):
    """NHANES training-partition participants meeting each trial's entry criteria (approximately)."""
    data = pd.read_csv(data_file)
    manifest = pd.read_csv(manifest_file)
    data = data[data.SEQN.isin(manifest.loc[manifest.partition.eq('train'), 'SEQN'])]
    data = data[data.BMXWT.notna() & data.BMXBMI.notna()]
    diabetes = data.apply(has_diabetes, axis=1)
    return {
        # 45-76 years, type 2 diabetes, BMI >= 25
        'look_ahead': data[data.RIDAGEYR.between(45, 76) & diabetes & (data.BMXBMI >= 25)],
        # at least 25 years, BMI >= 24, no diabetes, raised fasting glucose (the oral glucose test is not used)
        'dpp': data[(data.RIDAGEYR >= 25) & ~diabetes & (data.BMXBMI >= 24) & data.LBXGLU.between(95, 125)],
        # adults with overweight or obesity (BMI 25-40) and steatosis on a reliable CAP reading
        'vilar_gomez': data[(data.RIDAGEYR >= 18) & data.BMXBMI.between(25, 40) & (data.LUXCAPM >= 248)
                            & (data.LUXCPIQR < 40)],
    }


class Cohort:
    """One trial-like cohort with each person's random draws made once (they do not depend on the
    tuned settings), so that the simulator's weight model can be run for everyone at once."""

    def __init__(self, people, ev, months, seed=SEED, cal=None):
        cal = Calibration() if cal is None else cal
        self.people, self.months = people.reset_index(drop=True), months
        self.weight = self.people.BMXWT.to_numpy(float)
        draws = [trait_draws(s, seed, cal) for s in self.people.SEQN]
        self.pattern_u = np.array([d['pattern_u'] for d in draws])
        self.stop = {p: np.array([d['stop'][p] for d in draws]) for p in ('regaining', 'dropping_out')}
        self.adherence_z = np.array([d['adherence_z'] for d in draws])
        self.weight_z = np.array([d['weight_z'] for d in draws])
        self.drift_steps = np.array([np.diff(drift_path(s, seed, replace(cal, weight_drift_sd=1.0), months))
                                     for s in self.people.SEQN])
        response = [personal_weight_response(body_fat_kg(r.BMXWT, r.BMXBMI, r.RIDAGEYR, r.RIAGENDR == 1), ev)
                    for r in self.people.itertuples(index=False)]
        self.eventual = np.array([e for e, _ in response])
        self.decay = np.exp(-1 / np.array([tau for _, tau in response]))
        self.kg_per_kcal = ev.kg_per_100kj * KJ_PER_KCAL / 100

    def run(self, energy, cal, adherence_mean, contact_months=None):
        """Weight change (kg) at months 0..months, whether each person completed follow-up, and the
        diet-driven part alone: the simulator's adherence_path, weight_path and drift_path, vectorised.
        contact_months: the programme's intensive-contact phase (default: cal.contact_months)."""
        if contact_months is not None:
            cal = replace(cal, contact_months=contact_months)
        kcal = np.array([np.broadcast_to(energy(w, self.months), (self.months,)) for w in self.weight])
        pattern = pattern_of(self.pattern_u, [cal.pattern_shares[p] for p in PATTERNS])
        a0 = np.clip(adherence_mean + cal.adherence_sd * self.adherence_z, 0, 1)
        t = np.arange(self.months, dtype=float)[None, :]
        slipping = slipping_adherence(a0[:, None], t, cal)
        stop = np.where(pattern == 2, self.stop['regaining'], self.stop['dropping_out'])
        stopping = np.where(t < stop[:, None], a0[:, None], 0.0)
        adherence = np.select([pattern[:, None] == 0, pattern[:, None] == 1], [np.broadcast_to(a0[:, None], stopping.shape),
                                                                              slipping], stopping)
        adherence = np.where(np.any(kcal != 0, axis=1)[:, None], adherence, 0.0)
        scale = (self.kg_per_kcal * lognormal(cal.weight_response_sd, self.weight_z) * self.eventual)[:, None] * kcal
        diet = np.zeros((len(kcal), self.months + 1))
        for m in range(self.months):
            target = scale[:, m] * adherence[:, m]
            diet[:, m + 1] = target + (diet[:, m] - target) * self.decay
        drift = np.concatenate([np.zeros((len(kcal), 1)), np.cumsum(cal.weight_drift_sd * self.drift_steps, axis=1)],
                               axis=1)
        return diet + drift, pattern != 3, diet


def weight_change(people, energy, cal, adherence_mean, ev, months, seed=SEED):
    """Weight change (kg) of each person at months 0..months, completion, and the diet-driven part."""
    return Cohort(people, ev, months, seed, cal).run(energy, cal, adherence_mean)


def at(path, month):
    """Values at a (possibly fractional) month, by linear interpolation between monthly values."""
    low = int(np.floor(month))
    high = min(low + 1, path.shape[1] - 1)
    return path[:, low] + (month - low) * (path[:, high] - path[:, low])


def contact_phases(table=None):
    """Months of weekly contact at the start of each programme (RX-CONTACT-LA, RX-CONTACT-DPP); Vilar-Gomez
    gave routine clinical advice, so none."""
    table = load_table() if table is None else table
    return {'look_ahead': float(table.loc['RX-CONTACT-LA', 'value']),
            'dpp': float(table.loc['RX-CONTACT-DPP', 'value']) * 7 / 30.4375, 'vilar_gomez': 0.0}


def outcomes(cohorts, rx, cal, adherence, ev=None, months=12, seed=SEED, contact=None):
    """Simulated value of every target, keyed by target_id. cohorts: trial name to Cohort (built for
    at least `months`) or DataFrame. contact: each trial's intensive-contact phase (contact_phases())."""
    built = {t: c if isinstance(c, Cohort) else Cohort(c, ev, months, seed, cal) for t, c in cohorts.items()}
    contact = contact_phases() if contact is None else contact
    sim = {}
    runs = {'ILI': built['look_ahead'].run(rx['look_ahead'], cal, adherence['look_ahead'], contact['look_ahead']),
            'DSE': built['look_ahead'].run(rx['usual_care'], cal, 0.0),
            'DPP': built['dpp'].run(rx['dpp'], cal, adherence['dpp'], contact['dpp']),
            'VG': built['vilar_gomez'].run(rx['vilar_gomez'], cal, adherence['vilar_gomez'], contact['vilar_gomez'])}
    paths = {arm: run[0] for arm, run in runs.items()}
    completed = runs['VG'][1]
    ili_diet = runs['ILI'][2]
    weights = {'ILI': built['look_ahead'].weight, 'DSE': built['look_ahead'].weight, 'DPP': built['dpp'].weight,
               'VG': built['vilar_gomez'].weight}
    horizon = min(c.months for c in built.values())

    def pct(arm, month):
        return 100 * at(paths[arm], month) / weights[arm]
    for arm in ('ILI', 'DSE'):
        sim[f'LA-Y1-MEAN-{arm}'] = pct(arm, 12).mean()
        sim[f'LA-Y1-GE5-{arm}'] = 100 * (pct(arm, 12) <= -5).mean()
        sim[f'LA-Y1-GE10-{arm}'] = 100 * (pct(arm, 12) <= -10).mean()
        sim[f'LA-Y1-SD-{arm}'] = paths[arm][:, 12].std(ddof=1)
    sim['DPP-GOAL-24W'] = 100 * (pct('DPP', 24 * 7 / 30.4375) <= -7).mean()
    sim['VG-GE5-52W'] = 100 * (pct('VG', 12) <= -5).mean()
    sim['VG-PCT-52W'] = pct('VG', 12).mean()
    lost = -pct('VG', 12)
    for key, (low, high) in VG_GROUPS.items():
        sim[key] = 100 * ((lost >= low) & (lost < high)).mean()
    # Supplementary Table 2 reports completers only; its shape is used: each week's mean loss as a
    # share of the week-52 loss.
    course = {w: at(paths['VG'], w * 7 / 30.4375)[completed].mean() for w in VG_WEEKS}
    for weeks in VG_WEEKS:
        sim[f'VG-SHAPE-{weeks}W'] = course[weeks] / course[52]
    # Regain from each person's lowest point to month 12, diet-driven part only (Neiberg's curves are
    # fitted with random error removed), averaged over the third who regained most.
    diet_pct = 100 * ili_diet[:, :13] / weights['ILI'][:, None]
    regain = diet_pct[:, 12] - diet_pct.min(axis=1)
    sim['LA-Y1-REGAIN-T1'] = np.sort(regain)[-len(regain) // 3:].mean()
    if horizon >= 96:
        sim['LA-Y8-MEAN-ILI'] = pct('ILI', 96).mean()
        sim['LA-Y8-GE5-ILI'] = 100 * (pct('ILI', 96) <= -5).mean()
        sim['LA-Y8-GE10-ILI'] = 100 * (pct('ILI', 96) <= -10).mean()
    if horizon >= 34:
        sim['DPP-GOAL-LAST'] = 100 * (pct('DPP', 34) <= -7).mean()
        sim['DPP-MEAN-LIFESTYLE'] = -at(paths['DPP'], 34).mean()
    return sim


def targets(target_file=TARGET_FILE):
    """Observed value, unit and use of every target, with the combined Look AHEAD SD row split in two."""
    table = pd.read_csv(target_file)
    rows = {}
    for t in table.itertuples(index=False):
        if t.target_id == 'LA-Y1-KG-SD':
            for arm, sd in zip(('ILI', 'DSE'), (7.6, 5.0)):       # spread column: "7.6 (ILI); 5.0 (DSE) SD"
                assert f'{sd} ({arm})' in t.spread
                rows[f'LA-Y1-SD-{arm}'] = (sd, 'kg', t.use)
        elif t.target_id == 'DPP-MEAN':
            rows['DPP-MEAN-LIFESTYLE'] = (float(str(t.value).split()[0]), 'kg', t.use)
        elif t.use == 'shape':                       # Vilar-Gomez completers: share of the week-52 loss
            final = float(table.loc[table.target_id.eq('VG-KG-52W'), 'value'].iloc[0])
            rows[t.target_id.replace('-KG-', '-SHAPE-')] = (float(t.value) / final, 'fraction', 'calibrate')
        elif t.target_id != 'VJ-CAP-6M':
            rows[t.target_id] = (float(t.value), t.unit, t.use)
    return rows


# Tuned settings, each kept within plausible bounds. The regaining share is a fraction of what the
# cap leaves after the fixed dropout share; slipping is a fraction of the rest; steady takes the remainder.
BOUNDS = {'regaining_fraction': (0, 1), 'slipping_fraction': (0, 1), 'adherence_sd': (0, 0.4),
          'slip_months': (0.5, 24), 'slip_floor': (0, 0.9), 'weight_response_sd': (0, 1),
          'weight_drift_sd': (0.3, 3), **{f'adherence_{t}': (0.05, 1) for t in TRIALS}}


def unpack(x, base, dropout, quit_cap):
    v = dict(zip(BOUNDS, map(float, x)))
    regaining = (quit_cap - dropout) * v['regaining_fraction']
    slipping = (1 - dropout - regaining) * v['slipping_fraction']
    shares = {'steady': 1 - dropout - regaining - slipping, 'slipping': slipping, 'regaining': regaining,
              'dropping_out': dropout}
    cal = replace(base, pattern_shares=shares, adherence_sd=v['adherence_sd'], slip_months=v['slip_months'],
                  slip_floor=v['slip_floor'], weight_response_sd=v['weight_response_sd'],
                  weight_drift_sd=v['weight_drift_sd'])
    return cal, {t: v[f'adherence_{t}'] for t in TRIALS}


def pack(cal, adherence, dropout, quit_cap):
    s = cal.pattern_shares
    return np.array([s['regaining'] / (quit_cap - dropout), s['slipping'] / (1 - dropout - s['regaining']),
                     cal.adherence_sd, cal.slip_months, cal.slip_floor, cal.weight_response_sd,
                     cal.weight_drift_sd, *[adherence[t] for t in TRIALS]])


def loss(sim, goal):
    return sum(((sim[k] - value) / SCALE[unit]) ** 2 for k, (value, unit, use) in goal.items()
               if use == 'calibrate' and k in sim)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quit-cap', type=float, default=0.2,
                        help='largest share who stop the diet completely (regaining plus dropping out)')
    args = parser.parse_args()
    out = OUTPUT / f'quit_cap_{round(100 * args.quit_cap):02d}'
    out.mkdir(parents=True, exist_ok=True)
    table, ev = load_table(), Evidence.from_table()
    rx, people, goal = prescriptions(table), trial_cohorts(), targets()
    dropout = float(table.loc['CAL-DROPOUT', 'value']) / 100
    if not dropout < args.quit_cap <= 1:
        raise ValueError(f'--quit-cap must lie between the dropout share {dropout} and 1')
    base = Calibration()
    year = {t: Cohort(c, ev, 12, SEED, base) for t, c in people.items()}
    print({t: len(c) for t, c in people.items()}, flush=True)

    def objective(x):
        cal, adherence = unpack(x, base, dropout, args.quit_cap)
        return loss(outcomes(year, rx, cal, adherence), goal)

    # Differential evolution searches the whole bounded space; a short Nelder-Mead run then polishes.
    result = differential_evolution(objective, list(BOUNDS.values()), seed=SEED, popsize=12, maxiter=80,
                                    tol=1e-6, polish=False, init='sobol')
    polished = minimize(objective, result.x, method='Nelder-Mead', bounds=list(BOUNDS.values()),
                        options={'maxfev': 600, 'xatol': 1e-4, 'fatol': 1e-4})
    best = polished if polished.fun < result.fun else result
    cal, adherence = unpack(best.x, base, dropout, args.quit_cap)
    long = {t: Cohort(c, ev, 96, SEED, base) for t, c in people.items()}
    tuned = outcomes(long, rx, cal, adherence)
    rows = [{'target_id': k, 'use': use, 'unit': unit, 'published': value, 'after': tuned.get(k)}
            for k, (value, unit, use) in goal.items()]
    report = pd.DataFrame(rows)
    report.to_csv(out / 'targets.csv', index=False)
    settings = {'quit_cap': args.quit_cap, 'calibration': asdict(cal), 'adherence_by_trial': adherence,
                'loss': float(best.fun), 'evaluations': int(result.nfev + polished.nfev), 'seed': SEED,
                'cohort_sizes': {t: len(c) for t, c in people.items()},
                'cohort_summary': {t: {'age': round(c.RIDAGEYR.mean(), 1), 'bmi': round(c.BMXBMI.mean(), 1),
                                       'weight_kg': round(c.BMXWT.mean(), 1),
                                       'women_pct': round(100 * (c.RIAGENDR == 2).mean(), 1)}
                                   for t, c in people.items()}}
    (out / 'settings.json').write_text(json.dumps(settings, indent=2))
    print(json.dumps(settings, indent=2))
    print(report.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
