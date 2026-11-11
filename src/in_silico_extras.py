"""The remaining in-silico measures of Proposal section 12.3 (the main ones are in src/in_silico_evaluation.py).

1. Months to detect a change.
   - Weight: the first month a patient is truly behind the plan (less than half of the typical loss, on the
     true weight) against the first check-in at which the twin says "behind plan".
   - Liver: among patients who start with CAP >= 280 and whom the twin flags at sign-up (chance >= 0.5),
     the first month the true CAP falls below 280 against the first month the twin's chance, recomputed
     every month from the record it holds (weights monthly, bloods quarterly), falls below 0.5.
2. Whether forecast error falls as the twin learns: the 3-month-ahead weight forecast from check-ins up to
   month 1, 2, ... 9, against the textbook curve (no learning) and the last weight carried forward.
3. Robustness of the month-6 forecast of month 12: honest answers; 30% more missed visits; a scale that
   reads within about 1 kg (normal noise, SD 1 kg); answers inflated by under-reporting; no answers. And the
   month-12 liver chance when no follow-up bloods are taken (weights and waist only).
4. Whether scenarios rank in their true order: per patient, the twin's projected 12-month weight under
   usual diet, Scenario B and Scenario A at sign-up, and its month-6 forecasts of month 12 under A and B,
   against the true month-12 weight; and the month-6 liver forecasts under A and B against the true CAP.

Adults (18-59) for the weight results, as in the main evaluation.

    .venv/Scripts/python.exe -m src.in_silico_extras
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.in_silico_evaluation import (DATA_FILE, ID_COLUMNS, OUTPUT, SIM, behind_at, chances, forecasts, patient_records,
                                      run_loop, scenario_energy)
from src.reassessment import TYPICAL_ADHERENCE, Response, WeightRule, predict

SEED = 11
EXTRA_MISSED = 0.3
SCALE_SD = 1.0
LEARN_CUTOFFS = range(1, 10)


def mae(errors):
    return round(float(np.mean(np.abs(errors))), 2)


# ---------- 1. detection ----------
def weight_detection(records, energy, rule, adult):
    lags, missed, false_flags, n_true = [], 0, 0, 0
    for (seqn, scenario), rec in records.items():
        if scenario not in energy or energy[scenario] == 0 or not adult.get(seqn, False):
            continue
        true_onset = next((m for m in range(2, 13) if behind_at(rec, energy[scenario], rule, m)), None)
        out = run_loop(rec, energy[scenario], rule, 12, 0.0)
        if out is None:
            continue
        loop, seen = out
        flagged = next((m for (m, _), v in zip(seen[1:], loop['visits'][1:]) if v['progress'] == 'behind plan'), None)
        if true_onset is None:
            false_flags += flagged is not None
            continue
        n_true += 1
        if flagged is None:
            missed += 1
        else:
            lags.append(flagged - true_onset)
    lags = np.array(lags)
    return {'truly_behind': n_true, 'detected': int(len(lags)), 'never_detected': missed,
            'median_lag_months': float(np.median(lags)) if len(lags) else None,
            'share_detected_same_month': round(float((lags <= 0).mean()), 3) if len(lags) else None,
            'share_detected_within_1_month': round(float((lags <= 1).mean()), 3) if len(lags) else None,
            'flagged_but_never_truly_behind': false_flags}


def monthly_inputs(observed, baseline, columns, blank_bloods=False):
    """The record the twin holds each month: baseline examination updated by every observed value so far."""
    obs = observed.sort_values(['SEQN', 'scenario', 'month']).reset_index(drop=True)
    vals = obs[columns].copy()
    vals.loc[obs.month == 0, :] = np.nan                       # month 0 = the NHANES examination itself
    if blank_bloods:
        keep = {'BMXWT', 'BMXBMI', 'BMXWAIST', 'BMXHIP', 'BMXARMC', 'WHD050'}
        vals.loc[:, [c for c in columns if c not in keep]] = np.nan
    vals = vals.groupby([obs.SEQN, obs.scenario]).ffill()
    base = baseline.reindex(obs.SEQN).reset_index(drop=True)
    full = base.copy()
    full[columns] = vals.combine_first(base[columns])
    full['SEQN'], full['scenario'], full['month'] = obs.SEQN.to_numpy(), obs.scenario.to_numpy(), obs.month.to_numpy()
    return full


def liver_detection(monthly, truth):
    t = truth.set_index(['SEQN', 'scenario', 'month']).LUXCAPM
    m = monthly.set_index(['SEQN', 'scenario', 'month']).chance
    out = {}
    for scenario in ('scenario A', 'scenario B'):
        lags, missed, false_drop, n = [], 0, 0, 0
        for seqn in truth.SEQN.unique():
            tr, ch = t.loc[(seqn, scenario)], m.loc[(seqn, scenario)]
            if tr.loc[0] < 280 or ch.loc[0] < 0.5:
                continue
            n += 1
            true_cross = next((k for k in range(1, 13) if tr.loc[k] < 280), None)
            twin_cross = next((k for k in range(1, 13) if ch.loc[k] < 0.5), None)
            if true_cross is None:
                false_drop += twin_cross is not None
            elif twin_cross is None:
                missed += 1
            else:
                lags.append(twin_cross - true_cross)
        lags = np.array(lags)
        out[scenario] = {'start_with_a_lot_and_flagged': n, 'truly_fell_below_280': int(len(lags) + missed),
                         'detected': int(len(lags)), 'not_detected_by_month_12': missed,
                         'median_lag_months': float(np.median(lags)) if len(lags) else None,
                         'share_detected_within_1_month': round(float((lags <= 1).mean()), 3) if len(lags) else None,
                         'twin_dropped_but_truth_did_not': false_drop}
    return out


# ---------- 2. learning ----------
def learning_curve(records, energy, rule, adult):
    rows = []
    for (seqn, scenario), rec in records.items():
        if scenario not in energy or energy[scenario] == 0 or not adult.get(seqn, False) or not rec['weights'] or rec['weights'][0][0] != 0:
            continue
        start = rec['weights'][0][1]
        for c in LEARN_CUTOFFS:
            out = run_loop(rec, energy[scenario], rule, c, 0.0)
            if out is None or out[1][-1][0] != c:
                continue
            loop, seen = out
            twin = next(p for p in loop['projection'] if p['months_ahead'] == 3)
            book = predict(start, Response().at(TYPICAL_ADHERENCE), float(rule.full_effect(energy[scenario], c + 3)), c + 3, rule)[0]
            truth = rec['true_weight'][c + 3]
            rows.append({'cutoff': c, 'twin': twin['weight_kg'] - truth, 'textbook': book - truth, 'last': seen[-1][1] - truth,
                         'width': twin['range_kg'][1] - twin['range_kg'][0],
                         'covered': twin['range_kg'][0] <= truth <= twin['range_kg'][1]})
    f = pd.DataFrame(rows)
    return {f'month {c} -> {c + 3}': {'n': int(len(g)), 'twin_mae_kg': mae(g.twin), 'textbook_mae_kg': mae(g.textbook),
                                       'last_weight_mae_kg': mae(g['last']), 'twin_range_width_kg': round(float(g.width.mean()), 1),
                                       'twin_coverage_90': round(float(g.covered.mean()), 3)}
            for c, g in f.groupby('cutoff')}


# ---------- 3. robustness ----------
def perturbed(rec, rng, missed=0.0, scale_sd=0.0):
    weights = [(m, w + (rng.normal(0, scale_sd) if m else 0.0)) for m, w in rec['weights'] if m == 0 or rng.random() >= missed]
    return {**rec, 'weights': weights}


def robustness(records, energy, rule, adult, reporting, ages):
    rng = np.random.default_rng(SEED)
    conditions = {'honest answers': dict(share=lambda s: 0.0),
                  f'{int(EXTRA_MISSED * 100)}% more missed visits': dict(share=lambda s: 0.0, missed=EXTRA_MISSED),
                  f'scale noise SD {SCALE_SD:g} kg': dict(share=lambda s: 0.0, scale_sd=SCALE_SD),
                  'over-reported answers': dict(share=lambda s: float(reporting.get(s, 0))),
                  'no answers': dict(share=lambda s: None)}
    out = {}
    for label, cond in conditions.items():
        rows = []
        for (seqn, scenario), rec in records.items():
            if scenario not in energy or energy[scenario] == 0 or not adult.get(seqn, False):
                continue
            r = perturbed(rec, rng, cond.get('missed', 0.0), cond.get('scale_sd', 0.0))
            rows += [x for x in forecasts(r, energy[scenario], rule, cond['share'](seqn), float(ages.get(seqn, 30)))
                     if x['cutoff'] == 6 and x['ahead'] == 6]
        f = pd.DataFrame(rows)
        out[label] = {'n': int(len(f)), 'twin_mae_kg': mae(f.twin_kg - f.true_kg),
                      'twin_coverage_90': round(float(f.true_kg.between(f.twin_low, f.twin_high).mean()), 3)}
    return out


# ---------- 4. scenario order ----------
def scenario_order(records, energy, rule, adult, truth, liver_forecast):
    t12 = truth[truth.month == 12].pivot(index='SEQN', columns='scenario', values='BMXWT')
    t0 = truth[truth.month == 0].groupby('SEQN').BMXWT.first()
    keep = [s for s in t12.index if adult.get(s, False) and np.isfinite(t0.get(s, np.nan))]
    t12 = t12.loc[keep]
    true_order = (t12['scenario A'] < t12['scenario B']) & (t12['scenario B'] < t12['usual diet'])
    a_vs_b = t12['scenario A'] < t12['scenario B']
    # the twin's sign-up projection orders the scenarios by their energy change for everyone
    book = {s: np.array([predict(float(w), Response().at(TYPICAL_ADHERENCE), float(rule.full_effect(e, 12)), 12, rule)[0]
                         for w in t0.loc[keep]]) for s, e in energy.items()}
    twin_order = bool(np.all(book['scenario A'] < book['scenario B'])) and bool(np.all(book['scenario B'] < book['usual diet']))
    month6 = {}
    for (seqn, scenario), rec in records.items():
        if scenario in ('scenario A', 'scenario B') and seqn in t12.index:
            out = run_loop(rec, energy[scenario], rule, 6, 0.0)
            if out is not None and out[1][-1][0] == 6:
                month6[(seqn, scenario)] = next(p for p in out[0]['projection'] if p['months_ahead'] == 6)['weight_kg']
    pairs = [(s, month6[(s, 'scenario A')] < month6[(s, 'scenario B')], a_vs_b.loc[s]) for s in t12.index
             if (s, 'scenario A') in month6 and (s, 'scenario B') in month6]
    lf = liver_forecast.pivot(index='SEQN', columns='scenario', values=['chance', 'true_cap']).dropna()
    differs = lf[('true_cap', 'scenario A')] != lf[('true_cap', 'scenario B')]
    lf = lf[differs]
    liver_agree = ((lf[('chance', 'scenario A')] < lf[('chance', 'scenario B')]) ==
                   (lf[('true_cap', 'scenario A')] < lf[('true_cap', 'scenario B')]))
    return {'patients': int(len(t12)),
            'cohort_mean_true_weight_month12': {s: round(float(t12[s].mean()), 2) for s in t12.columns},
            'twin_signup_projection_orders_A_B_usual_for_everyone': twin_order,
            'share_truly_ordered_A_B_usual': round(float(true_order.mean()), 3),
            'share_truly_A_below_B': round(float(a_vs_b.mean()), 3),
            'month6_forecast_A_vs_B_matches_truth': {'n': len(pairs), 'share': round(float(np.mean([p == q for _, p, q in pairs])), 3)},
            'month6_liver_forecast_A_vs_B_matches_truth': {'n': int(len(lf)), 'share': round(float(liver_agree.mean()), 3)}}


def main():
    truth = pd.read_csv(SIM / 'truth.csv.gz')
    observed = pd.read_csv(SIM / 'observed.csv.gz')
    reporting = pd.read_csv(SIM / 'reporting.csv').set_index('SEQN').energy_underreport
    data = pd.read_csv(DATA_FILE)
    ages = data.set_index('SEQN').RIDAGEYR
    adult = (ages >= 18).to_dict()
    energy, rule = scenario_energy(), WeightRule.from_tables()
    records = {(r['SEQN'], r['scenario']): r for r in patient_records(truth, observed)}

    measured = [c for c in data.columns if c not in ID_COLUMNS]
    columns = [c for c in observed.columns if c in measured and c != 'LUXCAPM']
    baseline = data.set_index('SEQN')[measured]
    monthly = monthly_inputs(observed, baseline, columns)
    monthly['chance'] = chances(monthly[measured])
    no_bloods = monthly_inputs(observed, baseline, columns, blank_bloods=True)
    no_bloods = no_bloods[no_bloods.month == 12]
    no_bloods['chance'] = chances(no_bloods[measured])
    true12 = truth[truth.month == 12].set_index(['SEQN', 'scenario']).LUXCAPM
    def auc12(frame, scenario):
        f = frame[(frame.month == 12) & (frame.scenario == scenario)].set_index(['SEQN', 'scenario'])
        y = (true12.reindex(f.index) >= 280).astype(int)
        return round(float(roc_auc_score(y, f.chance)), 3)

    summary = {
        'detection': {'weight_behind_plan': weight_detection(records, energy, rule, adult),
                      'liver_below_cap280': liver_detection(monthly, truth)},
        'learning_curve_3_months_ahead': learning_curve(records, energy, rule, adult),
        'robustness_month6_to_12': robustness(records, energy, rule, adult, reporting, ages),
        'robustness_liver_month12_auc': {s: {'with follow-up bloods': auc12(monthly, s), 'weights and waist only': auc12(no_bloods, s)}
                                         for s in ('scenario A', 'scenario B')},
        'scenario_order': scenario_order(records, energy, rule, adult, truth,
                                         pd.read_csv(OUTPUT / 'liver_forecast_month12.csv')),
    }
    monthly[['SEQN', 'scenario', 'month', 'chance']].to_csv(OUTPUT / 'liver_chance_monthly.csv', index=False)
    (OUTPUT / 'extras.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
