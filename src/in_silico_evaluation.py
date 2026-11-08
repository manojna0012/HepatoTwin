"""In-silico evaluation of the twin's follow-up loop on the calibrated virtual cohort (Proposal 17.14).

The simulator (src/simulation/) knows each virtual patient's truth; the observation layer turns it into what a
clinic records (monthly weights with missed visits, quarterly bloods, CAP at months 0, 6 and 12). The twin sees
only the observed record and the patient's monthly answer to "how many planned meals did you cook?", and is
graded against the truth on four questions:

1. Weight forecasts. From the check-ins up to month 3 or 6, the twin (src/reassessment.py, which learns each
   patient's response) forecasts the weight 3 or 6 months later. Baselines: the textbook curve from the start
   with no learning (typical adherence and response), and the last observed weight carried forward. Error is
   the absolute difference from the true weight; coverage is the share of true weights inside the 90% range.
2. Causes of slow progress. Each check-in the twin calls "behind plan" gets a cause from the patient's answer:
   "plan not followed" (reported adherence below 0.4) or "no progress despite effort" (0.6 or above). The
   truth is the simulator's adherence that month and the patient's body response (weight_response x
   eventual_scale, 1 = textbook).
3. Clinician alerts. A patient is alerted after two "no progress despite effort" check-ins in a row. The
   alert claims only what is observed, so it is graded on (a) the true reason behind each alert: intake
   higher than reported (true adherence over the two months below 0.6), a slow body (response in the
   slowest fifth of the cohort), or neither (ordinary weight drift); and (b) whether the lack of progress
   lasts: the share of alerted patients still behind plan on their true weight at month 12, against all
   patients. A first version graded the alert as a detector of slow bodies; it barely beat chance (precision
   0.26 against 0.20), because the learned response cannot be told apart from drift within a year, which is
   why the alert and its message were changed (src/api/care.py).
4. Liver fat. Layer 2 runs on the record carried forward at months 0, 6 and 12 and its chance of CAP >= 280
   is compared with the true CAP (AUC, Brier score, mean chance against the true share). At month 6 the twin
   also forecasts month 12 (weight forecast fed through projected_inputs) against the true CAP at month 12.

Answers. The simulator does not produce questionnaire answers, so they are derived from the true adherence of
that month (most >= 0.7, about half >= 0.35, else few). Honest answers are the main analysis; because people
over-report diet effort as they under-report intake, a second run inflates each answer by the person's own
energy under-reporting share from the observation layer (reported = true / (1 - share)). Without answers the
twin falls back to typical adherence.

Ages 12-59 are in scope; teenagers grow during follow-up, which the twin's weight rule does not model, so the
weight results are given for adults (18-59) and for everyone.

    .venv/Scripts/python.exe -m src.in_silico_evaluation
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.care_plan import MEAL_ANSWERS
from src.predict_cap import MODEL_FILE, predict_steatosis
from src.reassessment import (NO_PROGRESS, TYPICAL_ADHERENCE, Response, WeightRule, is_behind, predict,
                              projected_inputs, reassess)

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / 'data' / 'results' / 'simulation'
DATA_FILE = ROOT / 'data' / 'processed' / 'nhanes_combined.csv'
OUTPUT = ROOT / 'data' / 'results' / 'in_silico'
START = date(2026, 1, 1)
CUTOFFS = ((3, 3), (6, 3), (6, 6))           # (months of check-ins seen, months ahead)
LIVER_MONTHS = (0, 6, 12)
SLOW_FIFTH = 0.2
ID_COLUMNS = {'SEQN', 'cycle'}


def visit_date(month):
    return START + timedelta(days=round(30.4375 * month))


def answer(adherence):
    return 'most' if adherence >= 0.7 else 'half' if adherence >= 0.35 else 'few'


def reported(adherence, share=0.0):
    """The meals answer as a share of the plan (care_plan.MEAL_ANSWERS), optionally inflated by over-reporting."""
    return MEAL_ANSWERS[answer(min(1.0, adherence / (1 - share)))]


def scenario_energy():
    info = json.loads((SIM / 'run_info.json').read_text())
    return {s['name']: float(s['energy_change_kcal']) for s in info['scenarios']}


def patient_records(truth, observed):
    """One record per patient and diet scenario: observed weights and true adherence and weights by month."""
    obs = observed.set_index(['SEQN', 'scenario', 'month'])
    for (seqn, scenario), t in truth.groupby(['SEQN', 'scenario']):
        t = t.set_index('month').sort_index()
        o = obs.loc[(seqn, scenario)].sort_index()
        weights = [(m, float(o.loc[m, 'BMXWT'])) for m in o.index if o.loc[m, 'attended'] and pd.notna(o.loc[m, 'BMXWT'])]
        yield {'SEQN': seqn, 'scenario': scenario, 'weights': weights, 'adherence': t.adherence.to_dict(),
               'true_weight': t.BMXWT.to_dict()}


def run_loop(rec, energy, rule, upto=12, share=None):
    """The twin on the observed weights up to a month; share None = no answers, else answers inflated by share."""
    seen = [(m, w) for m, w in rec['weights'] if m <= upto]
    if not seen or seen[0][0] != 0:
        return None
    adherence = None if share is None else [None] + [reported(rec['adherence'][m], share) for m, _ in seen[1:]]
    return reassess([(visit_date(m), w) for m, w in seen], START.isoformat(), energy, rule, adherence), seen


def forecasts(rec, energy, rule, share, age):
    rows = []
    start = rec['weights'][0][1] if rec['weights'] and rec['weights'][0][0] == 0 else None
    for cutoff, ahead in CUTOFFS:
        target = cutoff + ahead
        out = run_loop(rec, energy, rule, cutoff, share)
        if out is None or start is None or out[1][-1][0] < cutoff - 1:      # need a weight in the last 2 months
            continue
        loop, seen = out
        twin = next(p for p in loop['projection'] if p['months_ahead'] == ahead)
        # the projection starts from the last weight seen; if that was a month early, it runs one month more
        if seen[-1][0] < cutoff:
            continue
        _, low, high = predict(start, Response().at(TYPICAL_ADHERENCE), float(rule.full_effect(energy, target)), target, rule)
        textbook = predict(start, Response().at(TYPICAL_ADHERENCE), float(rule.full_effect(energy, target)), target, rule)[0]
        truth = rec['true_weight'][target]
        rows.append({'SEQN': rec['SEQN'], 'scenario': rec['scenario'], 'adult': age >= 18, 'cutoff': cutoff, 'ahead': ahead,
                     'true_kg': truth, 'twin_kg': twin['weight_kg'], 'twin_low': twin['range_kg'][0], 'twin_high': twin['range_kg'][1],
                     'textbook_kg': textbook, 'textbook_low': low, 'textbook_high': high, 'last_kg': seen[-1][1]})
    return rows


def verdicts(rec, energy, rule, share):
    out = run_loop(rec, energy, rule, 12, share)
    if out is None:
        return []
    loop, seen = out
    return [{'SEQN': rec['SEQN'], 'scenario': rec['scenario'], 'month': m, 'progress': v['progress'], 'cause': v['cause'],
             'reported': v['adherence'], 'true_adherence': rec['adherence'][m]}
            for (m, _), v in zip(seen[1:], loop['visits'][1:])]


def alert_rows(rows):
    """The first pair of 'no progress despite effort' check-ins in a row (care.alerts 'behind_twice'), or None."""
    rows = sorted(rows, key=lambda r: r['month'])
    return next(((a, b) for a, b in zip(rows, rows[1:]) if a['cause'] == b['cause'] == NO_PROGRESS), None)


def alerted(rows):
    return alert_rows(rows) is not None


def alert_reason(pair, response, slow_cut):
    """The true reason behind an alert: intake higher than reported, a slow body, or ordinary drift."""
    if np.mean([r['true_adherence'] for r in pair]) < 0.6:
        return 'intake higher than reported'
    return 'slow body' if response <= slow_cut else 'ordinary weight drift'


def behind_at(rec, energy, rule, month):
    """Whether the true weight is behind the plan at a month (reassessment.is_behind on the truth)."""
    start = rec['true_weight'][0]
    typical, _, high = predict(start, Response().at(TYPICAL_ADHERENCE), float(rule.full_effect(energy, month)), month, rule)
    return bool(is_behind(start, rec['true_weight'][month], typical, high, month))


def forecast_summary(frame):
    out = {}
    for (cutoff, ahead), g in frame.groupby(['cutoff', 'ahead']):
        key = f'month {cutoff} -> month {cutoff + ahead}'
        out[key] = {'n': int(len(g)),
                    'twin_mae_kg': round(float((g.twin_kg - g.true_kg).abs().mean()), 2),
                    'textbook_mae_kg': round(float((g.textbook_kg - g.true_kg).abs().mean()), 2),
                    'last_weight_mae_kg': round(float((g.last_kg - g.true_kg).abs().mean()), 2),
                    'twin_coverage_90': round(float(g.true_kg.between(g.twin_low, g.twin_high).mean()), 3),
                    'textbook_coverage_90': round(float(g.true_kg.between(g.textbook_low, g.textbook_high).mean()), 3),
                    'twin_range_width_kg': round(float((g.twin_high - g.twin_low).mean()), 1)}
    return out


def cause_summary(frame, traits, slow_cut):
    behind = frame[frame.progress == 'behind plan'].merge(traits[['SEQN', 'response']], on='SEQN')
    out = {'checkins': int(len(frame)), 'behind_plan_share': round(float((frame.progress == 'behind plan').mean()), 3)}
    for cause, g in behind.groupby(behind.cause.fillna('partly followed')):
        out[cause] = {'n': int(len(g)), 'true_adherence_below_0_4': round(float((g.true_adherence < 0.4).mean()), 3),
                      'true_adherence_0_6_or_more': round(float((g.true_adherence >= 0.6).mean()), 3),
                      'body_response_below_textbook': round(float((g.response < 1).mean()), 3),
                      'in_slowest_fifth': round(float((g.response <= slow_cut).mean()), 3)}
    return out


def alert_summary(frame, traits, slow_cut, records, energy, rule):
    response = traits.set_index('SEQN').response
    per = []
    for (seqn, scenario), g in frame.groupby(['SEQN', 'scenario']):
        pair, rec = alert_rows(g.to_dict('records')), records[(seqn, scenario)]
        per.append({'alerted': pair is not None, 'slow': response[seqn] <= slow_cut,
                    'reason': alert_reason(pair, response[seqn], slow_cut) if pair else None,
                    'behind_month_12': behind_at(rec, energy[scenario], rule, 12)})
    per = pd.DataFrame(per)
    hit = per[per.alerted]
    tp, fn = int((hit.slow).sum()), int((~per.alerted & per.slow).sum())
    return {'patients': int(len(per)), 'alerted': int(len(hit)),
            'reasons_behind_alerts': {k: round(float(v), 3) for k, v in hit.reason.value_counts(normalize=True).items()},
            'still_behind_at_month_12': {'alerted': round(float(hit.behind_month_12.mean()), 3) if len(hit) else None,
                                         'all_patients': round(float(per.behind_month_12.mean()), 3)},
            'as_a_slow_body_detector': {'precision': round(tp / len(hit), 3) if len(hit) else None,
                                        'recall': round(tp / (tp + fn), 3) if tp + fn else None,
                                        'base_rate': round(float(per.slow.mean()), 3)}}


def carried_inputs(baseline, observed, seqn, scenario, month, columns):
    """The record the twin would hold at a month: the NHANES examination updated with every later observed value."""
    inputs = dict(baseline)
    rows = observed[(observed.SEQN == seqn) & (observed.scenario == scenario) & observed.month.between(1, month)].sort_values('month')
    for _, r in rows.iterrows():
        inputs.update({c: float(r[c]) for c in columns if pd.notna(r[c])})
    return inputs


def chances(frame):
    """Layer 2's chance of CAP >= 280 for many records at once (the same models as twin_pipeline.layer2)."""
    return predict_steatosis(frame, Path(MODEL_FILE).parent / 'steatosis_cap280.joblib').to_numpy()


def liver_evaluation(truth, observed, energy, rule, share, records_by_key):
    data = pd.read_csv(DATA_FILE)
    measured = [c for c in data.columns if c not in ID_COLUMNS]
    base = data.set_index('SEQN')
    columns = [c for c in observed.columns if c in measured and c != 'LUXCAPM']
    rows, projected = [], []
    for (seqn, scenario), _ in truth.groupby(['SEQN', 'scenario']):
        baseline = base.loc[seqn, measured].to_dict()
        t = truth[(truth.SEQN == seqn) & (truth.scenario == scenario)].set_index('month')
        for m in LIVER_MONTHS:
            rows.append({'SEQN': seqn, 'scenario': scenario, 'month': m, 'true_cap': float(t.loc[m, 'LUXCAPM']),
                         **carried_inputs(baseline, observed, seqn, scenario, m, columns)})
        rec = records_by_key[(seqn, scenario)]
        out = run_loop(rec, energy[scenario], rule, 6, share)
        if out and out[1][-1][0] == 6:
            change = next(p for p in out[0]['projection'] if p['months_ahead'] == 6)['change_kg']
            inputs6 = carried_inputs(baseline, observed, seqn, scenario, 6, columns)
            projected.append({'SEQN': seqn, 'scenario': scenario, 'true_cap': float(t.loc[12, 'LUXCAPM']),
                              **projected_inputs(inputs6, change, rule)})
    liver = pd.DataFrame(rows)
    liver['chance'] = chances(liver)
    proj = pd.DataFrame(projected)
    proj['chance'] = chances(proj)
    return liver[['SEQN', 'scenario', 'month', 'true_cap', 'chance']], proj[['SEQN', 'scenario', 'true_cap', 'chance']]


def liver_summary(liver, proj):
    def score(g):
        y = (g.true_cap >= 280).astype(int)
        return {'n': int(len(g)), 'true_share': round(float(y.mean()), 3), 'mean_chance': round(float(g.chance.mean()), 3),
                'auc': round(float(roc_auc_score(y, g.chance)), 3) if y.nunique() == 2 else None,
                'brier': round(float(brier_score_loss(y, g.chance)), 3)}
    out = {f'{s}, month {m}': score(g) for (s, m), g in liver.groupby(['scenario', 'month'])}
    month0 = liver[liver.month == 0].set_index(['SEQN', 'scenario']).chance
    for s, g in proj.groupby('scenario'):
        out[f'{s}, forecast at month 6 of month 12'] = score(g)
        same = g.set_index(['SEQN', 'scenario']).true_cap.to_frame().join(month0.rename('chance'))
        out[f'{s}, month-0 chance reused for month 12'] = score(same)
    pairs = liver.pivot_table(index=['SEQN', 'scenario'], columns='month', values=['true_cap', 'chance'])
    d_true = pairs[('true_cap', 12)] - pairs[('true_cap', 0)]
    d_chance = pairs[('chance', 12)] - pairs[('chance', 0)]
    out['change_month0_to_12_spearman'] = round(float(d_true.corr(d_chance, method='spearman')), 3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--limit', type=int, default=None, help='first N patients only (for a quick run)')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(SIM / 'truth.csv.gz')
    observed = pd.read_csv(SIM / 'observed.csv.gz')
    traits = pd.read_csv(SIM / 'traits.csv')
    reporting = pd.read_csv(SIM / 'reporting.csv').set_index('SEQN').energy_underreport
    ages = pd.read_csv(DATA_FILE, usecols=['SEQN', 'RIDAGEYR']).set_index('SEQN').RIDAGEYR
    if args.limit:
        keep = truth.SEQN.drop_duplicates().head(args.limit)
        truth, observed, traits = truth[truth.SEQN.isin(keep)], observed[observed.SEQN.isin(keep)], traits[traits.SEQN.isin(keep)]
    traits = traits.assign(response=traits.weight_response * traits.eventual_scale)
    slow_cut = float(traits.response.quantile(SLOW_FIFTH))
    energy, rule = scenario_energy(), WeightRule.from_tables()
    plans = [s for s, e in energy.items() if e != 0]
    records = {(r['SEQN'], r['scenario']): r for r in patient_records(truth, observed)}

    summary = {'patients': int(truth.SEQN.nunique()), 'scenarios': {s: energy[s] for s in plans}}
    for label, share_of in (('honest answers', lambda s: 0.0), ('over-reported answers', lambda s: float(reporting.get(s, 0))),
                            ('no answers', lambda s: None)):
        fc, vd = [], []
        for (seqn, scenario), rec in records.items():
            if scenario not in plans:
                continue
            share = share_of(seqn)
            fc += forecasts(rec, energy[scenario], rule, share, float(ages.get(seqn, 30)))
            if share is not None:
                vd += verdicts(rec, energy[scenario], rule, share)
        fc = pd.DataFrame(fc)
        fc.to_csv(OUTPUT / f'forecasts_{label.replace(" ", "_")}.csv', index=False)
        part = {'forecasts_adults': forecast_summary(fc[fc.adult]), 'forecasts_all_ages': forecast_summary(fc)}
        if vd:
            vd = pd.DataFrame(vd)
            vd.to_csv(OUTPUT / f'verdicts_{label.replace(" ", "_")}.csv', index=False)
            part['causes'] = cause_summary(vd, traits, slow_cut)
            part['alerts'] = alert_summary(vd, traits, slow_cut, records, energy, rule)
        summary[label] = part
        print(label, json.dumps(part, indent=1), flush=True)

    liver, proj = liver_evaluation(truth, observed, energy, rule, 0.0, records)
    liver.to_csv(OUTPUT / 'liver_by_month.csv', index=False)
    proj.to_csv(OUTPUT / 'liver_forecast_month12.csv', index=False)
    summary['liver'] = liver_summary(liver, proj)
    print('liver', json.dumps(summary['liver'], indent=1))
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
