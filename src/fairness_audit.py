"""Fairness audit (Proposal section 17.8): Layer 2 and Layer 4 results by sex, race and ethnicity, and age.

Layer 2: on the held-out 2017-2023 test participants inside the model's scope (ages 12-59, reliable
reference exam), the calibrated chances of CAP >= 248 and >= 280 are scored per group: AUC, Brier score,
mean chance against the observed share (calibration in the large) and the share flagged at a chance of 0.5,
with 95% percentile intervals from 1,000 bootstrap resamples of participants within the group.

Layer 4: for the 300 held-out patients of the ranking evaluation (src/evaluate_ranking.py), the share of the
proposed ranking's top-10 recipes meeting the health, cuisine and feasibility criteria, and all three, per
group (bootstrap over patients). Cuisine preferences there are simulated, as in the main evaluation.

Groups with fewer than 30 participants, or without both outcomes for an AUC, are reported without that
metric. The largest gap between groups is given for each metric. Unweighted.

    .venv/Scripts/python.exe -m src.fairness_audit
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.train_combined_model import in_scope
from src.train_final_model import ROOT

DATA = ROOT / 'data' / 'processed' / 'nhanes_combined.csv'
PREDICTIONS = ROOT / 'data' / 'results' / 'combined' / 'test_predictions.csv'
RANKING = ROOT / 'data' / 'results' / 'layer4' / 'method_comparison_by_patient.csv'
OUTPUT = ROOT / 'data' / 'results' / 'fairness'
PROPOSED = 'Multi-criteria, health-gated (proposed)'
BOOT = 1000
SEED = 42
MIN_N = 30
SEX = {1: 'men', 2: 'women'}
RACE = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black',
        6: 'Non-Hispanic Asian', 7: 'Other or multiracial'}
AGES = {'12-17': (12, 18), '18-39': (18, 40), '40-59': (40, 60)}


def groups(d):
    """{(dimension, group name): boolean mask}."""
    out = {('sex', name): d.RIAGENDR.eq(code) for code, name in SEX.items()}
    out |= {('race and ethnicity', name): d.RIDRETH3.eq(code) for code, name in RACE.items()}
    out |= {('age', name): d.RIDAGEYR.between(lo, hi - 1) for name, (lo, hi) in AGES.items()}
    return out


def interval(values):
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [round(float(low), 3), round(float(high), 3)]


def layer2_metrics(y, p, rng, boot=BOOT):
    """AUC, Brier, mean chance vs share and flagged share, each with a bootstrap interval."""
    y, p = np.asarray(y), np.asarray(p)
    def stats(yy, pp):
        auc = roc_auc_score(yy, pp) if yy.min() != yy.max() else np.nan
        return auc, brier_score_loss(yy, pp), pp.mean() - yy.mean(), (pp >= 0.5).mean()
    point = stats(y, p)
    draws = np.array([stats(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(boot))])
    names = ('auc', 'brier', 'mean_chance_minus_share', 'flagged_share')
    out = {'n': int(len(y)), 'share': round(float(y.mean()), 3), 'mean_chance': round(float(p.mean()), 3)}
    for k, name in enumerate(names):
        if name == 'auc' and np.isnan(point[0]):
            continue
        out[name] = {'value': round(float(point[k]), 3), 'ci95': interval(draws[:, k])}
    return out


def layer2(d, rng):
    out = {}
    for cutoff in (248, 280):
        y, p = (d.LUXCAPM >= cutoff).astype(int), d[f'p_cap{cutoff}']
        rows = {'all in scope': layer2_metrics(y, p, rng)}
        for (dim, name), mask in groups(d).items():
            if mask.sum() >= MIN_N:
                rows[f'{dim}: {name}'] = layer2_metrics(y[mask], p[mask], rng)
        out[f'CAP >= {cutoff}'] = rows
    return out


def layer4(r, rng, boot=BOOT):
    criteria = ('health', 'cuisine', 'feasibility', 'all_three')
    def summarise(g):
        values = g[list(criteria)].to_numpy()
        draws = np.array([values[rng.integers(0, len(values), len(values))].mean(axis=0) for _ in range(boot)])
        return {'n': int(len(g)), **{c: {'value': round(float(values[:, k].mean()), 3), 'ci95': interval(draws[:, k])}
                                     for k, c in enumerate(criteria)}}
    out = {'all patients': summarise(r)}
    for (dim, name), mask in groups(r).items():
        if mask.sum() >= MIN_N:
            out[f'{dim}: {name}'] = summarise(r[mask])
    return out


def gaps(table, metric):
    """Largest difference in a metric between groups of the same dimension."""
    out = {}
    for dim in ('sex', 'race and ethnicity', 'age'):
        values = {k: v[metric]['value'] for k, v in table.items() if k.startswith(dim + ':') and metric in v}
        if len(values) > 1:
            hi, lo = max(values, key=values.get), min(values, key=values.get)
            out[dim] = {'gap': round(values[hi] - values[lo], 3), 'highest': hi.split(': ')[1], 'lowest': lo.split(': ')[1]}
    return out


def main():
    rng = np.random.default_rng(SEED)
    data = pd.read_csv(DATA, usecols=['SEQN', 'LUXCAPM', 'LUXCPIQR', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3'])
    test = pd.read_csv(PREDICTIONS).merge(data, on='SEQN', how='left')
    test = test[in_scope(test)].reset_index(drop=True)
    ranking = pd.read_csv(RANKING)
    ranking = ranking[ranking.method == PROPOSED].merge(data, on='SEQN', how='left')

    l2, l4 = layer2(test, rng), layer4(ranking, rng)
    summary = {'layer2': l2, 'layer2_largest_gaps': {c: {m: gaps(t, m) for m in ('auc', 'brier', 'mean_chance_minus_share')}
                                                     for c, t in l2.items()},
               'layer4': l4, 'layer4_largest_gaps': {m: gaps(l4, m) for m in ('health', 'cuisine', 'feasibility', 'all_three')}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'fairness.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    rows = [{'layer': 'Layer 2', 'target': c, 'group': g, 'n': v['n'], 'share': v['share'], 'mean_chance': v['mean_chance'],
             **{f'{m}': v[m]['value'] for m in ('auc', 'brier') if m in v},
             **{f'{m}_ci95': v[m]['ci95'] for m in ('auc', 'brier') if m in v}}
            for c, t in l2.items() for g, v in t.items()]
    rows += [{'layer': 'Layer 4', 'target': 'top-10 recipes', 'group': g, 'n': v['n'],
              **{c: v[c]['value'] for c in ('health', 'cuisine', 'feasibility', 'all_three')},
              'all_three_ci95': v['all_three']['ci95']} for g, v in l4.items()]
    pd.DataFrame(rows).to_csv(OUTPUT / 'fairness_by_group.csv', index=False)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
