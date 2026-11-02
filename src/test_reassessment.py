from datetime import date, timedelta

import numpy as np
import pytest

from src.reassessment import PRIOR_MEAN, WeightRule, projected_inputs, reassess

RULE = WeightRule.from_tables()
START = date(2026, 1, 1)


def weights(k, months, start_kg=90.0, plan=-500, skip=()):
    """Weights a patient with response factor k would show, without drift."""
    return [(START + timedelta(days=round(30.4375 * m)),
             None if m in skip else round(start_kg + k * float(RULE.full_effect(plan, m)), 2)) for m in months]


def test_rule_follows_hall():
    assert RULE.kg_per_100kj == 1.0 and 12 < RULE.tau_months < 18
    assert RULE.drift_sd == pytest.approx(5.0 / np.sqrt(12))
    year = float(RULE.full_effect(-100 / 4.184, 12))     # 100 kJ/day for a year: about half of 1 kg
    assert -0.65 < year < -0.45


def test_a_patient_who_follows_the_plan_is_on_track_and_k_moves_towards_the_truth():
    out = reassess(weights(0.9, range(0, 13)), START.isoformat(), -500, RULE)
    assert out['visits'][0]['verdict'] == 'start'
    assert all(v['verdict'] == 'on track' for v in out['visits'][1:])
    assert PRIOR_MEAN < out['response']['mean'] < 0.9 and out['response']['sd'] < 0.35


def test_a_patient_who_regains_is_flagged_and_k_falls():
    path = weights(1.0, range(0, 4))
    path += [(START + timedelta(days=round(30.4375 * 4)), path[-1][1] + 6.0)]   # +6 kg in a month
    out = reassess(path, START.isoformat(), -500, RULE)
    assert out['visits'][-1]['verdict'] == 'slower than expected'
    assert out['visits'][-1]['k_mean'] < out['visits'][-2]['k_mean']


def test_missed_weights_are_skipped_and_projections_widen():
    out = reassess(weights(0.5, range(0, 7), skip={2, 3}), START.isoformat(), -500, RULE)
    assert len(out['visits']) == 5
    widths = [p['range_kg'][1] - p['range_kg'][0] for p in out['projection']]
    assert [p['months_ahead'] for p in out['projection']] == [3, 6, 12] and widths == sorted(widths)
    assert all(p['change_kg'] < 0 for p in out['projection'])
    assert reassess([], START.isoformat(), -500, RULE)['visits'] == []


def test_projected_inputs_move_weight_bmi_and_waist_together():
    out = projected_inputs({'BMXWT': 100.0, 'BMXHT': 200.0, 'BMXWAIST': 110.0, 'LBXSATSI': 40}, -10, RULE)
    assert out['BMXWT'] == 90 and out['BMXBMI'] == pytest.approx(22.5)
    assert out['BMXWAIST'] == pytest.approx(110 - 8.6) and out['LBXSATSI'] == 40
    assert np.isnan(projected_inputs({'BMXWT': float('nan'), 'BMXHT': 170.0}, -5, RULE)['BMXWT'])
