import numpy as np
import pandas as pd
import pytest

from src.simulation.observe import (BLOODS, LAB_CVI, ObservationDesign, ObservationEvidence, cap_error_sd,
                                    multiplicative, observe_cohort, person_reporting)
from src.simulation.parameters import Calibration
from src.simulation.simulator import MONTHS, simulate_cohort
from src.simulation.test_simulation import CAL, EV, cohort, patient

OEV = ObservationEvidence.from_table()
NEVER_MISS = ObservationDesign(miss_probability=0, miss_probability_gaining=0)
SCENARIOS = [{'name': 'usual', 'energy_change_kcal': 0}, {'name': 'A', 'energy_change_kcal': -500}]


def observed(rows=None, cal=CAL, design=NEVER_MISS, seed=1):
    rows = cohort(10) if rows is None else rows
    truth = simulate_cohort(rows, SCENARIOS, seed=seed, ev=EV, cal=cal)[0]
    return truth, *observe_cohort(truth, rows, seed=seed, oev=OEV, design=design)


def test_total_cv_combines_biological_and_analytical_variation():
    assert OEV.total_cv['LBXSATSI'] == pytest.approx(12.6 * np.sqrt(1.25))
    assert OEV.total_cv['LBXGLU'] == OEV.total_cv['LBXSGL']
    z = np.random.default_rng(0).standard_normal(200_000)
    noisy = multiplicative(100.0, 20.0, z)
    assert noisy.mean() == pytest.approx(100, rel=0.01) and noisy.std() == pytest.approx(20, rel=0.02)


def test_cap_error_gives_the_published_operator_agreement():
    rng = np.random.default_rng(0)
    true_cap = rng.normal(255, 60, 100_000)
    sd = cap_error_sd(true_cap, 0.82)
    first, second = (true_cap + sd * rng.standard_normal(true_cap.size) for _ in range(2))
    concordance = 2 * np.cov(first, second)[0, 1] / (first.var() + second.var() + (first.mean() - second.mean()) ** 2)
    assert concordance == pytest.approx(0.82, abs=0.01)


def test_month_zero_is_the_real_examination_and_the_schedule_is_kept():
    rows = cohort(10)
    _, obs, _, _ = observed(rows)
    zero = obs[obs.month.eq(0)].drop_duplicates('SEQN').set_index('SEQN')
    base = rows.set_index('SEQN')
    for col in ('BMXWT', 'LBXSATSI', 'LUXCAPM', 'DR1TKCAL'):
        assert np.allclose(zero[col], base.loc[zero.index, col])
    assert obs.BMXWT.notna().all()                                   # weight every month
    assert set(obs.loc[obs.LBXSATSI.notna(), 'month']) == {0, 3, 6, 9, 12}
    assert set(obs.loc[obs.LUXCAPM.notna(), 'month']) == {0, 6, 12}


def test_lab_noise_is_shared_across_scenarios_and_unbiased():
    truth, obs, _, _ = observed()
    later = obs[obs.month.eq(6)]
    usual, a = (later[later.scenario.eq(s)].set_index('SEQN') for s in ('usual', 'A'))
    t = truth[truth.month.eq(6)]
    t_usual, t_a = (t[t.scenario.eq(s)].set_index('SEQN') for s in ('usual', 'A'))
    # Same draw for one person: the ratio of observed to true is identical in both scenarios.
    assert np.allclose(usual.LBXIN / t_usual.LBXIN, a.LBXIN / t_a.LBXIN)
    assert not np.allclose(usual.LBXIN, t_usual.LBXIN)


def test_food_logs_report_a_smaller_cut_than_was_made():
    rows = pd.DataFrame([patient()])
    steady = Calibration(pattern_shares={'steady': 1, 'slipping': 0, 'regaining': 0, 'dropping_out': 0},
                         adherence_sd=0)
    truth, obs, reporting, _ = observed(rows, cal=steady)
    share = reporting.energy_underreport.iloc[0]
    assert 0.12 <= share <= 0.14                                     # a man, OPEN study range
    usual = obs[obs.scenario.eq('usual')]
    assert np.allclose(usual.DR1TKCAL, 2400)                         # no change: the log matches the recall
    a = obs[obs.scenario.eq('A') & obs.month.eq(12)].iloc[0]
    true_cut = 500 * 0.7
    assert 2400 - a.DR1TKCAL == pytest.approx(true_cut * (1 - share))
    assert person_reporting(1001, 2, 1, OEV)['energy'] >= 0.16       # women under-report more


def test_dropouts_report_nothing_and_missed_visits_follow_weight_gain():
    rows = cohort(10)
    leaving = Calibration(pattern_shares={'steady': 0, 'slipping': 0, 'regaining': 0, 'dropping_out': 1})
    truth, obs, _, _ = observed(rows, cal=leaving)
    gone = truth.dropped_out.to_numpy()
    assert not obs.attended.to_numpy()[gone].any()
    assert obs.loc[~obs.attended, ['BMXWT', 'DR1TKCAL'] + list(BLOODS)].isna().all().all()
    always_miss_if_gaining = ObservationDesign(miss_probability=0, miss_probability_gaining=1)
    regain = Calibration(pattern_shares={'steady': 0, 'slipping': 0, 'regaining': 1, 'dropping_out': 0},
                         regain_month_range=(4, 4))
    truth, obs, _, _ = observed(rows, cal=regain, design=always_miss_if_gaining)
    a = obs[obs.scenario.eq('A')]
    assert a.loc[a.month.between(1, 4), 'attended'].all()            # losing weight: never missed
    assert not a.loc[a.month.ge(6), 'attended'].any()                # regaining: always missed
    assert obs[obs.scenario.eq('usual')].attended.all()              # no diet change, no gain


def test_observation_is_reproducible():
    rows = cohort(5)
    first, again = observed(rows, design=ObservationDesign())[1], observed(rows, design=ObservationDesign())[1]
    pd.testing.assert_frame_equal(first, again)
    assert len(first) == len(rows) * len(SCENARIOS) * (MONTHS + 1)
    assert set(LAB_CVI) <= set(first.columns)
