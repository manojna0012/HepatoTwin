from dataclasses import replace

import numpy as np
import pytest

from src.simulation.calibrate import BOUNDS, TRIALS, Cohort, outcomes, pack, prescriptions, targets, unpack, \
    weight_change
from src.simulation.parameters import Calibration, body_fat_kg, load_table, personal_weight_response
from src.simulation.simulator import adherence_path, drift_path, person_traits, weight_path
from src.simulation.test_simulation import EV, cohort

RX = prescriptions(load_table())


def walk(kg, minutes):
    return -(3.8 - 1) * kg * minutes / 60 / 7


def test_prescriptions_follow_the_trial_papers():
    # From month 6 on, walking is at its goal and Look AHEAD's meal replacements have ended.
    late = 8
    assert RX['vilar_gomez'](80, 12)[late] == pytest.approx(-750 + walk(80, 200))
    assert RX['dpp'](70, 12)[late] == pytest.approx(-500 + walk(70, 150))           # 154 lb
    assert RX['dpp'](120, 12)[late] == pytest.approx(-1000 + walk(120, 150))        # 265 lb
    assert RX['dpp'](90, 12)[late] == pytest.approx(-667 + walk(90, 150))           # 198 lb
    # Look AHEAD: goal minus the DPP maintenance calories (goal + deficit) of the weight band.
    assert RX['look_ahead'](90, 12)[late] == pytest.approx(1350 - (1500 + 667) + walk(90, 175))
    assert RX['look_ahead'](120, 12)[late] == pytest.approx(1650 - (2000 + 1000) + walk(120, 175))
    assert np.all(RX['usual_care'](100, 12) == 0)


def test_walking_builds_up_and_meal_replacements_cover_weeks_3_to_19():
    vg = RX['vilar_gomez'](80, 12) + 750
    assert vg[0] == pytest.approx(walk(80, 90 + 110 * 0.5 / 6)) and np.all(np.diff(vg[:6]) < 0)
    assert np.allclose(vg[6:], walk(80, 200))
    la = RX['look_ahead'](90, 12) - RX['look_ahead'](90, 12)[8]
    walking = np.array([walk(90, 175 * min((m + 0.5) / 6, 1)) - walk(90, 175) for m in range(12)])
    meal = la - walking
    assert meal[0] == pytest.approx(-514 * (1 - 14 / 30.4375), rel=1e-3)        # weeks 3-4 of month 1
    assert np.allclose(meal[1:4], -514) and meal[4] < 0 and np.allclose(meal[5:], 0)


def test_targets_split_the_look_ahead_spread_row_and_use_the_vilar_gomez_shape():
    goal = targets()
    assert goal['LA-Y1-SD-ILI'][:2] == (7.6, 'kg') and goal['LA-Y1-SD-DSE'][0] == 5.0
    assert goal['VG-SHAPE-52W'] == (1.0, 'fraction', 'calibrate') and 'VG-KG-52W' not in goal
    assert goal['VG-SHAPE-8W'][0] == pytest.approx(3.3 / 4.6)
    assert goal['LA-Y8-MEAN-ILI'][2] == 'held back' and 'VJ-CAP-6M' not in goal
    assert goal['VG-GE5-52W'][2] == 'check' and goal['LA-Y1-REGAIN-T1'][2] == 'select'
    groups = [goal[k][0] for k in ('VG-GRP-LT5', 'VG-GRP-5TO7', 'VG-GRP-7TO10', 'VG-GRP-GE10')]
    assert sum(groups) == pytest.approx(100) and sum(groups[1:]) == pytest.approx(goal['VG-GE5-52W'][0])


def test_dropout_is_fixed_and_quitting_stays_under_the_cap():
    middle = np.array([(low + high) / 2 for low, high in BOUNDS.values()])
    for regaining in (0.0, 0.5, 1.0):
        cal, adherence = unpack(np.r_[regaining, middle[1:]], Calibration(), 0.082, 0.2)
        shares = cal.pattern_shares
        assert shares['dropping_out'] == 0.082 and sum(shares.values()) == pytest.approx(1)
        assert shares['regaining'] + shares['dropping_out'] <= 0.2 + 1e-12
    cal, adherence = unpack(middle, Calibration(), 0.082, 0.3)
    assert cal.slip_floor == pytest.approx(0.45) and adherence['dpp'] == pytest.approx(0.525)
    again = unpack(pack(cal, adherence, 0.082, 0.3), Calibration(), 0.082, 0.3)
    assert again[0].slip_floor == pytest.approx(cal.slip_floor)
    assert again[0].pattern_shares == pytest.approx(cal.pattern_shares)
    assert again[1] == pytest.approx(adherence) and set(adherence) == set(TRIALS)


def test_drift_is_a_seeded_random_walk_that_is_off_by_default():
    assert np.all(drift_path(1001, 1, Calibration()) == 0)
    walk = drift_path(1001, 1, Calibration(weight_drift_sd=1.0), months=24)
    assert walk[0] == 0 and len(walk) == 25
    assert np.allclose(walk, drift_path(1001, 1, Calibration(weight_drift_sd=1.0), months=24))
    ends = [drift_path(i, 1, Calibration(weight_drift_sd=1.0))[-1] for i in range(3000)]
    assert np.std(ends) == pytest.approx(np.sqrt(12), rel=0.05)


def test_the_fast_cohort_matches_the_simulator_person_by_person():
    people = cohort(25)
    cal = Calibration(weight_drift_sd=1.1, slip_floor=0.3, slip_months=2.5, adherence_sd=0.2, weight_response_sd=0.3,
                      pattern_shares={'steady': 0.3, 'slipping': 0.3, 'regaining': 0.2, 'dropping_out': 0.2})
    total, completed, diet = Cohort(people, EV, 18, 5, cal).run(RX['dpp'], cal, 0.8)
    for i, row in enumerate(people.itertuples(index=False)):
        traits = person_traits(row.SEQN, 5, replace(cal, adherence_mean=0.8))
        eventual, tau = personal_weight_response(body_fat_kg(row.BMXWT, row.BMXBMI, row.RIDAGEYR, row.RIAGENDR == 1), EV)
        expected = weight_path(RX['dpp'](row.BMXWT, 18), adherence_path(traits, cal, 18),
                               traits['weight_response'] * eventual, EV, tau)
        assert np.allclose(diet[i], expected)
        assert np.allclose(total[i], expected + drift_path(row.SEQN, 5, cal, 18))
        assert completed[i] == (traits['pattern'] != 'dropping_out')


def test_trial_outcomes_respond_to_adherence():
    people = cohort(30)
    low, done, _ = weight_change(people, RX['vilar_gomez'], Calibration(), 0.3, EV, 12)
    high, _, diet = weight_change(people, RX['vilar_gomez'], Calibration(), 0.9, EV, 12)
    assert np.allclose(high, diet)                  # no drift by default
    assert high[:, 12].mean() < low[:, 12].mean() < 0 and done.dtype == bool
    cohorts = {'look_ahead': people, 'dpp': people, 'vilar_gomez': people}
    sim = outcomes(cohorts, RX, Calibration(), {t: 0.7 for t in TRIALS}, EV, months=34)
    assert sim['LA-Y1-MEAN-DSE'] == 0 and sim['LA-Y1-MEAN-ILI'] < 0
    assert 'DPP-GOAL-LAST' in sim and 'LA-Y8-MEAN-ILI' not in sim
    assert sim['VG-SHAPE-52W'] == pytest.approx(1) and sim['LA-Y1-REGAIN-T1'] >= 0
    assert sum(sim[k] for k in ('VG-GRP-LT5', 'VG-GRP-5TO7', 'VG-GRP-7TO10', 'VG-GRP-GE10')) == pytest.approx(100)
