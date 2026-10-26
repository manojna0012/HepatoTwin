import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.simulation.growth import Growth
from src.simulation.parameters import (KJ_PER_KCAL, Calibration, Evidence, body_fat_kg, load_table,
                                       personal_weight_response, weight_time_constant)
from src.simulation.simulator import (BOUNDED, MONTHS, adherence_path, baseline_pdff, bounded, marker_bounds,
                                      person_traits, simulate_cohort, simulate_patient, weight_path)

SIMULATION_DIR = Path(__file__).resolve().parent
# Anything that would let the simulator see the twin's model or the twin's other layers.
FORBIDDEN = ('src.train_combined_model', 'src.train_final_model', 'src.steatosis_model', 'src.predict_cap',
             'src.twin_pipeline', 'src.recipe_scoring', 'src.cap_features', 'src.combined_features',
             'xgboost', 'joblib', 'sklearn')

EV = Evidence.from_table()
CAL = Calibration()


def patient(**overrides):
    row = {'SEQN': 1001, 'RIDAGEYR': 45, 'RIAGENDR': 1, 'DIQ010': 2, 'BMXWT': 95.0, 'BMXHT': 170.0,
           'BMXBMI': 95.0 / 1.7 ** 2, 'BMXWAIST': 108.0, 'BMXHIP': 110.0, 'BMXARMC': 35.0, 'LBXIN': 15.0,
           'LBXHSCRP': 3.0, 'LBXWBCSI': 7.5, 'LBXLYPCT': 30.0, 'LBXNEPCT': 58.0, 'LBXSUA': 6.0, 'LBXGH': 6.0, 'LBDHDD': 42.0,
           'LBXSTR': 180.0, 'LBXGLU': 110.0, 'LBXSGL': 112.0, 'BPXOSY1': 130.0, 'BPXOSY2': 128.0,
           'BPXOSY3': 129.0, 'LBXSATSI': 40.0, 'LBXSASSI': 30.0, 'LBXSGTSI': 45.0, 'LBXTC': 200.0,
           'DR1TKCAL': 2400.0, 'DR1TPROT': 90.0, 'DR1TCARB': 300.0, 'DR1TTFAT': 90.0, 'WHD050': 205.0,
           'LUXCAPM': 310.0}
    return row | overrides


def cohort(n=40):
    rng = np.random.default_rng(0)
    rows = [patient(SEQN=2000 + i, BMXWT=(w := float(rng.uniform(60, 130))), BMXBMI=w / 1.7 ** 2,
                    LUXCAPM=float(rng.uniform(200, 380)),
                    LBXSATSI=float(rng.uniform(12, 90)), LBXSTR=float(rng.uniform(60, 300)))
            for i in range(n)]
    return pd.DataFrame(rows)


def test_simulator_never_imports_the_twin():
    for path in SIMULATION_DIR.glob('*.py'):
        if path.name.startswith('test_'):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        names += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        bad = [n for n in names if any(n == f or n.startswith(f + '.') for f in FORBIDDEN)]
        assert not bad, f'{path.name} imports {bad}'
        assert '.joblib' not in path.read_text(encoding='utf-8'), f'{path.name} refers to a saved model'


def test_scope_rule_matches_the_training_module():
    from src.simulation.cohort import in_scope as simulator_scope
    from src.train_combined_model import in_scope as training_scope
    data = pd.DataFrame({'RIDAGEYR': [12, 30, 59, 60, 75, 45], 'LUXCPIQR': [10, 39.9, 20, 10, 5, 40]})
    assert simulator_scope(data).tolist() == training_scope(data).tolist()


def test_every_parameter_row_is_cited():
    table = load_table()
    assert table.source.notna().all() and table.access.notna().all()
    assert set(table.access) <= {'full text', 'abstract', 'database', 'design', 'secondary'}


def test_weight_follows_halls_rule_of_thumb():
    tau = weight_time_constant(12, 36)
    assert 12 < tau < 18                     # between 12/ln 20 and 12/ln 2
    # A sustained 100 kJ/day reduction at full adherence: about 1 kg in the end, roughly half by month 12.
    kcal = -100 / KJ_PER_KCAL
    change = weight_path(kcal, np.ones(MONTHS), 1.0, EV)
    eventual = -EV.kg_per_100kj
    assert change[0] == 0
    assert 0.45 < change[12] / eventual < 0.65
    assert np.all(np.diff(change) < 0)


def test_no_prescribed_change_means_no_change():
    frame = simulate_cohort(pd.DataFrame([patient()]), [{'name': 'usual', 'energy_change_kcal': 0}],
                            seed=1, ev=EV, cal=CAL)[0]
    for col in ('BMXWT', 'LBXGH', 'LBXSATSI', 'LUXCAPM', 'DR1TKCAL'):
        assert np.allclose(frame[col], frame[col].iloc[0])


def test_weight_loss_lowers_markers_and_cap_and_raises_hdl():
    steady = Calibration(pattern_shares={'steady': 1, 'slipping': 0, 'regaining': 0, 'dropping_out': 0},
                         adherence_sd=0)
    frame = simulate_cohort(pd.DataFrame([patient()]), [{'name': 'A', 'energy_change_kcal': -600}],
                            seed=1, ev=EV, cal=steady)[0]
    first, last = frame.iloc[0], frame.iloc[-1]
    assert last.BMXWT < first.BMXWT and last.BMXWAIST < first.BMXWAIST
    assert last.BMXHIP < first.BMXHIP and last.BMXARMC < first.BMXARMC
    assert last.LUXCAPM < first.LUXCAPM and last.true_pdff < first.true_pdff
    assert last.LBXSATSI <= first.LBXSATSI and last.LBXGH <= first.LBXGH and last.LBXIN <= first.LBXIN
    assert last.LBDHDD >= first.LBDHDD     # no diabetes: no HDL change (Appuhamy 2014)
    assert last.DR1TKCAL == pytest.approx(2400 - 600 * 0.7)


def test_scenarios_for_one_person_differ_only_in_diet():
    rows = pd.DataFrame([patient()])
    frame, traits = simulate_cohort(rows, [{'name': 'A', 'energy_change_kcal': -500},
                                           {'name': 'B', 'energy_change_kcal': -250}], seed=3, ev=EV, cal=CAL)
    a, b = (frame[frame.scenario.eq(s)].reset_index(drop=True) for s in ('A', 'B'))
    assert a.pattern.equals(b.pattern) and a.adherence.equals(b.adherence)
    assert (a.BMXWT <= b.BMXWT + 1e-9).all()     # the larger reduction loses at least as much
    assert len(traits) == 1


def test_results_are_reproducible_and_seed_dependent():
    rows, scenarios = cohort(), [{'name': 'A', 'energy_change_kcal': -500}]
    first = simulate_cohort(rows, scenarios, seed=7, ev=EV, cal=CAL)[0]
    again = simulate_cohort(rows, scenarios, seed=7, ev=EV, cal=CAL)[0]
    other = simulate_cohort(rows, scenarios, seed=8, ev=EV, cal=CAL)[0]
    pd.testing.assert_frame_equal(first, again)
    assert not first.BMXWT.equals(other.BMXWT)


def test_adherence_patterns():
    for pattern in ('steady', 'slipping', 'regaining', 'dropping_out'):
        traits = {'pattern': pattern, 'adherence0': 0.8, 'stop_month': 5}
        path = adherence_path(traits, CAL)
        assert len(path) == MONTHS and path[0] == pytest.approx(0.8)
        if pattern == 'steady':
            assert np.allclose(path, 0.8)
        elif pattern == 'slipping':
            assert np.all(np.diff(path) < 0)
        else:
            assert np.all(path[:5] == 0.8) and np.all(path[5:] == 0)


def test_person_traits_are_fixed_by_person_and_seed():
    assert person_traits(1001, 5, CAL) == person_traits(1001, 5, CAL)
    assert person_traits(1001, 5, CAL) != person_traits(1002, 5, CAL)
    multipliers = [person_traits(i, 5, CAL)['markers']['LBXSATSI'] for i in range(4000)]
    assert np.mean(multipliers) == pytest.approx(1, abs=0.05)   # the published slope on average


def test_bounds_stop_changes_at_the_cohort_range():
    assert bounded(30, -20, 15, 80) == 15
    assert bounded(12, -5, 15, 80) == 12       # already below the floor: does not move further down
    assert bounded(70, 20, 15, 80) == 80
    assert np.isnan(bounded(np.nan, -5, 15, 80))
    bounds = marker_bounds(cohort())
    assert bounds['LBXSATSI'][0] < bounds['LBXSATSI'][1]


def test_baseline_liver_fat_line():
    assert baseline_pdff(EV.anchor_cap, EV, CAL) == pytest.approx(EV.anchor_pdff)
    assert baseline_pdff(EV.anchor_cap + EV.cap_per_pdff_point, EV, CAL) == pytest.approx(EV.anchor_pdff + 1)
    assert baseline_pdff(100, EV, CAL) == CAL.min_pdff


def test_dropout_and_weight_a_year_ago():
    out = simulate_cohort(pd.DataFrame([patient()]), [{'name': 'A', 'energy_change_kcal': -500}],
                          seed=1, ev=EV, cal=Calibration(
                              pattern_shares={'steady': 0, 'slipping': 0, 'regaining': 0, 'dropping_out': 1}))[0]
    stop = out.loc[out.dropped_out, 'month'].min()
    assert out.loc[out.month < stop, 'dropped_out'].eq(False).all()
    assert out.WHD050.iloc[0] == pytest.approx(205.0)            # the reported value at month 0
    assert out.WHD050.iloc[12] == pytest.approx(out.BMXWT.iloc[0] / 0.4536)


STEADY = Calibration(pattern_shares={'steady': 1, 'slipping': 0, 'regaining': 0, 'dropping_out': 0},
                     adherence_sd=0, weight_response_sd=0, marker_response_sd=0, liver_response_sd=0)


OPEN_BOUNDS = {col: (-np.inf, np.inf) for col in BOUNDED}


def run(row, energy=-500, cal=STEADY):
    """One patient with no cohort bounds (a one-patient cohort would allow no change at all)."""
    return simulate_patient(pd.Series(row), {'name': 'A', 'energy_change_kcal': energy}, EV, cal, OPEN_BOUNDS, 1)


def test_more_body_fat_means_slower_but_larger_eventual_loss():
    lean, heavy = personal_weight_response(15, EV), personal_weight_response(50, EV)
    reference = personal_weight_response(EV.hall['fat_ref'], EV)
    assert reference == pytest.approx((1.0, EV.weight_tau_months))
    assert lean[0] < 1 < heavy[0]               # eventual loss
    assert lean[1] < EV.weight_tau_months < heavy[1]   # time constant
    assert body_fat_kg(100, 35, 45, male=False) > body_fat_kg(100, 35, 45, male=True)


def test_adolescents_grow_on_their_usual_diet_without_liver_change():
    teen = patient(SEQN=3001, RIDAGEYR=13, RIAGENDR=2, BMXWT=50.0, BMXHT=155.0, BMXBMI=50 / 1.55 ** 2)
    frame = run(teen, energy=0)[0]
    first, last = frame.iloc[0], frame.iloc[-1]
    assert 2 < last.BMXWT - first.BMXWT < 8 and 1 < last.BMXHT - first.BMXHT < 8
    assert last.LUXCAPM == pytest.approx(first.LUXCAPM) and last.LBXSATSI == pytest.approx(first.LBXSATSI)
    adult = run(patient(), energy=0)[0]
    assert adult.BMXWT.nunique() == 1 and adult.BMXHT.nunique() == 1
    growth = Growth()
    assert growth.weight_path(50, 2, 19.9, [0, 12])[1] < growth.weight_path(50, 2, 12, [0, 12])[1]


def test_diabetes_status_selects_the_marker_evidence():
    no_diabetes = run(patient())[0].iloc[-1]
    diabetes = run(patient(DIQ010=1))[0].iloc[-1]
    base = patient()['LBXGLU']
    assert base - diabetes.LBXGLU > base - no_diabetes.LBXGLU > 0
    assert diabetes.LBDHDD > no_diabetes.LBDHDD == pytest.approx(patient()['LBDHDD'])


def test_liver_enzymes_follow_liver_fat_not_weight():
    fatty = run(patient(LUXCAPM=340.0))[0].iloc[-1]
    healthy = run(patient(SEQN=1002, LUXCAPM=200.0))[0].iloc[-1]   # at the 1% PDFF floor
    fatty_drop, healthy_drop = (patient()['LBXSATSI'] - r.LBXSATSI for r in (fatty, healthy))
    assert fatty_drop > 3 and 0 <= healthy_drop < 2 and healthy_drop < fatty_drop / 5


def test_liver_fat_falls_ahead_of_weight():
    frame = run(patient(LUXCAPM=320.0), energy=-600)[0]
    month1 = frame.iloc[1]
    weight_pct = 100 * (month1.BMXWT / frame.BMXWT.iloc[0] - 1)
    fat_pct = 100 * (month1.true_pdff / frame.true_pdff.iloc[0] - 1)
    assert fat_pct < EV.liver_fat_pct_per_pct * weight_pct < 0   # more than the weight alone explains


def test_white_cells_fall_and_lymphocyte_share_rises_with_weight_loss():
    frame = run(patient())[0]
    first, last = frame.iloc[0], frame.iloc[-1]
    kg = last.BMXWT - first.BMXWT
    assert kg < 0
    assert last.LBXWBCSI == pytest.approx(7.5 * (1 + 0.0042 * kg))
    assert last.LBXLYPCT > first.LBXLYPCT and abs(last.LBXNEPCT - first.LBXNEPCT) < 0.1


def test_uric_acid_falls_per_kg_with_the_diabetes_specific_slope():
    for diq, slope in ((2, 0.03), (1, 0.04)):
        frame = run(patient(DIQ010=diq))[0]
        kg = frame.BMXWT.iloc[-1] - frame.BMXWT.iloc[0]
        assert frame.LBXSUA.iloc[-1] == pytest.approx(6.0 + slope * kg)
