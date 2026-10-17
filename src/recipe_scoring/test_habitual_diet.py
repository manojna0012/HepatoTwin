import pytest

from src.recipe_scoring.evidence_rules import EVIDENCE_RULES
from src.recipe_scoring.habitual_diet import habitual_intake, hei2020, recall_reliable
from src.recipe_scoring.patient_objective_generator import generate_patient_objectives
from src.recipe_scoring.patient_state import derive_patient_state

THRESHOLDS = {"cap248": 0.43, "cap280": 0.36}
LEAN_STEATOSIS = {"RIAGENDR": 2, "RIDRETH3": 3, "BMXBMI": 22, "BMXWAIST": 80,
                  "LBXGH": 5.2, "LBXTR": 100, "LBDHDD": 60}


def recall(**values):
    """A reliable, typical 2,000 kcal day for a woman."""
    row = {"RIAGENDR": 2, "DR1DRSTZ": 1, "DR1_300": 2, "DR1TKCAL": 2000,
           "DR1TSFAT": 20, "DR1TSUGR": 50, "DR1TFIBE": 30, "DR1T_ADD_SUGARS": 5}
    row.update(values)
    return row


def priorities(diet):
    state = derive_patient_state({**LEAN_STEATOSIS, **diet}, {"cap248": 0.6, "cap280": 0.2}, THRESHOLDS)
    return {p["id"]: p for p in generate_patient_objectives(state, EVIDENCE_RULES)["priorities"]}


def test_intake_uses_recipe_measures():
    intake = habitual_intake(recall())
    assert intake["recall_reliable"] is True
    assert intake["saturated_fat_pct_energy"] == pytest.approx(9.0)
    assert intake["sugar_pct_energy"] == pytest.approx(10.0)
    assert intake["added_sugar_pct_energy"] == pytest.approx(4.0)
    assert intake["fiber_per_1000kcal"] == pytest.approx(15.0)


@pytest.mark.parametrize("change", [{"DR1DRSTZ": 2}, {"DR1_300": 3}, {"DR1TKCAL": 400},
                                    {"RIAGENDR": 1, "DR1TKCAL": 4500}, {"DR1TKCAL": None}])
def test_unreliable_or_atypical_recalls_are_not_used(change):
    assert recall_reliable(recall(**change)) is False
    assert habitual_intake(recall(**change))["saturated_fat_pct_energy"] is None


def test_priorities_move_one_level_with_intake():
    good = priorities(recall())
    assert good["limit_saturated_fat"]["priority"] == "low"      # base medium, intake met
    assert good["diet_quality"]["priority"] == "low"
    assert good["limit_excess_sugar"]["priority"] == "low"       # already at the floor
    assert "kept" in good["limit_excess_sugar"]["priority_basis"]

    poor = priorities(recall(DR1TSFAT=40, DR1TFIBE=10, DR1T_ADD_SUGARS=20))
    assert poor["limit_saturated_fat"]["priority"] == "high"
    assert poor["diet_quality"]["priority"] == "high"
    assert poor["limit_excess_sugar"]["priority"] == "medium"
    assert poor["limit_saturated_fat"]["intake"] == {
        "measure": "saturated_fat_pct_energy", "value": 18.0, "reference": 10,
        "direction": "lower", "meets_reference": False}


def test_without_recall_priorities_are_unchanged():
    found = priorities({})
    assert found["limit_saturated_fat"]["priority"] == "medium"
    assert found["limit_saturated_fat"]["intake"] is None
    assert found["limit_excess_sugar"]["priority"] == "low"


def test_missing_added_sugars_leaves_sugar_priority_unchanged():
    found = priorities(recall(DR1T_ADD_SUGARS=None))
    assert found["limit_excess_sugar"]["intake"] is None
    assert found["limit_saturated_fat"]["intake"] is not None


def test_hei2020_bounds():
    best = recall(DR1T_F_TOTAL=2, DR1T_F_CITMLB=1, DR1T_F_OTHER=1, DR1T_V_TOTAL=3, DR1T_V_LEGUMES=0.5,
                  DR1T_V_DRKGR=0.5, DR1T_G_WHOLE=4, DR1T_G_REFINED=2, DR1T_D_TOTAL=3, DR1T_PF_TOTAL=5,
                  DR1T_PF_LEGUMES=1, DR1T_PF_SEAFD_HI=1, DR1T_PF_SEAFD_LOW=0.5, DR1T_PF_SOY=0,
                  DR1T_PF_NUTSDS=1, DR1T_ADD_SUGARS=2, DR1TSFAT=12, DR1TMFAT=30, DR1TPFAT=20, DR1TSODI=1800)
    assert hei2020(best)["total"] == pytest.approx(100)
    empty = {k: 0 for k in best if k.startswith("DR1T_")}
    worst = {**best, **empty, "DR1T_ADD_SUGARS": 40, "DR1T_G_REFINED": 10, "DR1TSFAT": 50,
             "DR1TMFAT": 20, "DR1TPFAT": 10, "DR1TSODI": 5000}
    assert hei2020(worst)["total"] == pytest.approx(0)
    assert hei2020(recall()) is None


def test_derived_fped_scales_by_grams_and_reports_coverage():
    import pandas as pd
    from src.build_habitual_diet import FPED_COLUMNS, derive_fped
    per_100g = {c.removeprefix("DR1T_"): 0.0 for c in FPED_COLUMNS}
    foods = pd.DataFrame([{"FOODCODE": 1, **per_100g, "ADD_SUGARS": 2.0, "F_TOTAL": 0.5},
                          {"FOODCODE": 2, **per_100g, "ADD_SUGARS": 10.0}])
    eaten = pd.DataFrame({"SEQN": [7, 7, 7, 8], "DR1IFDCD": [1, 2, 99, 1],
                          "DR1IGRMS": [200, 50, 100, 100], "DR1IKCAL": [300, 100, 100, 150]})
    day = derive_fped(eaten, foods).set_index("SEQN")
    assert day.loc[7, "DR1T_ADD_SUGARS"] == pytest.approx(2.0 * 2 + 10.0 * 0.5)   # food 99 has no code
    assert day.loc[7, "DR1T_F_TOTAL"] == pytest.approx(1.0)
    assert day.loc[7, "fped_energy_matched"] == pytest.approx(0.8)
    assert day.loc[8, "fped_energy_matched"] == pytest.approx(1.0)
