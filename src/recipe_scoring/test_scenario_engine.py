import pytest

from src.recipe_scoring.evidence_rules import EVIDENCE_RULES
from src.recipe_scoring.evidence_table import load_evidence
from src.recipe_scoring.scenario_engine import (compare_scenarios, current_state, evaluate_scenario,
                                                steatosis_grade, timeline)

# Usual diet: 2,400 kcal a day (800 per meal), 13.5% saturated fat, 25% sugars, 7.5 g fibre/1,000 kcal.
PATIENT = {"RIAGENDR": 1, "DR1DRSTZ": 1, "DR1_300": 2, "DR1TKCAL": 2400, "DR1TSFAT": 36,
           "DR1TSUGR": 150, "DR1TFIBE": 18, "DR1T_ADD_SUGARS": 20}
PRIORITIES = [{"id": "weight_management", "priority": "high"},
              {"id": "limit_saturated_fat", "priority": "medium"},
              {"id": "limit_excess_sugar", "priority": "medium"},
              {"id": "diet_quality", "priority": "high"}]


def recipe(rid, calories, saturated_fat, sugar, fiber):
    return {"recipe_id": rid, "title": f"Recipe {rid}",
            "nutrition": {"calories": calories, "saturated_fat": saturated_fat, "sugar": sugar, "fiber": fiber}}


HEALTHY = recipe(1, 450, 2, 5, 12)          # 4% sat fat, 4.4% sugars, 26.7 g fibre/1,000 kcal
LIKE_USUAL = recipe(2, 800, 12, 50, 6)      # close to the usual diet


def by_id(result):
    return {c["priority_id"]: c for c in result["criteria"]}


def test_healthier_scenario_improves_with_graded_evidence():
    result = evaluate_scenario([HEALTHY], PRIORITIES, PATIENT)
    assert result["projected_direction"] == "improvement"
    assert result["evidence_strength"] == "MODERATE"
    criteria = by_id(result)
    assert all(c["direction"] == "improves" for c in criteria.values())
    assert criteria["weight_management"]["conditional"]["evidence"][0]["evidence_strength"] == "STRONG"
    assert criteria["limit_excess_sugar"]["evidence_strength"] == "MODERATE"   # energy also lower
    assert result["label"] == "Simulated projection, not a clinical forecast"


def test_sugar_without_energy_reduction_is_weak():
    swap = recipe(3, 800, 12, 20, 6)        # sugars down, energy unchanged
    sugar = by_id(evaluate_scenario([swap], PRIORITIES, PATIENT))["limit_excess_sugar"]
    assert sugar["direction"] == "improves"
    assert sugar["evidence_strength"] == "WEAK / INCONCLUSIVE"


def test_scenario_like_usual_diet_shows_no_change_and_small_shift():
    result = evaluate_scenario([LIKE_USUAL], PRIORITIES, PATIENT)
    assert result["projected_direction"] == "no clear change"
    assert result["evidence_strength"] is None
    assert result["shift_from_usual"] == "small"


def test_worse_scenario_is_reported_as_worsening():
    worse = recipe(4, 1100, 25, 80, 3)
    assert evaluate_scenario([worse], PRIORITIES, PATIENT)["projected_direction"] == "worsening"


def test_without_recall_compares_with_reference_values():
    result = evaluate_scenario([HEALTHY], PRIORITIES, {})
    assert {c["compared_with"] for c in result["criteria"]} == {"reference value"}
    assert result["shift_from_usual"] is None


def test_no_mapped_priority_is_insufficient():
    result = evaluate_scenario([HEALTHY], [{"id": "unknown", "priority": "low"}], PATIENT)
    assert result["evidence_strength"] == "INSUFFICIENT"


def test_compare_scenarios_b_is_closer_to_usual_than_a():
    ranked = [{"recipe_id": 1, "meets_all_references": True, "health_margin": 90, "overall": 90},
              {"recipe_id": 2, "meets_all_references": True, "health_margin": 10, "overall": 80}]
    a, b = compare_scenarios(ranked, [HEALTHY, LIKE_USUAL], PRIORITIES, PATIENT, size=1)
    assert a["recipe_ids"] == [1] and b["recipe_ids"] == [2]
    assert a["shift_score"] > b["shift_score"]


def test_grades_and_timeline_never_predict_a_number():
    assert [steatosis_grade(c) for c in (230, 250, 270, 300)] == ["S0", "S1", "S2", "S3"]
    now = current_state({"predicted_cap": 290.0, "probabilities": {"cap248": 0.9, "cap280": 0.6}})
    steps = timeline(now, evaluate_scenario([HEALTHY], PRIORITIES, PATIENT))
    assert [s["horizon"] for s in steps] == ["today", "3 months", "6 months", "1 year"]
    assert steps[0]["grade"] == "S3"
    assert all("grade" not in s and s["direction"] == "towards a lower grade" for s in steps[1:])


def test_scenario_needs_recipes():
    with pytest.raises(ValueError):
        evaluate_scenario([], PRIORITIES, PATIENT)


def test_rules_never_cite_observational_rows():
    evidence = load_evidence()
    cited = {e for rule in EVIDENCE_RULES for e in rule["candidate"]["evidence_ids"]}
    assert all(evidence[e]["evidence_strength"] != "OBSERVATIONAL ASSOCIATION" for e in cited)
    assert sum(r["evidence_strength"] == "OBSERVATIONAL ASSOCIATION" for r in evidence.values()) == 3
