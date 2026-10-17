from src.recipe_scoring.evidence_rules import EVIDENCE_RULES
from src.recipe_scoring.evidence_table import load_evidence
from src.recipe_scoring.patient_objective_generator import generate_patient_objectives
from src.recipe_scoring.patient_state import derive_patient_state

THRESHOLDS = {"cap248": 0.43, "cap280": 0.36}


def priorities_for(measurements, probabilities):
    state = derive_patient_state(measurements, probabilities, THRESHOLDS)
    result = generate_patient_objectives(state, EVIDENCE_RULES)
    return {item["id"]: item for item in result["priorities"]}


def test_high_risk_patient_gets_graded_priorities():
    measurements = {"RIAGENDR": 1, "RIDRETH3": 3, "BMXBMI": 33, "BMXWAIST": 110,
                    "LBXGH": 6.1, "LBXSTR": 210, "LBDHDD": 35}
    found = priorities_for(measurements, {"cap248": 0.92, "cap280": 0.81})

    assert set(found) == {"weight_management", "diet_quality",
                          "limit_saturated_fat", "limit_excess_sugar"}
    assert found["weight_management"]["priority"] == "high"
    assert found["diet_quality"]["priority"] == "high"
    assert found["limit_saturated_fat"]["priority"] == "high"
    assert found["limit_excess_sugar"]["priority"] == "medium"
    assert all(item["numeric_target"] is None for item in found.values())


def test_lean_patient_with_steatosis_gets_diet_but_not_weight_priorities():
    measurements = {"RIAGENDR": 2, "RIDRETH3": 3, "BMXBMI": 22, "BMXWAIST": 80,
                    "LBXGH": 5.2, "LBXTR": 100, "LBDHDD": 60}
    found = priorities_for(measurements, {"cap248": 0.6, "cap280": 0.2})

    assert "weight_management" not in found
    assert found["diet_quality"]["priority"] == "medium"
    assert found["limit_saturated_fat"]["priority"] == "medium"
    assert found["limit_excess_sugar"]["priority"] == "low"


def test_unlikely_steatosis_gives_no_liver_priorities():
    measurements = {"RIAGENDR": 1, "BMXBMI": 31, "BMXWAIST": 108}
    assert priorities_for(measurements, {"cap248": 0.2, "cap280": 0.05}) == {}


def test_asian_bmi_cutoffs_and_unknown_values():
    state = derive_patient_state({"RIDRETH3": 6, "BMXBMI": 24}, {"cap248": 0.5},
                                 THRESHOLDS)
    assert state["overweight"] == 1
    assert state["obesity"] == 0
    assert state["central_obesity"] is None
    assert state["steatosis_moderate_severe_likely"] is None


def test_every_rule_cites_existing_evidence():
    evidence = load_evidence()
    for rule in EVIDENCE_RULES:
        ids = rule["candidate"]["evidence_ids"]
        assert ids and all(e in evidence for e in ids)
        assert any(evidence[e]["study_id"] == rule["candidate"]["evidence_id"] for e in ids)


def test_priorities_carry_their_evidence():
    found = priorities_for({"RIAGENDR": 2, "BMXBMI": 22}, {"cap248": 0.6})
    strengths = {e["evidence_strength"] for e in found["limit_excess_sugar"]["supporting_evidence"]}
    assert "WEAK / INCONCLUSIVE" in strengths
