
from src.recipe_scoring.patient_workflow import run_patient_workflow
from src.recipe_scoring.evidence_rules import EVIDENCE_RULES


def test_workflow_returns_priorities_without_inventing_targets():
    result = run_patient_workflow(
        patient_state={"weight_management_appropriate": 1},
        recipes=[],
        evidence_rules=EVIDENCE_RULES,
    )

    assert result["nutritional_priorities"][0]["priority"] == "medium"
    assert result["nutritional_priorities"][0]["numeric_target"] is None
    assert result["ranked_recipes"] == []
    assert result["scoring_status"].startswith("not_scored")


def test_workflow_does_not_trigger_candidates_without_context():
    result = run_patient_workflow(
        patient_state={},
        recipes=[],
        evidence_rules=EVIDENCE_RULES,
    )

    assert result["nutritional_priorities"] == []
