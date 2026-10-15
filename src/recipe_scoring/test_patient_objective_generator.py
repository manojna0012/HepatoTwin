from src.recipe_scoring.patient_objective_generator import (
    generate_patient_objectives,
)
from src.recipe_scoring.evidence_rules import EVIDENCE_RULES


def test_explicit_weight_management_context_generates_candidate():
    result = generate_patient_objectives(
        {"weight_management_appropriate": 1},
        EVIDENCE_RULES,
    )

    assert len(result["priorities"]) == 1
    item = result["priorities"][0]
    assert item["id"] == "weight_management"
    assert item["numeric_target"] is None
    assert item["requires_clinical_review"] is True


def test_missing_context_does_not_trigger_personalized_candidate():
    result = generate_patient_objectives({}, EVIDENCE_RULES)
    assert result["priorities"] == []


def test_diet_quality_candidate_requires_explicit_context():
    result = generate_patient_objectives(
        {"diet_quality_review_requested": 1},
        EVIDENCE_RULES,
    )

    assert len(result["priorities"]) == 1
    assert result["priorities"][0]["id"] == "diet_quality"


def test_duplicate_candidates_are_removed():
    rules = [EVIDENCE_RULES[0], EVIDENCE_RULES[0]]
    result = generate_patient_objectives(
        {"weight_management_appropriate": 1},
        rules,
    )
    assert len(result["priorities"]) == 1