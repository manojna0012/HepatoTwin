"""Connect patient-state priorities with recipe ranking safely."""

from src.recipe_scoring.evidence_rules import EVIDENCE_RULES
from src.recipe_scoring.patient_objective_generator import (
    generate_patient_objectives,
)
from src.recipe_scoring.rank_patient_recipes import rank_patient_recipes


def run_patient_workflow(
    patient_state,
    recipes,
    patient_profile=None,
    evidence_rules=None,
    extreme_limits=None,
):
    """Generate candidate priorities and rank recipes when valid objectives exist.

    Candidate priorities are explanatory guidance, not clinical prescriptions.
    Recipe ranking requires separately validated numeric nutrition objectives.
    """
    candidates = generate_patient_objectives(
        patient_state,
        EVIDENCE_RULES if evidence_rules is None else evidence_rules,
    )

    profile = patient_profile or {}
    priorities = profile.get("priorities")

    if isinstance(priorities, list) and priorities:
        ranked = rank_patient_recipes(
            recipes,
            profile,
            extreme_limits=extreme_limits,
        )
        scoring_status = "scored_using_validated_profile_objectives"
    else:
        ranked = []
        scoring_status = (
            "not_scored: validated numeric patient objectives are required"
        )

    return {
        "nutritional_priorities": candidates["priorities"],
        "ranked_recipes": ranked,
        "scoring_status": scoring_status,
    }
