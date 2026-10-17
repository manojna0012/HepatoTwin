"""Connect patient-state priorities with recipe ranking safely."""

from src.recipe_scoring.evidence_rules import EVIDENCE_RULES
from src.recipe_scoring.multi_criteria import DEFAULT_WEIGHTS, rank_recipes
from src.recipe_scoring.patient_objective_generator import (
    generate_patient_objectives,
)
from src.recipe_scoring.rank_patient_recipes import rank_patient_recipes
from src.recipe_scoring.recipe_criteria import criteria_for


def run_patient_workflow(
    patient_state,
    recipes,
    patient_profile=None,
    evidence_rules=None,
    extreme_limits=None,
    weights=DEFAULT_WEIGHTS,
    preferences=None,
):
    """Generate candidate priorities and rank recipes against them.

    Candidate priorities are explanatory guidance, not clinical prescriptions,
    and never carry a patient-level numeric target. Without an explicit
    patient profile, recipes are ranked on the evidence-linked recipe
    criteria in recipe_criteria.py, whose reference values describe recipe
    composition. preferences may give requested_cuisine, related_cuisines,
    diet and meals_only.
    """
    candidates = generate_patient_objectives(
        patient_state,
        EVIDENCE_RULES if evidence_rules is None else evidence_rules,
    )
    priorities = candidates["priorities"]

    profile = patient_profile or {}
    profile_priorities = profile.get("priorities")

    if isinstance(profile_priorities, list) and profile_priorities:
        ranked = rank_patient_recipes(
            recipes,
            profile,
            extreme_limits=extreme_limits,
        )
        return {
            "nutritional_priorities": priorities,
            "ranked_recipes": ranked,
            "not_ranked": [],
            "unmapped_priorities": [],
            "scoring_status": "scored_using_profile_objectives",
        }

    objectives, unmapped = criteria_for(priorities)
    if not objectives:
        status = "not_scored: no priority with a recipe criterion"
        ranked, not_ranked = [], []
    else:
        ranked, not_ranked = rank_recipes(recipes, objectives, weights=weights,
                                          **(preferences or {}))
        status = ("scored_using_evidence_linked_criteria" if recipes
                  else "not_scored: no candidate recipes")
    return {
        "nutritional_priorities": priorities,
        "ranked_recipes": ranked,
        "not_ranked": not_ranked,
        "unmapped_priorities": unmapped,
        "scoring_status": status,
    }
