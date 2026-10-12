
from src.recipe_scoring.patient_recipe_scoring import score_recipe_for_patient


def rank_patient_recipes(recipes, patient_profile, extreme_limits=None):
    """Rank recipes by patient-specific health alignment.

    Recipes with no valid health-alignment score are placed last.
    """
    scored_recipes = [
        score_recipe_for_patient(
            recipe,
            patient_profile,
            extreme_limits=extreme_limits,
        )
        for recipe in recipes
    ]

    return sorted(
        scored_recipes,
        key=lambda item: (
            item["health_alignment"]["score"] is None,
            -(item["health_alignment"]["score"] or 0),
        ),
    )
