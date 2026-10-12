from src.recipe_scoring.patient_objectives import build_patient_objectives
from src.recipe_scoring.health_alignment import health_alignment_score
from src.recipes.nutrient_profile import create_nutrient_profile


def score_recipe_for_patient(recipe, patient_profile, extreme_limits=None):
    """Score a recipe against a patient's nutrition priorities."""

    objectives = build_patient_objectives(patient_profile)
    profile = create_nutrient_profile(recipe)

    # The scorer expects nutrition values nested under "nutrition".
    nutrition = {
        key: value
        for key, value in profile.items()
        if key not in {"recipe_id", "title"}
    }

    result = health_alignment_score(
        recipe={"nutrition": nutrition},
        objectives=objectives,
        extreme_limits=extreme_limits,
    )

    return {
        "recipe_id": recipe.recipe_id,
        "title": recipe.title,
        "health_alignment": result,
    }
