from src.recipes.recipe_model import Recipe
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



def test_workflow_scores_and_ranks_recipes_with_valid_objectives():
    from src.recipes.recipe_model import Recipe

    def make_recipe(recipe_id, title, protein):
        return Recipe(
            recipe_id=recipe_id,
            title=title,
            cuisine="test",
            category="main",
            servings=1,
            prep_time=5,
            cook_time=10,
            total_time=15,
            nutrition={
                "calories": 300,
                "protein": protein,
                "carbohydrates": 30,
                "fat": 10,
                "saturated_fat": 2,
                "cholesterol": 0,
                "fiber": 5,
                "sugar": 3,
                "sodium": 200,
            },
            ingredients=[],
            instructions=[],
            diet=None,
            diets=[],
            ratings=None,
            ratings_count=None,
            source="test",
            url="",
        )

    recipes = [
        make_recipe("low-protein", "Lower Protein", 10),
        make_recipe("high-protein", "Higher Protein", 30),
    ]

    patient_profile = {
        "priorities": [
            {
                "nutrient": "protein",
                "direction": "higher",
                "target": 30,
                "tolerance": 5,
                "priority": "high",
            }
        ]
    }

    result = run_patient_workflow(
        patient_state={},
        recipes=recipes,
        patient_profile=patient_profile,
    )

    assert result["scoring_status"] == (
        "scored_using_profile_objectives"
    )
    assert len(result["ranked_recipes"]) == 2
    assert all(
        item["health_alignment"]["score"] is not None
        for item in result["ranked_recipes"]
    )
    assert result["ranked_recipes"][0]["recipe_id"] == "high-protein"

