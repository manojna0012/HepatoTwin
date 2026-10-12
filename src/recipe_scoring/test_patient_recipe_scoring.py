from src.recipes.recipe_model import Recipe
from src.recipe_scoring.patient_recipe_scoring import score_recipe_for_patient


def make_recipe(recipe_id, title, sodium, fiber):
    return Recipe(
        recipe_id=recipe_id,
        title=title,
        cuisine="Indian",
        category="main",
        servings=2,
        prep_time=10,
        cook_time=20,
        total_time=30,
        nutrition={
            "calories": 300,
            "protein": 12,
            "carbohydrates": 40,
            "fat": 8,
            "saturated_fat": 2,
            "cholesterol": 0,
            "fiber": fiber,
            "sugar": 5,
            "sodium": sodium,
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


def test_recipe_is_scored_for_patient():
    recipe = make_recipe("r1", "Low Sodium Meal", sodium=100, fiber=8)

    patient = {
        "priorities": [
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "high",
                "target": 100,
                "tolerance": 50,
            }
        ]
    }

    result = score_recipe_for_patient(recipe, patient)

    assert result["recipe_id"] == "r1"
    assert result["title"] == "Low Sodium Meal"
    assert result["health_alignment"]["score"] == 100


def test_patient_profiles_can_score_recipes_differently():
    low_sodium = make_recipe("r1", "Low Sodium", sodium=100, fiber=3)
    high_fiber = make_recipe("r2", "High Fiber", sodium=500, fiber=12)

    sodium_patient = {
        "priorities": [
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "high",
                "target": 100,
                "tolerance": 500,
            }
        ]
    }

    fiber_patient = {
        "priorities": [
            {
                "nutrient": "fiber",
                "direction": "higher",
                "priority": "high",
                "target": 12,
                "tolerance": 5,
            }
        ]
    }

    sodium_result_1 = score_recipe_for_patient(low_sodium, sodium_patient)
    sodium_result_2 = score_recipe_for_patient(high_fiber, sodium_patient)

    fiber_result_1 = score_recipe_for_patient(low_sodium, fiber_patient)
    fiber_result_2 = score_recipe_for_patient(high_fiber, fiber_patient)

    assert (
        sodium_result_1["health_alignment"]["score"]
        > sodium_result_2["health_alignment"]["score"]
    )
    assert (
        fiber_result_2["health_alignment"]["score"]
        > fiber_result_1["health_alignment"]["score"]
    )
