
from src.recipe_scoring.rank_patient_recipes import rank_patient_recipes
from src.recipes.recipe_model import Recipe


def make_recipe(recipe_id, title, sodium, fiber, protein):
    return Recipe(
        recipe_id=recipe_id,
        title=title,
        cuisine="general",
        category="main",
        servings=2,
        prep_time=10,
        cook_time=20,
        total_time=30,
        nutrition={"sodium": sodium, "fiber": fiber, "protein": protein},
        ingredients=[],
        instructions=[],
        diet=None,
        diets=[],
        ratings=None,
        ratings_count=0,
        source="test",
        url="",
    )


def test_rank_patient_recipes_returns_scored_results():
    recipes = [
        make_recipe("r1", "High sodium meal", 1800, 2, 15),
        make_recipe("r2", "Lower sodium meal", 400, 8, 20),
    ]

    patient_profile = {
        "priorities": [
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "high",
                "target": 500,
                "tolerance": 200,
            }
        ]
    }

    ranked = rank_patient_recipes(recipes, patient_profile)

    assert len(ranked) == 2
    assert all("health_alignment" in item for item in ranked)
    assert ranked[0]["title"] == "Lower sodium meal"
