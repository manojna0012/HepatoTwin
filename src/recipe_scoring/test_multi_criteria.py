import pytest

from src.recipe_scoring.multi_criteria import explain, is_candidate, overall_score, rank_recipes
from src.recipe_scoring.patient_workflow import run_patient_workflow
from src.recipe_scoring.recipe_criteria import criteria_for, derived_metrics


def recipe(recipe_id, calories, saturated_fat, sugar, fiber, cuisine="Asian", minutes=30,
           category="One Dish Meal", title=None, diet="vegan", diets=("vegan",)):
    protein, fat = 20, 15
    carbohydrates = max(0.0, (calories - 4 * protein - 9 * fat) / 4)
    return {
        "recipe_id": recipe_id, "title": title or f"Recipe {recipe_id}", "cuisine": cuisine,
        "category": category, "total_time": minutes, "diet": diet, "diets": list(diets),
        "ingredients": ["a"] * 6, "instructions": ["step"] * 5,
        "nutrition": {"calories": calories, "protein": protein, "fat": fat, "carbohydrates": carbohydrates,
                      "saturated_fat": saturated_fat, "sugar": sugar, "fiber": fiber, "sodium": 400},
    }


PRIORITIES = [{"id": "limit_saturated_fat", "priority": "high", "supporting_evidence": []},
              {"id": "diet_quality", "priority": "medium", "supporting_evidence": []}]


def test_derived_metrics_are_energy_relative():
    metrics = derived_metrics({"calories": 500, "saturated_fat": 5, "sugar": 10, "fiber": 7})
    assert metrics["saturated_fat_pct_energy"] == pytest.approx(9.0)
    assert metrics["sugar_pct_energy"] == pytest.approx(8.0)
    assert metrics["fiber_per_1000kcal"] == pytest.approx(14.0)
    assert derived_metrics({"calories": 0})["sugar_pct_energy"] is None


def test_priority_level_sets_weight_and_unknown_priorities_are_reported():
    objectives, unmapped = criteria_for(PRIORITIES + [{"id": "exercise", "priority": "low"}])
    assert [o["weight"] for o in objectives] == [3.0, 2.0]
    assert unmapped == ["exercise"]


def test_recipe_meeting_references_ranks_first():
    objectives, _ = criteria_for(PRIORITIES)
    recipes = [recipe("fatty", 600, 20, 5, 10), recipe("lean", 500, 3, 5, 10)]
    ranked, _ = rank_recipes(recipes, objectives)
    assert [r["recipe_id"] for r in ranked] == ["lean", "fatty"]
    assert ranked[0]["meets_all_references"] and not ranked[1]["meets_all_references"]
    assert "Saturated fat" in explain(ranked[0])


def test_health_gate_stops_trading_health_for_cuisine():
    objectives, _ = criteria_for(PRIORITIES)
    recipes = [recipe("healthy_other", 500, 3, 5, 10, cuisine="European"),
               recipe("close_asian", 500, 6.5, 5, 10, cuisine="Asian")]
    gated, _ = rank_recipes(recipes, objectives, requested_cuisine="Asian")
    ungated, _ = rank_recipes(recipes, objectives, requested_cuisine="Asian", health_gate=False)
    assert gated[0]["recipe_id"] == "healthy_other"
    assert ungated[0]["recipe_id"] == "close_asian"


def test_candidate_filter_excludes_non_meals_and_conflicting_diet_tags():
    assert not is_candidate(recipe("d", 400, 3, 5, 10, category="Dessert"))
    assert not is_candidate(recipe("s", 400, 3, 5, 10, title="Taco Seasoning Mix"))
    assert not is_candidate(recipe("tiny", 90, 1, 1, 3))
    prawns = recipe("p", 400, 3, 5, 10, diet="vegan", diets=("pescatarian",))
    assert is_candidate(prawns, diet="pescetarian")
    assert not is_candidate(prawns, diet="vegetarian")


def test_implausible_nutrition_is_held_for_review():
    objectives, _ = criteria_for(PRIORITIES)
    ranked, not_ranked = rank_recipes([recipe("bad", 38000, 3, 5, 10)], objectives, meals_only=False)
    assert ranked == []
    assert not_ranked[0]["status"] == "held_for_review"


def test_missing_cuisine_preference_renormalises_weights():
    assert overall_score({"health": 100, "feasibility": 50}) == pytest.approx(87.5)


def test_workflow_ranks_from_layer3_priorities_without_profile():
    state = {"steatosis_likely": 1, "diet_quality_review_requested": 1}
    result = run_patient_workflow(state, [recipe("a", 500, 3, 5, 10), recipe("b", 500, 12, 5, 2)])
    assert result["scoring_status"] == "scored_using_evidence_linked_criteria"
    assert result["ranked_recipes"][0]["recipe_id"] == "a"
    assert all(p["numeric_target"] is None for p in result["nutritional_priorities"])
