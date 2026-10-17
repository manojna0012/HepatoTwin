
from src.recipe_scoring.health_alignment import health_alignment_score


def test_objectives_are_scored():
    recipe = {
        "nutrition": {
            "sodium": 500,
            "fiber": 10,
        }
    }

    objectives = [
        {
            "nutrient": "sodium",
            "direction": "lower",
            "target": 500,
            "tolerance": 1000,
            "weight": 1,
        },
        {
            "nutrient": "fiber",
            "direction": "higher",
            "target": 10,
            "tolerance": 10,
            "weight": 1,
        },
    ]

    result = health_alignment_score(recipe, objectives)

    assert result["score"] == 100
    assert result["coverage"] == 1.0
    assert result["warnings"] == []


def test_missing_nutrient():
    recipe = {"nutrition": {"sodium": 500}}

    objectives = [
        {
            "nutrient": "fiber",
            "direction": "higher",
            "target": 10,
            "tolerance": 10,
            "weight": 1,
        }
    ]

    result = health_alignment_score(recipe, objectives)

    assert result["score"] is None
    assert result["coverage"] == 0.0
    assert result["warnings"]


def test_extreme_value_is_flagged():
    recipe = {"nutrition": {"sodium": 50000}}

    objectives = [
        {
            "nutrient": "sodium",
            "direction": "lower",
            "target": 500,
            "tolerance": 1000,
            "weight": 1,
        }
    ]

    result = health_alignment_score(
        recipe, objectives, extreme_limits={"sodium": 10000}
    )

    assert result["score"] is None
    assert any("review limit" in warning for warning in result["warnings"])

def test_extreme_value_is_not_rewarded_with_other_objectives():
    objectives = [
        {"nutrient": "saturated_fat", "direction": "lower",
         "target": 5, "tolerance": 10, "weight": 3},
        {"nutrient": "fiber", "direction": "higher",
         "target": 8, "tolerance": 8, "weight": 1},
    ]
    limits = {"saturated_fat": 40}

    moderate = health_alignment_score(
        {"nutrition": {"saturated_fat": 12, "fiber": 8}}, objectives, limits
    )
    extreme = health_alignment_score(
        {"nutrition": {"saturated_fat": 60, "fiber": 8}}, objectives, limits
    )

    assert moderate["score"] is not None
    assert extreme["score"] is None
    assert any("held for review" in warning for warning in extreme["warnings"])

def test_repeated_nutrient_objectives_coverage():
    recipe = {"nutrition": {"sodium": 500}}

    objectives = [
        {
            "nutrient": "sodium",
            "direction": "lower",
            "target": 500,
            "tolerance": 1000,
            "weight": 1,
        },
        {
            "nutrient": "sodium",
            "direction": "lower",
            "target": 400,
            "tolerance": 1000,
            "weight": 1,
        },
    ]

    result = health_alignment_score(recipe, objectives)

    assert result["coverage"] == 1.0
    assert result["score"] is not None

if __name__ == "__main__":
    test_objectives_are_scored()
    test_repeated_nutrient_objectives_coverage()
    test_missing_nutrient()
    test_extreme_value_is_flagged()
    print("All health alignment tests passed.")