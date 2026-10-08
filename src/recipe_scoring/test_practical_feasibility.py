
from practical_feasibility import practical_feasibility_score


def test_quick_recipe():
    recipe = {
        "total_time": 20,
        "prep_time": 10,
        "cook_time": 10,
        "ingredients": ["rice", "eggs", "onion"],
        "instructions": ["Chop ingredients", "Cook everything"],
    }

    result = practical_feasibility_score(recipe)

    assert result["score"] is not None
    assert 0 <= result["score"] <= 100
    assert result["coverage"] == 1.0
    assert result["warnings"] == []


def test_missing_information():
    recipe = {
        "total_time": None,
        "ingredients": [],
        "instructions": [],
    }

    result = practical_feasibility_score(recipe)

    assert result["score"] is None
    assert result["coverage"] == 0.0
    assert len(result["warnings"]) > 0


def test_negative_time_is_not_used():
    recipe = {
        "total_time": -10,
        "prep_time": 15,
        "cook_time": 20,
        "ingredients": ["rice", "beans"],
        "instructions": ["Prepare", "Cook"],
    }

    result = practical_feasibility_score(recipe)

    assert result["score"] is not None
    assert any("estimated" in warning.lower()
               for warning in result["warnings"])


if __name__ == "__main__":
    test_quick_recipe()
    test_missing_information()
    test_negative_time_is_not_used()
    print("All practical feasibility tests passed.")