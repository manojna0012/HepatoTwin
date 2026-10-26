
from src.recipe_scoring.practical_feasibility import practical_feasibility_score


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


def test_demanding_techniques_and_equipment_lower_the_score():
    from src.recipe_scoring.practical_feasibility import techniques_and_equipment
    easy = {"total_time": 30, "ingredients": ["a", "b"], "instructions": ["Chop.", "Saute for 10 minutes."]}
    hard = dict(easy, instructions=["Knead the dough and let it rise until doubled.", "Deep-fry in batches.",
                                    "Mix in a stand mixer."])
    assert techniques_and_equipment(hard["instructions"]) == (
        ["deep-frying", "kneading or proving dough"], ["stand or electric mixer"])
    easy_result, hard_result = practical_feasibility_score(easy), practical_feasibility_score(hard)
    assert easy_result["technique_factor"] == 1.0 and hard_result["technique_factor"] == 0.4
    hard_without_penalty = sum(hard_result["components"].values()) / len(hard_result["components"])
    assert hard_result["score"] == round(0.4 * hard_without_penalty, 2) and hard_result["score"] < easy_result["score"]


if __name__ == "__main__":
    test_quick_recipe()
    test_missing_information()
    test_negative_time_is_not_used()
    print("All practical feasibility tests passed.")