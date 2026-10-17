from src.recipes.nutrition_quality import nutrition_issues


def test_plausible_serving_has_no_issues():
    nutrition = {"calories": 520, "protein": 50, "carbohydrates": 23,
                 "fat": 24.5, "sodium": 1573}
    assert nutrition_issues(nutrition) == []


def test_implausible_servings_are_flagged():
    assert nutrition_issues({"calories": 0})
    assert nutrition_issues({"calories": 38680})
    assert nutrition_issues({"calories": 400, "sodium": 226783})
    assert nutrition_issues({"calories": 300, "protein": 200,
                             "carbohydrates": 100, "fat": 50})


def test_missing_values_are_left_to_the_scorer():
    assert nutrition_issues({"sodium": 400}) == []
    assert nutrition_issues(None) == []
