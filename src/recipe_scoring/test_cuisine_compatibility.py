
from src.recipe_scoring.cuisine_compatibility import cuisine_compatibility_score


def test_exact_match():
    result = cuisine_compatibility_score(
        {"cuisine": "Indian"}, "indian"
    )
    assert result["score"] == 100
    assert result["category"] == "exact_match"


def test_related_match():
    result = cuisine_compatibility_score(
        {"cuisine": "Punjabi"},
        "Indian",
        related_cuisines={"indian": ["Punjabi"]},
    )
    assert result["score"] == 75
    assert result["category"] == "related_match"


def test_no_match():
    result = cuisine_compatibility_score(
        {"cuisine": "Italian"}, "Indian"
    )
    assert result["score"] == 0
    assert result["category"] == "no_match"


def test_missing_cuisine():
    result = cuisine_compatibility_score(
        {"cuisine": None}, "Indian"
    )
    assert result["score"] is None
    assert result["category"] == "unknown"


def test_labelled_recipes_keep_their_label_score():
    exact = cuisine_compatibility_score({"cuisine": "Asian"}, "Asian", typicality=20)
    assert exact["score"] == 100 and exact["typicality"] == 20
    assert cuisine_compatibility_score({"cuisine": "Mexican"}, "Asian", typicality=90)["score"] == 0


def test_unlabelled_recipe_is_scored_on_ingredients_up_to_50():
    result = cuisine_compatibility_score({"cuisine": "Other"}, "Asian", typicality=100)
    assert result["score"] == 50 and result["category"] == "ingredients_only"
    assert cuisine_compatibility_score({"cuisine": "Other"}, "Asian")["score"] == 0   # label only


def test_cooccurrence_separates_cuisines():
    from src.recipe_scoring.cuisine_typicality import CuisineTypicality
    asian = [{"cuisine": "Asian", "ingredients": ["soy sauce", "ginger", "garlic", "rice"]}] * 20
    mexican = [{"cuisine": "Mexican", "ingredients": ["tortilla", "cumin", "salsa", "beans"]}] * 20
    model = CuisineTypicality(asian + mexican)
    stir_fry = {"cuisine": "Other", "ingredients": ["soy sauce", "ginger", "garlic"]}
    assert model.score(stir_fry, "asian") == 100 and model.score(stir_fry, "Mexican") == 0
    assert model.score(stir_fry, "Klingon") is None


if __name__ == "__main__":
    test_exact_match()
    test_related_match()
    test_no_match()
    test_missing_cuisine()
    print("All cuisine compatibility tests passed.")