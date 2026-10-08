
from cuisine_compatibility import cuisine_compatibility_score


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


if __name__ == "__main__":
    test_exact_match()
    test_related_match()
    test_no_match()
    test_missing_cuisine()
    print("All cuisine compatibility tests passed.")