
def cuisine_compatibility_score(recipe, requested_cuisine, related_cuisines=None):
    """
    Score recipe cuisine compatibility from 0 to 100.

    Uses existing cuisine metadata only.
    Related cuisine mappings must be explicitly supplied.

    Accepts either a Recipe dataclass or a dictionary.
    """

    def get_value(name, default=None):
        if isinstance(recipe, dict):
            return recipe.get(name, default)
        return getattr(recipe, name, default)

    recipe_cuisine = get_value("cuisine")

    warnings = []

    if not isinstance(requested_cuisine, str) or not requested_cuisine.strip():
        return {
            "score": None,
            "category": "unknown",
            "recipe_cuisine": recipe_cuisine,
            "requested_cuisine": requested_cuisine,
            "warnings": ["Requested cuisine was not provided."]
        }

    if not isinstance(recipe_cuisine, str) or not recipe_cuisine.strip():
        return {
            "score": None,
            "category": "unknown",
            "recipe_cuisine": recipe_cuisine,
            "requested_cuisine": requested_cuisine,
            "warnings": ["Recipe cuisine metadata is missing."]
        }

    requested = requested_cuisine.strip().casefold()
    actual = recipe_cuisine.strip().casefold()

    if requested == actual:
        score = 100
        category = "exact_match"

    else:
        related = related_cuisines or {}
        related_values = related.get(requested, [])
        related_normalized = {
            value.strip().casefold()
            for value in related_values
            if isinstance(value, str)
        }

        if actual in related_normalized:
            score = 75
            category = "related_match"
        else:
            score = 0
            category = "no_match"
            warnings.append(
                "No exact or explicitly configured related cuisine match."
            )

    return {
        "score": score,
        "category": category,
        "recipe_cuisine": recipe_cuisine,
        "requested_cuisine": requested_cuisine,
        "warnings": warnings
    }