
# The label is direct evidence of cuisine and ingredients indirect, so labelled recipes keep their label score
# and only unlabelled recipes are scored on ingredients, up to 50: they fill gaps between a correct label (100)
# and a wrong one (0) without displacing labelled matches. Blending label and ingredients for every recipe was
# tried and lowered the share of top-10 recipes in the patient's cuisine from 0.998 to 0.865.
UNLABELLED_CAP = 0.5
UNLABELLED = {"other", "none", "nan"}


def cuisine_compatibility_score(recipe, requested_cuisine, related_cuisines=None, typicality=None):
    """
    Score recipe cuisine compatibility from 0 to 100.

    Without `typicality` the score uses the cuisine label only: 100 for an exact
    match, 75 for an explicitly configured related cuisine, 0 otherwise.
    With `typicality` (0-100, how typical the recipe's ingredient pairs are of
    the requested cuisine; cuisine_typicality.py), a recipe without a specific
    label ("Other" in RecipeDB) is judged on its ingredients, up to 50; labelled
    recipes keep their label score and report the typicality alongside it.

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

    unlabelled = not isinstance(recipe_cuisine, str) or recipe_cuisine.strip().casefold() in UNLABELLED | {""}
    if unlabelled and typicality is not None:
        return {
            "score": round(UNLABELLED_CAP * typicality, 2),
            "category": "ingredients_only",
            "recipe_cuisine": recipe_cuisine,
            "requested_cuisine": requested_cuisine,
            "typicality": typicality,
            "warnings": ["No specific cuisine label; scored on ingredient co-occurrence only."]
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
        "typicality": typicality,
        "warnings": warnings
    }