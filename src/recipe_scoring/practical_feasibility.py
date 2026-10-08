
def practical_feasibility_score(recipe, max_total_time=60,
                                max_ingredients=15,
                                max_instructions=12):
    """
    Score recipe practicality from 0 to 100.

    Time, ingredient and instruction thresholds are configurable
    screening preferences, not clinical standards.

    Accepts either a Recipe dataclass or a dictionary.
    Missing information is excluded from scoring and reported.
    """

    def get_value(name, default=None):
        if isinstance(recipe, dict):
            return recipe.get(name, default)
        return getattr(recipe, name, default)

    components = {}
    warnings = []

    def valid_time(value):
        try:
            number = float(value)
            if number >= 0 and number < float("inf"):
                return number
        except (TypeError, ValueError):
            pass
        return None

    # Prefer total time; otherwise use prep + cook time.
    total_time = valid_time(get_value("total_time"))
    if total_time is None:
        prep = valid_time(get_value("prep_time"))
        cook = valid_time(get_value("cook_time"))
        if prep is not None and cook is not None:
            total_time = prep + cook
            warnings.append("Total time estimated from prep and cook time.")

    if total_time is not None:
        components["time"] = max(
            0.0, min(100.0, 100 * (1 - total_time / max_total_time))
        )
    else:
        warnings.append("Preparation time unavailable.")

    ingredients = get_value("ingredients")
    if isinstance(ingredients, list) and len(ingredients) > 0:
        components["ingredient_count"] = max(
            0.0,
            min(100.0, 100 * (1 - (len(ingredients) - 1)
                              / max(1, max_ingredients - 1)))
        )
    else:
        warnings.append("Ingredient list unavailable or empty.")

    instructions = get_value("instructions")
    if isinstance(instructions, list) and len(instructions) > 0:
        components["instruction_count"] = max(
            0.0,
            min(100.0, 100 * (1 - (len(instructions) - 1)
                              / max(1, max_instructions - 1)))
        )
    else:
        warnings.append("Instructions unavailable or empty.")

    if not components:
        return {
            "score": None,
            "components": {},
            "coverage": 0.0,
            "warnings": warnings + ["Insufficient data to score recipe."]
        }

    # Equal weights among available components only.
    score = sum(components.values()) / len(components)

    return {
        "score": round(score, 2),
        "components": {
            key: round(value, 2)
            for key, value in components.items()
        },
        "coverage": round(len(components) / 3, 2),
        "warnings": warnings
    }