
DEFAULT_NUTRIENTS = {
    "calories",
    "protein",
    "carbohydrates",
    "fat",
    "saturated_fat",
    "cholesterol",
    "fiber",
    "sugar",
    "sodium",
}


def health_alignment_score(recipe, objectives, extreme_limits=None):
    """
    Score recipe alignment with explicitly supplied nutritional objectives.

    Each objective must define:
      - nutrient: key in recipe nutrition
      - direction: "lower" or "higher"
      - target: preferred value
      - weight: positive importance weight
      - tolerance: non-negative distance over which score declines to zero

    Returns a score from 0 to 100, or None if no objectives can be scored.

    This is a configurable heuristic, not a clinically validated score.
    Nutrient units and serving basis must be consistent before use.
    """

    def get_value(name, default=None):
        if isinstance(recipe, dict):
            return recipe.get(name, default)
        return getattr(recipe, name, default)

    nutrition = get_value("nutrition", {})
    if not isinstance(nutrition, dict):
        nutrition = {}

    warnings = []
    components = {}
    weighted_scores = []
    total_weight = 0.0

    if not isinstance(objectives, list) or not objectives:
        return {
            "score": None,
            "components": {},
            "coverage": 0.0,
            "warnings": ["No nutritional objectives were provided."]
        }

    for objective in objectives:
        nutrient = objective.get("nutrient")
        direction = objective.get("direction")
        target = objective.get("target")
        tolerance = objective.get("tolerance")
        weight = objective.get("weight")

        if nutrient not in DEFAULT_NUTRIENTS:
            warnings.append(f"Unsupported nutrient objective: {nutrient}")
            continue

        try:
            target = float(target)
            tolerance = float(tolerance)
            weight = float(weight)
            value = float(nutrition[nutrient])
        except (TypeError, ValueError, KeyError):
            warnings.append(f"Missing or invalid value for {nutrient}.")
            continue

        if (
            tolerance <= 0
            or weight <= 0
            or target < 0
            or value < 0
            or direction not in {"lower", "higher"}
        ):
            warnings.append(f"Invalid objective or value for {nutrient}.")
            continue

        if extreme_limits and nutrient in extreme_limits:
            if value > extreme_limits[nutrient]:
                warnings.append(
                    f"{nutrient} exceeds the configured review limit."
                )
                continue

        distance = (
            max(0.0, value - target)
            if direction == "lower"
            else max(0.0, target - value)
        )

        score = max(0.0, 100.0 * (1.0 - distance / tolerance))

        components[nutrient] = round(score, 2)
        weighted_scores.append(score * weight)
        total_weight += weight

    score = (
        round(sum(weighted_scores) / total_weight, 2)
        if total_weight > 0
        else None
    )

    valid_objectives = len(weighted_scores)

    coverage = (
        valid_objectives / len(objectives)
        if objectives else 0.0
    )

    if score is None:
        warnings.append("Insufficient valid nutritional data to score.")

    return {
        "score": score,
        "components": components,
        "coverage": round(coverage, 2),
        "warnings": warnings
    }