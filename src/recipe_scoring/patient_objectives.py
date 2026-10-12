
PRIORITY_WEIGHTS = {
    "low": 1.0,
    "medium": 2.0,
    "high": 3.0,
}


def build_patient_objectives(patient_profile):
    """Convert patient priorities into health-scoring objectives."""

    if not isinstance(patient_profile, dict):
        raise ValueError("Patient profile must be a dictionary.")

    priorities = patient_profile.get("priorities")

    if not isinstance(priorities, list) or not priorities:
        raise ValueError("Patient must have at least one priority.")

    objectives = []
    seen_nutrients = set()

    for item in priorities:
        if not isinstance(item, dict):
            raise ValueError("Each priority must be a dictionary.")

        nutrient = item.get("nutrient")
        direction = item.get("direction")
        priority = item.get("priority")
        from src.recipe_scoring.health_alignment import DEFAULT_NUTRIENTS
        if nutrient not in DEFAULT_NUTRIENTS:
            raise ValueError(f"Unsupported nutrient: {nutrient}")

        if nutrient in seen_nutrients:
            raise ValueError(f"Duplicate nutrient objective: {nutrient}")

        if direction not in {"lower", "higher"}:
            raise ValueError(f"Invalid direction for {nutrient}.")

        if priority not in PRIORITY_WEIGHTS:
            raise ValueError(f"Invalid priority for {nutrient}.")

        try:
            target = float(item["target"])
            tolerance = float(item["tolerance"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid target or tolerance for {nutrient}.")

        if target < 0 or tolerance <= 0:
            raise ValueError(f"Invalid target or tolerance for {nutrient}.")

        objectives.append({
            "nutrient": nutrient,
            "direction": direction,
            "target": target,
            "tolerance": tolerance,
            "weight": PRIORITY_WEIGHTS[priority],
        })

        seen_nutrients.add(nutrient)

    return objectives
