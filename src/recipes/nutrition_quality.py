"""Plausibility screen for per-serving recipe nutrition.

About 4% of RecipeDB recipes have values that cannot describe one serving:
zero energy, tens of thousands of kcal, or macronutrients that do not add up
to the stated energy. Those recipes should be held for review rather than
ranked. Limits are data-quality screens, not dietary recommendations.
"""

MAX_CALORIES = 2500        # kcal per serving
MAX_SODIUM = 6000          # mg per serving
MIN_CALORIES_FOR_ENERGY_CHECK = 50
ENERGY_MISMATCH = 0.30     # relative gap between stated and Atwater energy


def nutrition_issues(nutrition):
    """Return a list of reasons the per-serving nutrition is implausible."""
    if not isinstance(nutrition, dict):
        return []

    def number(name):
        try:
            value = float(nutrition.get(name))
        except (TypeError, ValueError):
            return None
        return value if value == value else None  # drop NaN

    # Missing values are left to the scorer, which skips them per nutrient.
    issues = []
    calories = number("calories")
    if calories is None:
        pass
    elif calories <= 0:
        issues.append("Calories are zero or negative.")
    elif calories > MAX_CALORIES:
        issues.append(f"Calories exceed {MAX_CALORIES} kcal per serving.")

    sodium = number("sodium")
    if sodium is not None and sodium > MAX_SODIUM:
        issues.append(f"Sodium exceeds {MAX_SODIUM} mg per serving.")

    macros = [number(n) for n in ("protein", "carbohydrates", "fat")]
    if calories and calories >= MIN_CALORIES_FOR_ENERGY_CHECK and None not in macros:
        protein, carbohydrates, fat = macros
        atwater = 4 * protein + 4 * carbohydrates + 9 * fat
        if abs(atwater - calories) > ENERGY_MISMATCH * calories:
            issues.append("Macronutrients do not match stated calories.")

    return issues
