"""Map Layer 3 priorities to recipe-level criteria (proposal section 6.1).

Each priority is linked to one recipe-composition measure with a reference
value from public dietary guidance. Reference values describe the recipe,
not the patient: no individual intake target is inferred. The priority level
sets the criterion weight; the evidence rows travel with the criterion so a
ranked recipe can be explained.

Reference values:
- Energy per serving <= 600 kcal for a main meal: Public Health England,
  "400-600-600" calorie guidance, 2018.
- Saturated fat <= 10% of energy: WHO guideline on saturated and trans fatty
  acid intake, 2023; Dietary Guidelines for Americans 2020-2025.
- Sugars <= 10% of energy: WHO guideline on sugars intake, 2015 (free
  sugars). RecipeDB reports total sugars, so fruit and milk sugars count too.
- Fibre >= 14 g per 1,000 kcal: Dietary Guidelines for Americans 2020-2025.

The score for a criterion is 100 at or past the reference value and falls
linearly to 0 over the tolerance. For composition criteria ("margin"), the
margin is how far past the reference a recipe goes, as a share of the same
tolerance (0-100); energy has no margin, since a meal far below 600 kcal is
not better for that reason.
"""

from src.recipe_scoring.patient_objectives import PRIORITY_WEIGHTS

CRITERIA = {
    "weight_management": {
        "nutrient": "calories", "direction": "lower", "target": 600, "tolerance": 400,
        "label": "Energy per serving (kcal)",
        "reference": "Public Health England 400-600-600 meal guidance, 2018",
    },
    "limit_saturated_fat": {
        "nutrient": "saturated_fat_pct_energy", "direction": "lower", "target": 10, "tolerance": 10,
        "label": "Saturated fat (% of energy)", "margin": True,
        "reference": "WHO saturated fatty acid guideline, 2023; DGA 2020-2025",
    },
    "limit_excess_sugar": {
        "nutrient": "sugar_pct_energy", "direction": "lower", "target": 10, "tolerance": 15,
        "label": "Total sugars (% of energy)", "margin": True,
        "reference": "WHO sugars guideline, 2015 (free sugars; total sugars used here)",
    },
    "diet_quality": {
        "nutrient": "fiber_per_1000kcal", "direction": "higher", "target": 14, "tolerance": 14,
        "label": "Fibre (g per 1,000 kcal)", "margin": True,
        "reference": "Dietary Guidelines for Americans 2020-2025",
    },
}
DERIVED_NUTRIENTS = {"saturated_fat_pct_energy", "sugar_pct_energy", "fiber_per_1000kcal"}


def derived_metrics(nutrition):
    """Energy-relative composition measures; None when energy is unusable."""
    def number(name):
        try:
            value = float(nutrition.get(name))
        except (TypeError, ValueError, AttributeError):
            return None
        return value if value == value and value >= 0 else None

    calories = number("calories")
    if not calories:
        return {name: None for name in DERIVED_NUTRIENTS}
    saturated, sugar, fiber = number("saturated_fat"), number("sugar"), number("fiber")
    return {
        "saturated_fat_pct_energy": None if saturated is None else 900 * saturated / calories,
        "sugar_pct_energy": None if sugar is None else 400 * sugar / calories,
        "fiber_per_1000kcal": None if fiber is None else 1000 * fiber / calories,
    }


def margin_score(nutrition, objectives):
    """Priority-weighted margin past the reference values (0-100), or None.

    None when no objective has a margin or a needed value is missing.
    """
    total = weight = 0.0
    for o in objectives:
        if not o.get("margin"):
            continue
        value = nutrition.get(o["nutrient"])
        if value is None:
            return None
        past = o["target"] - value if o["direction"] == "lower" else value - o["target"]
        total += o["weight"] * 100 * min(max(past / o["tolerance"], 0.0), 1.0)
        weight += o["weight"]
    return round(total / weight, 2) if weight else None


def criteria_for(priorities):
    """Return health-alignment objectives for the given Layer 3 priorities.

    Priorities without a recipe criterion are skipped and listed separately.
    """
    objectives, unmapped = [], []
    for item in priorities:
        criterion = CRITERIA.get(item["id"])
        if criterion is None:
            unmapped.append(item["id"])
            continue
        objectives.append({
            **criterion,
            "priority_id": item["id"],
            "priority": item["priority"],
            "weight": PRIORITY_WEIGHTS[item["priority"]],
            "evidence": item.get("supporting_evidence", []),
        })
    return objectives, unmapped
