import re

# Techniques and equipment read from the instructions (RecipeDB's API gives no process or utensil
# fields). They scale the score down rather than averaging in as a fourth component, so a recipe needing
# none keeps its time, ingredient and step score: each demanding technique removes 25% and each piece of
# special equipment 10%. Design choices for screening, not validated difficulty weights. Averaging them in
# as a fourth component was tried and lowered the share of practical top-10 recipes from 0.832 to 0.793.
DEMANDING_TECHNIQUES = {
    "deep-frying": r"deep[- ]?fr",
    "kneading or proving dough": r"knead|let (it |the dough )?rise|until doubled",
    "overnight or long chilling": r"overnight|refrigerate (for )?(at least )?\d{2,} hours",
    "marinating": r"marinat",
    "sugar or chocolate work with a thermometer": r"thermometer|temper",
    "flambeing": r"flamb",
}
SPECIAL_EQUIPMENT = {
    "stand or electric mixer": r"stand mixer|electric mixer",
    "food processor": r"food processor",
    "blender": r"blender",
    "slow cooker": r"slow cooker|crock ?pot",
    "pressure cooker": r"pressure cooker|instant pot",
    "grill": r"grill",
    "wok": r"wok",
    "deep fryer": r"deep fryer",
    "smoker": r"smoker",
    "mandoline": r"mandoline",
}
TECHNIQUE_COST, EQUIPMENT_COST = 25, 10


def techniques_and_equipment(instructions):
    """Demanding techniques and special equipment named in the instructions."""
    text = " ".join(str(step) for step in instructions or []).casefold()
    return ([name for name, pattern in DEMANDING_TECHNIQUES.items() if re.search(pattern, text)],
            [name for name, pattern in SPECIAL_EQUIPMENT.items() if re.search(pattern, text)])



def practical_feasibility_score(recipe, max_total_time=60,
                                max_ingredients=15,
                                max_instructions=12):
    """
    Score recipe practicality from 0 to 100.

    Time, ingredient and instruction thresholds are configurable
    screening preferences, not clinical standards.

    Accepts either a Recipe dataclass or a dictionary.
    Missing information is excluded from scoring and reported.
    Components: total time, ingredient count and instruction count, scaled
    down for demanding techniques and special equipment named in the instructions.
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

    techniques, equipment = techniques_and_equipment(instructions) if isinstance(instructions, list) else ([], [])
    technique_factor = max(0.0, 1 - (TECHNIQUE_COST * len(techniques) + EQUIPMENT_COST * len(equipment)) / 100)

    if not components:
        return {
            "score": None,
            "components": {},
            "coverage": 0.0,
            "warnings": warnings + ["Insufficient data to score recipe."]
        }

    # Equal weights among available components only, then the technique and equipment penalty.
    score = sum(components.values()) / len(components) * technique_factor

    return {
        "score": round(score, 2),
        "components": {
            key: round(value, 2)
            for key, value in components.items()
        },
        "coverage": round(len(components) / 3, 2),
        "technique_factor": round(technique_factor, 2),
        "techniques": techniques,
        "equipment": equipment,
        "warnings": warnings
    }