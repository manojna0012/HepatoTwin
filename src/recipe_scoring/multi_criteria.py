"""Layer 4 multi-criteria recipe ranking (proposal section 7.2).

Each candidate recipe gets three criterion scores (0-100): health relevance
against the evidence-linked criteria, cuisine compatibility, and practical
feasibility. The overall score is their weighted mean with explicit weights;
criteria that cannot be scored (for example, no cuisine requested) are left
out and the remaining weights renormalised. A recipe whose health relevance
cannot be scored is not ranked: it is reported as insufficient data, or held
for review when its nutrition is implausible. Every ranked recipe carries its
criterion scores, nutrient values, reference values and evidence rows.

With health_gate (the default), recipes that meet every evidence-linked
reference value rank ahead of those that do not, and the weighted score
orders recipes within each group. Without it, a recipe could trade health
for cuisine or convenience: on 300 held-out participants the weighted-only
top 10 met all three criteria 75% of the time, the gated top 10 90%
(src/evaluate_ranking.py).
"""

import re

from src.recipe_scoring.cuisine_compatibility import cuisine_compatibility_score
from src.recipe_scoring.health_alignment import health_alignment_score
from src.recipe_scoring.practical_feasibility import practical_feasibility_score
from src.recipe_scoring.recipe_criteria import derived_metrics
from src.recipes.nutrition_quality import nutrition_issues

DEFAULT_WEIGHTS = {"health": 0.6, "cuisine": 0.2, "feasibility": 0.2}

# Categories that are not main meals (or not food); matched as lower-case substrings.
NON_MEAL_KEYWORDS = ("dessert", "beverage", "sauce", "dressing", "spread", "smoothie",
                     "candy", "cookie", "cocktail", "punch", "shake", "jam", "jelly",
                     "frosting", "icing", "condiment", "marinade", "frozen", "household",
                     "bath", "beauty", "stock", "mixer", "chutney", "dip")
# Below this energy a serving is a side, snack or seasoning rather than a meal.
# A data screen, not a dietary threshold; lower "energy is better" would
# otherwise reward spice mixes and condiments.
MIN_MEAL_KCAL = 200
# Whole-batch seasonings and condiments listed as one serving pass the energy
# screen, so titles are checked too (whole words).
NON_MEAL_TITLE = re.compile(
    r"\b(seasoning|spice (mix|blend)|powder|rub|bread ?crumbs|marinade|dressing|sauce|"
    r"salsa|dip|syrup|scrub|soap|cleaner|mix)\b", re.IGNORECASE)
DIET_ALLOWS = {
    "vegan": {"vegan"},
    "vegetarian": {"vegetarian", "vegan"},
    "pescetarian": {"pescetarian", "vegetarian", "vegan"},
}


def _get(recipe, name, default=None):
    if isinstance(recipe, dict):
        return recipe.get(name, default)
    return getattr(recipe, name, default)


def is_candidate(recipe, diet=None, meals_only=True):
    """True when the recipe is a main meal that fits the dietary preference."""
    category = str(_get(recipe, "category") or "").casefold()
    if meals_only:
        if any(word in category for word in NON_MEAL_KEYWORDS):
            return False
        if NON_MEAL_TITLE.search(str(_get(recipe, "title") or "")):
            return False
        try:
            calories = float((_get(recipe, "nutrition") or {}).get("calories"))
        except (TypeError, ValueError):
            calories = None
        if calories is not None and calories < MIN_MEAL_KCAL:
            return False
    if diet:
        if diet not in DIET_ALLOWS:
            raise ValueError(f"Unsupported diet preference: {diet}")
        # About 1,000 RecipeDB recipes have conflicting diet and diets tags
        # (e.g. a prawn dish tagged vegan and pescatarian), so every tag
        # must be allowed.
        tags = {str(t).casefold().replace("pescatarian", "pescetarian")
                for t in list(_get(recipe, "diets") or []) + [_get(recipe, "diet")] if t}
        if not tags or not tags <= DIET_ALLOWS[diet]:
            return False
    return True


def score_components(recipe, objectives, requested_cuisine=None, related_cuisines=None):
    """Criterion scores and the details needed to explain them."""
    nutrition = dict(_get(recipe, "nutrition") or {})
    result = {"recipe_id": _get(recipe, "recipe_id"), "title": _get(recipe, "title"),
              "cuisine": _get(recipe, "cuisine"), "category": _get(recipe, "category")}

    issues = nutrition_issues(nutrition)
    if issues:
        return {**result, "status": "held_for_review", "scores": {}, "warnings": issues}

    nutrition.update(derived_metrics(nutrition))
    health = health_alignment_score({"nutrition": nutrition}, objectives)
    if health["score"] is None:
        return {**result, "status": "insufficient_data", "scores": {},
                "warnings": health["warnings"]}

    scores = {"health": health["score"]}
    feasibility = practical_feasibility_score(recipe)
    if feasibility["score"] is not None:
        scores["feasibility"] = feasibility["score"]
    if requested_cuisine:
        cuisine = cuisine_compatibility_score(recipe, requested_cuisine, related_cuisines)
        if cuisine["score"] is not None:
            scores["cuisine"] = cuisine["score"]

    criteria = [{
        "priority_id": o["priority_id"], "priority": o["priority"], "label": o["label"],
        "value": None if nutrition.get(o["nutrient"]) is None else round(float(nutrition[o["nutrient"]]), 1),
        "reference": f'{"<=" if o["direction"] == "lower" else ">="} {o["target"]} ({o["reference"]})',
        "score": health["components"].get(o["nutrient"]),
        "evidence": [f'{e["evidence_id"]} [{e["evidence_strength"]}]' for e in o.get("evidence", [])],
    } for o in objectives]
    return {**result, "status": "ranked", "scores": scores, "health_coverage": health["coverage"],
            "meets_all_references": all(c["score"] == 100 for c in criteria),
            "criteria": criteria, "warnings": health["warnings"] + feasibility["warnings"]}


def overall_score(scores, weights=DEFAULT_WEIGHTS):
    used = {k: w for k, w in weights.items() if k in scores and w > 0}
    total = sum(used.values())
    return round(sum(scores[k] * w for k, w in used.items()) / total, 2) if total else None


def rank_recipes(recipes, objectives, weights=DEFAULT_WEIGHTS, requested_cuisine=None,
                 related_cuisines=None, diet=None, meals_only=True, health_gate=True):
    """Return (ranked, not_ranked); ranked is sorted by overall score.

    With health_gate, recipes meeting every reference value come first.
    """
    if not objectives:
        raise ValueError("At least one recipe criterion is required.")
    if any(w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError("Weights must be non-negative and not all zero.")
    ranked, not_ranked = [], []
    for recipe in recipes:
        if not is_candidate(recipe, diet, meals_only):
            continue
        item = score_components(recipe, objectives, requested_cuisine, related_cuisines)
        if item["status"] != "ranked":
            not_ranked.append(item)
            continue
        item["overall"] = overall_score(item["scores"], weights)
        item["weights_used"] = {k: w for k, w in weights.items() if k in item["scores"]}
        ranked.append(item)
    ranked.sort(key=lambda item: (-(health_gate and item["meets_all_references"]), -item["overall"],
                                  -item["scores"]["health"], str(item["recipe_id"])))
    for position, item in enumerate(ranked, 1):
        item["rank"] = position
    return ranked, not_ranked


def explain(item):
    """Plain-text explanation of one ranked recipe (proposal section 7.3)."""
    met = "meets every reference value" if item["meets_all_references"] else "misses a reference value"
    lines = [f'#{item["rank"]} {item["title"]} ({item["cuisine"]}; {item["category"]}) '
             f'- overall {item["overall"]:.1f}/100; {met}',
             '   ' + ', '.join(f'{k} {v:.0f}' for k, v in item["scores"].items())]
    for c in item["criteria"]:
        value = "n/a" if c["value"] is None else c["value"]
        score = "not scored" if c["score"] is None else f'{c["score"]:.0f}/100'
        lines.append(f'   - {c["label"]}: {value} (reference {c["reference"]}) -> {score}; '
                     f'{c["priority"]} priority "{c["priority_id"]}"; evidence {", ".join(c["evidence"])}')
    return "\n".join(lines)
