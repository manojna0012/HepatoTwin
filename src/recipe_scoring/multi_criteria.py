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
orders recipes within each group. Inside the gated group every health score
is 100, so the patient's priority levels would no longer matter; there half
of the health component is instead the priority-weighted margin past the
reference values (recipe_criteria.margin_score), so a patient whose
saturated-fat priority is high is offered recipes further below the
saturated-fat reference. MARGIN_SHARE trades personalisation for
practicality: on the evaluation, a share of 0 gave 8 distinct top-10 lists
for 300 participants with 90% meeting all three criteria, 0.5 gave 33 lists
with 80%, and 1 gave 40 lists with 69% (src/evaluate_ranking.py).

Without the gate, a recipe could trade health for cuisine or convenience: the
weighted-only top 10 met every health reference 75% of the time, the gated
top 10 always did.
"""

import re

from src.recipe_scoring.cuisine_compatibility import cuisine_compatibility_score
from src.recipe_scoring.health_alignment import health_alignment_score
from src.recipe_scoring.practical_feasibility import practical_feasibility_score
from src.recipe_scoring.recipe_criteria import derived_metrics, margin_score
from src.recipes.nutrition_quality import nutrition_issues

# Weights are renormalised over the criteria a recipe has, so these act as
# 0.545 / 0.182 / 0.182 / 0.091.
DEFAULT_WEIGHTS = {"health": 0.6, "cuisine": 0.2, "feasibility": 0.2, "acceptability": 0.1}
# Acceptability (proposal section 17.12): RecipeDB ratings shrunk towards the
# corpus mean by a pseudo-count, so 5 stars from 3 raters does not beat 4.9
# from 3,000. Ratings cluster near 5, so 4.5-5.0 maps to 0-100. Weight 0.1
# raised the top 10 meeting all three criteria from 0.80 to 0.83 on 300
# held-out participants (src/evaluate_ranking.py).
RATING_PRIOR_MEAN, RATING_PRIOR_COUNT = 4.91, 20
ACCEPTABILITY_RANGE = (4.5, 5.0)
MARGIN_SHARE = 0.5

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


def acceptability_score(recipe):
    """0-100 from the shrunk rating, or None without ratings."""
    try:
        rating = float(_get(recipe, "ratings"))
        count = float(_get(recipe, "ratings_count"))
    except (TypeError, ValueError):
        return None
    if count <= 0 or rating != rating:
        return None
    shrunk = (count * rating + RATING_PRIOR_COUNT * RATING_PRIOR_MEAN) / (count + RATING_PRIOR_COUNT)
    low, high = ACCEPTABILITY_RANGE
    return round(100 * min(max((shrunk - low) / (high - low), 0.0), 1.0), 2)


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
    acceptability = acceptability_score(recipe)
    if acceptability is not None:
        scores["acceptability"] = acceptability
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
    meets_all = all(c["score"] == 100 for c in criteria)
    return {**result, "status": "ranked", "scores": scores, "health_coverage": health["coverage"],
            "meets_all_references": meets_all,
            "health_margin": margin_score(nutrition, objectives) if meets_all else None,
            "criteria": criteria, "warnings": health["warnings"] + feasibility["warnings"]}


def gated_health(margin):
    """Health component for a recipe meeting every reference (see module docstring)."""
    return 100 - MARGIN_SHARE * (100 - margin)


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
        scores = item["scores"]
        if health_gate and item["health_margin"] is not None:
            scores = {**scores, "health": gated_health(item["health_margin"])}
        item["overall"] = overall_score(scores, weights)
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
             '   ' + ', '.join(f'{k} {v:.0f}' for k, v in item["scores"].items())
             + ('' if item.get("health_margin") is None
                else f'; margin past references {item["health_margin"]:.0f}')]
    for c in item["criteria"]:
        value = "n/a" if c["value"] is None else c["value"]
        score = "not scored" if c["score"] is None else f'{c["score"]:.0f}/100'
        lines.append(f'   - {c["label"]}: {value} (reference {c["reference"]}) -> {score}; '
                     f'{c["priority"]} priority "{c["priority_id"]}"; evidence {", ".join(c["evidence"])}')
    return "\n".join(lines)
