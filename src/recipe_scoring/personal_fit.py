"""Personal fit: how close a recipe is to what the patient already eats and likes (Layer 4).

The health gate (multi_criteria.rank_recipes) puts every recipe that meets the evidence-linked references
first. Those recipes are healthy for almost everyone, so without a personal term two patients of the same
cuisine shared 93% of their top-10 recipes (src/evaluate_ranking.py, 2 November 2026) and test patients
without food answers were all offered the same few dishes (team review, 8 November 2026). Personal fit
re-orders only the recipes that pass the gate, so it never trades health for familiarity.

Distance from what the patient eats now has the two parts the style ladder uses (style_ladder.closeness),
averaged when both are known:

- diet: for each of the patient's priorities with a habitual measure (meal energy, saturated fat and sugar
  as a share of energy, fibre per 1,000 kcal), the gap between the recipe and the patient's 24-hour recall,
  in units of the criterion's tolerance (scenario_engine's shift), averaged and capped at 1;
- dishes: 1 - the largest ingredient overlap (Jaccard) with a favourite dish or the liked ingredients.

personal_fit = 100 x (1 - distance). Without a reliable recall and without favourites it is unknown, and
the ranking is the gated ranking as before. Its weight (DEFAULT_WEIGHTS['personal_fit']) sits beside
cuisine and feasibility.
"""
import numpy as np

from src.recipe_scoring.meal_plan import ingredient_set, jaccard
from src.recipe_scoring.multi_criteria import DEFAULT_WEIGHTS, gated_health, overall_score
from src.recipe_scoring.recipe_criteria import CRITERIA, derived_metrics
from src.recipe_scoring.scenario_engine import PATHWAYS, _usual

METRICS = ('calories', 'saturated_fat_pct_energy', 'sugar_pct_energy', 'fiber_per_1000kcal')
HABIT_TO_METRIC = {'meal_energy_kcal': 'calories'}


def recipe_metrics(recipes):
    """{measure: array over recipes} of the measures the diet distance compares."""
    rows = []
    for recipe in recipes:
        nutrition = dict(recipe.get('nutrition') or {})
        rows.append({'calories': nutrition.get('calories'), **derived_metrics(nutrition)})
    return {m: np.array([np.nan if r.get(m) is None else float(r[m]) for r in rows]) for m in METRICS}


def diet_distance(metrics, priorities, patient):
    """Distance of every recipe from the patient's recall on their priorities (0 = the same), or None."""
    usual = _usual(patient)
    if not usual:
        return None
    gaps = []
    for item in priorities:
        criterion, pathway = CRITERIA.get(item['id']), PATHWAYS.get(item['id'])
        if criterion is None or pathway is None or usual.get(pathway['habitual']) is None:
            continue
        values = metrics[HABIT_TO_METRIC.get(pathway['habitual'], criterion['nutrient'])]
        gaps.append(np.abs(values - usual[pathway['habitual']]) / criterion['tolerance'])
    if not gaps:
        return None
    with np.errstate(invalid='ignore'):
        return np.minimum(np.nanmean(np.vstack(gaps), axis=0), 1.0)


def dish_distance(recipes, favourites):
    if not favourites:
        return None
    sets = [ingredient_set(f) for f in favourites]
    return np.array([1 - max(jaccard(ingredient_set(r), s) for s in sets) for r in recipes])


def fit_scores(recipes, priorities, patient=None, favourites=(), metrics=None):
    """personal_fit (0-100) for every recipe, or None when nothing about the patient's food is known."""
    parts = [d for d in (diet_distance(metrics if metrics is not None else recipe_metrics(recipes), priorities, patient),
                         dish_distance(recipes, favourites)) if d is not None]
    if not parts:
        return None
    with np.errstate(invalid='ignore'):
        distance = np.nanmean(np.vstack(parts), axis=0)
    return np.round(100 * (1 - distance), 1)


def personalise(ranked, recipes, priorities, patient=None, favourites=(), weights=DEFAULT_WEIGHTS):
    """The ranked list with personal fit added to the recipes that meet every reference, re-ordered."""
    gated = [i for i in ranked if i.get('meets_all_references')]
    if not gated:
        return ranked
    by_id = {str(r.get('recipe_id')): r for r in recipes}
    pool = [by_id[str(i['recipe_id'])] for i in gated]
    fit = fit_scores(pool, priorities, patient, favourites)
    if fit is None:
        return ranked
    for item, value in zip(gated, fit):
        if np.isnan(value):
            continue
        item['scores'] = {**item['scores'], 'personal_fit': float(value)}
        scores = {**item['scores'], 'health': gated_health(item['health_margin'])}
        item['overall'] = overall_score(scores, weights)
        item['weights_used'] = {k: w for k, w in weights.items() if k in item['scores']}
    rest = [i for i in ranked if not i.get('meets_all_references')]
    gated.sort(key=lambda i: (-i['overall'], -i['scores']['health'], str(i['recipe_id'])))
    out = gated + rest
    for position, item in enumerate(out, 1):
        item['rank'] = position
    return out
