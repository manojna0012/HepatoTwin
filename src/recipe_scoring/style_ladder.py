"""Recipe style ladder from "closer to what you eat now" (Scenario B) to "healthiest" (Scenario A).

Both ends draw on the same safe pool: the first POOL ranked recipes that meet every gating reference
value (multi_criteria.py), so they are already in the patient's cuisine, practical and healthy.

Closeness to what the patient eats now has two parts, averaged when both are known:

- nutrients: the change from the patient's 24-hour recall on their own priorities (the scenario
  engine's shift score; scenario_engine.py), capped at 1;
- dishes: 1 - the largest ingredient overlap (Jaccard, meal_plan.py) with a favourite dish the patient
  picked at sign-up, so "closer" means familiar food, not only similar nutrients.

Without either, closeness is unknown and the ladder falls back to the health order (there is no
"closer to usual" plan to offer).

A step s of STEPS blends the two orders: score = s/STEPS x (health rank) + (1 - s/STEPS) x (closeness
rank), ranks scaled to 0-1. Step 0 is Scenario B's order, step STEPS Scenario A's. The weekly plan
(meal_plan.weekly_plan) is then built from the blended order.

The shopping list adds up one serving of each planned meal, a serving being the part of the dish
that carries RecipeDB's stated energy per serving (as in processing_and_glycaemic_load.py).
"""
from collections import defaultdict

from src.recipe_scoring.ingredient_amounts import ingredient_grams, link_table
from src.recipe_scoring.meal_plan import ingredient_set, jaccard
from src.recipe_scoring.multi_criteria import is_candidate
from src.recipe_scoring.processing_and_glycaemic_load import food_values
from src.recipe_scoring.scenario_engine import evaluate_scenario

POOL = 200
STEPS = 3
STEP_LABELS = ('closest to what you eat now', 'mostly familiar', 'mostly healthiest', 'healthiest')


def _health_key(item):
    return (-(item.get('health_margin') or 0), -item['overall'])


def closeness(recipe, priorities, patient=None, favourites=()):
    """Distance (0 = same) of a recipe from what the patient eats now, or None when unknown."""
    parts = []
    if patient:
        scenario = evaluate_scenario([recipe], priorities, patient)
        if scenario['shift_from_usual'] is not None:          # None without a reliable 24-hour recall
            parts.append(min(scenario['shift_score'], 1.0))
    if favourites:
        own = ingredient_set(recipe)
        parts.append(1 - max(jaccard(own, ingredient_set(f)) for f in favourites))
    return sum(parts) / len(parts) if parts else None


def safe_pool(ranked):
    passing = [i for i in ranked if i.get('meets_all_references')]
    return (passing or list(ranked))[:POOL]


def ladder_order(ranked, recipes, priorities, patient=None, favourites=(), step=STEPS):
    """The safe pool in the order for a ladder step; also whether closeness was known."""
    by_id = {str(r['recipe_id']): r for r in recipes}
    pool = safe_pool(ranked)
    health = {str(i['recipe_id']): n for n, i in enumerate(sorted(pool, key=_health_key))}
    distance = {str(i['recipe_id']): closeness(by_id[str(i['recipe_id'])], priorities, patient, favourites)
                for i in pool}
    known = all(d is not None for d in distance.values()) and pool
    if not known or step >= STEPS:
        return sorted(pool, key=_health_key), bool(known)
    near = {rid: n for n, rid in enumerate(sorted(distance, key=lambda rid: distance[rid]))}
    span, lam = max(len(pool) - 1, 1), step / STEPS
    score = {rid: lam * health[rid] / span + (1 - lam) * near[rid] / span for rid in health}
    return sorted(pool, key=lambda i: score[str(i['recipe_id'])]), True


def for_week(order):
    """The ladder order as items the weekly planner keeps in that order (its diversity step re-ranks
    by 'overall', so the blended position replaces it)."""
    n = max(len(order) - 1, 1)
    return [{**item, 'overall': 100 * (1 - pos / n)} for pos, item in enumerate(order)]


def favourite_choices(recipes, cuisine, k=12):
    """Popular main meals of a cuisine to pick favourites from at sign-up (most rated first)."""
    meals = [r for r in recipes if str(r.get('cuisine', '')).casefold() == str(cuisine).casefold()
             and is_candidate(r)]
    meals.sort(key=lambda r: -(r.get('ratings_count') or 0))
    return [{'recipe_id': r['recipe_id'], 'title': r['title']} for r in meals[:k]]


def shopping_list(meals):
    """Ingredients for one serving of each meal, grouped by food group, largest first."""
    names = sorted({i.get('name_lc', '') for m in meals for i in m.get('ingredients') or []})
    links, food = link_table(names), food_values()
    totals, to_taste, groups = defaultdict(float), set(), {}
    for meal in meals:
        lines, kcal = [], 0.0
        for i in meal.get('ingredients') or []:
            name = i.get('name_lc', '')
            fdc = links.fdc_id.get(name)
            if links.how.get(name) == 'left out' or not name:
                continue
            grams = ingredient_grams(i, int(fdc)) if fdc is not None and fdc == fdc else None
            lines.append((name, grams, i.get('predicted_category') or 'Other'))
            if grams and fdc is not None and fdc == fdc:
                kcal += grams / 100 * (food.kcal_100g.get(int(fdc)) or 0)
        stated = (meal.get('nutrition') or {}).get('calories')
        share = stated / kcal if stated and kcal > 0 else 0
        if not 0 < share <= 1:   # few ingredients linked: a serving can never need more than the whole recipe
            share = 1 / (meal.get('servings') or 1)
        for name, grams, group in lines:
            groups.setdefault(name, group)
            if grams:
                totals[name] += grams * share
            else:
                to_taste.add(name)
    out = defaultdict(list)
    for name, grams in totals.items():
        out[groups[name]].append({'ingredient': name, 'grams': round(grams / 5) * 5 or 5})
    for name in sorted(to_taste - set(totals)):
        out[groups[name]].append({'ingredient': name, 'grams': None})
    return {g: sorted(items, key=lambda x: -(x['grams'] or 0)) for g, items in sorted(out.items())}
