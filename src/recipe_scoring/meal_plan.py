"""Recommendation diversity and seven-day meal plans (proposal section 17.6).

Diversity: maximal marginal relevance (Carbonell and Goldstein, SIGIR 1998)
re-ranks the Layer 4 list so that near-duplicates (several chickpea curries)
do not fill the top 10. Each pick maximises
    LAMBDA * overall / 100 - (1 - LAMBDA) * max Jaccard similarity
to recipes already picked, on ingredient sets without seasonings and
pantry staples (salt, oil, spices, herbs and other additives). LAMBDA 0.9
was chosen on 60 held-out participants of all ages (17 October): the default top 10 is already
varied (mean diversity 0.94), and lower values traded practicality for
little extra diversity (0.7: diversity 0.97, all-three criteria 0.75 vs
0.83; 0.9: 0.96 and 0.80).

Meal plan: two main meals a day for seven days, drawn greedily from the
diversified list of recipes that meet every reference value, practical
recipes (at most 60 minutes, 15 ingredients and 12 steps) first. Weekly
constraints, all checked by plan_checks:
- every meal meets the patient's reference values;
- each day's two meals together meet them (energy per meal, and the
  energy-weighted saturated fat, sugars and fibre of the day);
- a recipe appears at most twice, never on the same or consecutive days;
- the two meals of a day have different main protein families, and no
  family appears more than MAX_FAMILY_PER_WEEK times;
- every meal is in the requested cuisine or a declared similar one.

A patient may not have the ingredients, tools or taste for another cuisine,
so the week is filled in three steps and stops at the first that completes it:
1. the requested cuisine only, no recipe repeated;
2. a favourite recipe may appear twice (still the requested cuisine);
3. recipes from the most similar cuisines, those whose ingredient profile
   has cosine similarity of at least MIN_SIMILARITY with the requested one
   (cuisine_profiles), each labelled as borrowed.
The main protein family is the first meat, fish, seafood, legume, egg, tofu
or cheese ingredient, grouped by keyword; recipes without one are
"plant-based".
"""

import math
from collections import Counter

from src.recipe_scoring.recipe_criteria import derived_metrics

LAMBDA = 0.9
DAYS, MEALS_PER_DAY = 7, 2
MAX_FAMILY_PER_WEEK = 3
MAX_REPEATS = 2
MIN_SIMILARITY = 0.8
MAX_TIME, MAX_INGREDIENTS, MAX_STEPS = 60, 15, 12
IGNORED_GROUPS = {"Additive", "Additive-Salt", "Additive-Sugar", "Additive-Yeast", "Additive-Vinegar",
                  "Spice", "Herb", "Essential Oil", "Condiment", ""}
PROTEIN_GROUPS = {"Meat", "Fish", "Seafood", "Legume", "Dairy"}
# Broths flavour a dish without being its protein; green beans and snap peas are vegetables.
NOT_PROTEIN = ("broth", "stock", "bouillon", "green bean", "string bean", "snap pea", "snow pea")
# RecipeDB files some of these as vegetables.
ALWAYS_PROTEIN = ("tofu", "tempeh", "egg", "chickpea", "garbanzo", "lentil", "bean")
FAMILIES = [
    ("chicken", ("chicken",)), ("turkey", ("turkey",)), ("beef", ("beef", "steak", "veal")),
    ("pork", ("pork", "ham", "bacon", "sausage", "chorizo", "prosciutto")),
    ("lamb", ("lamb", "mutton", "goat")),
    ("fish", ("fish", "salmon", "tuna", "cod", "tilapia", "halibut", "trout", "anchov", "sardine", "haddock")),
    ("shellfish", ("shrimp", "prawn", "crab", "lobster", "scallop", "clam", "mussel", "oyster", "squid")),
    ("legumes", ("bean", "lentil", "chickpea", "pea", "garbanzo", "dal", "hummus")),
    ("egg", ("egg",)), ("soy", ("tofu", "tempeh", "edamame")),
    ("cheese", ("cheese", "paneer", "ricotta", "feta", "mozzarella", "parmesan")),
]


def ingredient_set(recipe):
    return frozenset(i.get("name_lc") or str(i.get("name", "")).casefold()
                     for i in recipe.get("ingredients") or []
                     if i.get("predicted_category", "") not in IGNORED_GROUPS
                     and (i.get("name_lc") or i.get("name")) not in ("water", "ice"))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def protein_family(recipe):
    for ingredient in recipe.get("ingredients") or []:
        name = ingredient.get("name_lc") or str(ingredient.get("name", "")).casefold()
        name = name.replace("eggplant", "aubergine")
        if any(k in name for k in NOT_PROTEIN):
            continue
        group = ingredient.get("predicted_category", "")
        if group not in PROTEIN_GROUPS and not any(k in name for k in ALWAYS_PROTEIN):
            continue
        for family, keywords in FAMILIES:
            if any(k in name for k in keywords):
                return family
    return "plant-based"


def practical(recipe):
    """Within 60 minutes, 15 ingredients and 12 steps (the evaluation's feasibility limits)."""
    try:
        time = float(recipe.get("total_time"))
    except (TypeError, ValueError):
        return False
    return (time <= MAX_TIME and 0 < len(recipe.get("ingredients") or []) <= MAX_INGREDIENTS
            and 0 < len(recipe.get("instructions") or []) <= MAX_STEPS)


def cuisine_profiles(recipes, min_recipes=20):
    """Share of each cuisine's recipes using each ingredient ("Other" excluded)."""
    counts, sizes = {}, Counter()
    for recipe in recipes:
        cuisine = recipe.get("cuisine")
        if not cuisine or cuisine == "Other":
            continue
        sizes[cuisine] += 1
        counts.setdefault(cuisine, Counter()).update(ingredient_set(recipe))
    return {c: {k: v / sizes[c] for k, v in counter.items()}
            for c, counter in counts.items() if sizes[c] >= min_recipes}


def similar_cuisines(cuisine, profiles, threshold=MIN_SIMILARITY):
    """[(cuisine, cosine similarity)] close to `cuisine`, most similar first."""
    def cosine(a, b):
        dot = sum(v * b.get(k, 0) for k, v in a.items())
        return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    match = next((c for c in profiles if c.casefold() == str(cuisine).casefold()), None)
    if match is None:
        return []
    scored = sorted(((cosine(profiles[match], p), c) for c, p in profiles.items() if c != match), reverse=True)
    return [(c, round(v, 2)) for v, c in scored if v >= threshold]


def intra_list_diversity(recipes):
    """Mean pairwise 1 - Jaccard over ingredient sets (section 12.4)."""
    sets = [ingredient_set(r) for r in recipes]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return sum(1 - jaccard(a, b) for a, b in pairs) / len(pairs) if pairs else None


def diversify(ranked, recipes, k=10, pool=100, lam=LAMBDA):
    """MMR re-ranking of the first `pool` ranked items; returns k items.

    Items keep the health gate: an item that meets every reference value is
    never displaced by one that does not.
    """
    by_id = {str(r["recipe_id"]): r for r in recipes}
    candidates = list(ranked[:pool])
    sets = {str(i["recipe_id"]): ingredient_set(by_id[str(i["recipe_id"])]) for i in candidates}
    chosen = []
    while candidates and len(chosen) < k:
        gate = max(bool(i.get("meets_all_references")) for i in candidates)
        tier = [i for i in candidates if bool(i.get("meets_all_references")) == gate]

        def mmr(item):
            similarity = max((jaccard(sets[str(item["recipe_id"])], sets[str(c["recipe_id"])])
                              for c in chosen), default=0.0)
            return lam * item["overall"] / 100 - (1 - lam) * similarity

        best = max(tier, key=mmr)
        chosen.append(best)
        candidates.remove(best)
    return chosen


def _day_values(meals):
    calories = [float(m["nutrition"]["calories"]) for m in meals]
    total = sum(calories)
    day = {"calories": max(calories)}
    for key in ("saturated_fat", "sugar", "fiber"):
        values = [m["nutrition"].get(key) for m in meals]
        day[key] = None if any(v is None for v in values) else sum(float(v) for v in values)
    combined = derived_metrics({"calories": total, **{k: day[k] for k in ("saturated_fat", "sugar", "fiber")}})
    return {"calories": day["calories"], **combined}


def meets(values, objectives):
    for o in objectives:
        value = values.get(o["nutrient"])
        if value is None:
            return False
        if (o["direction"] == "lower" and value > o["target"]) or \
                (o["direction"] == "higher" and value < o["target"]):
            return False
    return True


def _meal_values(recipe):
    nutrition = dict(recipe.get("nutrition") or {})
    return {"calories": nutrition.get("calories"), **derived_metrics(nutrition)}


def _repeats_ok(plan):
    days = {}
    for d, day in enumerate(plan):
        for m in day:
            days.setdefault(str(m["recipe_id"]), []).append(d)
    return all(len(ds) <= MAX_REPEATS and all(b - a > 1 for a, b in zip(ds, ds[1:]))
               for ds in days.values())


def plan_checks(plan, objectives, cuisines=None):
    """Which weekly constraints a plan satisfies; cuisines include any borrowed ones."""
    meals = [m for day in plan for m in day]
    families = [protein_family(m) for m in meals]
    wanted = {c.casefold() for c in cuisines} if cuisines else None
    return {
        "complete": len(plan) == DAYS and all(len(day) == MEALS_PER_DAY for day in plan),
        "every_meal_meets_references": all(meets(_meal_values(m), objectives) for m in meals),
        # Empty days already fail "complete".
        "every_day_meets_references": all(meets(_day_values(day), objectives) for day in plan if day),
        "repeats_within_limits": _repeats_ok(plan),
        "varied_protein": all(len({protein_family(m) for m in day}) == len(day) for day in plan)
        and all(families.count(f) <= MAX_FAMILY_PER_WEEK for f in set(families)),
        "within_cuisines": wanted is None or all(str(m.get("cuisine")).casefold() in wanted for m in meals),
    }


def _ordered(ranked, by_id, cuisines, pool):
    """Diversified recipes meeting every reference in the given cuisines, practical ones first."""
    wanted = {c.casefold() for c in cuisines} if cuisines else None
    eligible = [i for i in ranked if i.get("meets_all_references") and
                (wanted is None or str(by_id[str(i["recipe_id"])].get("cuisine")).casefold() in wanted)]
    meals = [by_id[str(i["recipe_id"])] for i in diversify(eligible[:pool], list(by_id.values()),
                                                            k=pool, pool=pool)]
    return sorted(meals, key=lambda m: not practical(m))   # stable: keeps the diversified order


def _fill(plan, meals, objectives, repeats):
    """Add meals to the open slots of plan in place, day by day."""
    days_of, family_count = {}, Counter()
    for d, day in enumerate(plan):
        for m in day:
            days_of.setdefault(str(m["recipe_id"]), []).append(d)
            family_count[protein_family(m)] += 1
    for d, day in enumerate(plan):
        for meal in meals:
            if len(day) == MEALS_PER_DAY:
                break
            rid, family = str(meal["recipe_id"]), protein_family(meal)
            used = days_of.get(rid, [])
            if len(used) >= (MAX_REPEATS if repeats else 1) or any(abs(d - u) <= 1 for u in used):
                continue
            if family_count[family] >= MAX_FAMILY_PER_WEEK or any(protein_family(m) == family for m in day):
                continue
            if not meets(_day_values(day + [meal]), objectives):
                continue
            day.append(meal)
            days_of.setdefault(rid, []).append(d)
            family_count[family] += 1


def weekly_plan(ranked, recipes, objectives, cuisines=None, similar=None, pool=300):
    """Seven days of two meals, filled in the three steps of the module docstring.

    similar: [(cuisine, similarity), ...] from similar_cuisines, most similar
    first. Returns (plan, checks, notes): plan is a list of days, each a list
    of recipes; notes give the step that completed the week (None when even
    step 3 could not), the repeated recipe IDs and the borrowed meals.
    """
    by_id = {str(r["recipe_id"]): r for r in recipes}
    plan = [[] for _ in range(DAYS)]
    own = _ordered(ranked, by_id, cuisines, pool)

    def full():
        return all(len(day) == MEALS_PER_DAY for day in plan)

    _fill(plan, own, objectives, repeats=False)
    step = 1 if full() else None
    if step is None:
        _fill(plan, own, objectives, repeats=True)
        step = 2 if full() else None
    borrowed, allowed = {}, list(cuisines or [])
    if step is None and cuisines:
        for cuisine, _ in similar or []:
            before = {str(m["recipe_id"]) for day in plan for m in day}
            _fill(plan, _ordered(ranked, by_id, [cuisine], pool), objectives, repeats=False)
            added = {str(m["recipe_id"]) for day in plan for m in day} - before
            if added:
                allowed.append(cuisine)
                borrowed.update({rid: cuisine for rid in added})
            if full():
                step = 3
                break
    ids = [str(m["recipe_id"]) for day in plan for m in day]
    notes = {"step": step, "repeated": sorted({i for i in ids if ids.count(i) > 1}), "borrowed": borrowed}
    return plan, plan_checks(plan, objectives, allowed or None), notes
