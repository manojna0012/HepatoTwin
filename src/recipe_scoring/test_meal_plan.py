from src.recipe_scoring.meal_plan import (cuisine_profiles, diversify, ingredient_set, intra_list_diversity,
                                          jaccard, plan_checks, practical, protein_family, similar_cuisines,
                                          weekly_plan)
from src.recipe_scoring.multi_criteria import acceptability_score
from src.recipe_scoring.recipe_criteria import CRITERIA, criteria_for

OBJECTIVES, _ = criteria_for([{"id": k, "priority": "medium"} for k in CRITERIA])
PROTEINS = [("chicken breast", "Meat"), ("beef", "Meat"), ("pork", "Meat"), ("salmon", "Fish"),
            ("shrimp", "Seafood"), ("lentils", "Legume"), ("tofu", "Plant"), ("egg", "Dairy"),
            ("feta cheese", "Dairy"), ("turkey", "Meat"), ("lamb", "Meat")]


def recipe(rid, protein, group, vegetables, cuisine="European", calories=450):
    ingredients = [{"name_lc": protein, "predicted_category": group}, {"name_lc": "salt",
                                                                       "predicted_category": "Additive-Salt"}]
    ingredients += [{"name_lc": v, "predicted_category": "Vegetable"} for v in vegetables]
    return {"recipe_id": str(rid), "title": f"R{rid}", "cuisine": cuisine, "ingredients": ingredients,
            "nutrition": {"calories": calories, "saturated_fat": 3, "sugar": 5, "fiber": 10}}


def item(recipe_, overall, meets=True):
    return {"recipe_id": recipe_["recipe_id"], "overall": overall, "meets_all_references": meets,
            "health_margin": 50}


def test_protein_family_and_pantry_items_ignored():
    r = recipe(1, "chicken breasts", "Meat", ["carrot"])
    assert protein_family(r) == "chicken"
    assert protein_family(recipe(2, "eggplant", "Vegetable", [])) == "plant-based"
    assert ingredient_set(r) == {"chicken breasts", "carrot"}       # salt is ignored
    assert jaccard(frozenset("ab"), frozenset("bc")) == 1 / 3


def test_diversify_replaces_near_duplicate_and_keeps_gate():
    a = recipe(1, "chickpeas", "Legume", ["tomato", "onion", "spinach"])
    b = recipe(2, "chickpeas", "Legume", ["tomato", "onion", "spinach"])   # duplicate of a
    c = recipe(3, "salmon", "Fish", ["asparagus", "lemon"])
    d = recipe(4, "beef", "Meat", ["potato"])
    ranked = [item(a, 90), item(b, 89), item(c, 80), item(d, 99, meets=False)]
    picked = [i["recipe_id"] for i in diversify(ranked, [a, b, c, d], k=3, lam=0.6)]
    assert picked == ["1", "3", "2"]                  # duplicate drops below a less similar recipe
    assert intra_list_diversity([a, c]) > intra_list_diversity([a, b])


def test_weekly_plan_meets_every_constraint():
    recipes = [recipe(n, p, g, [f"veg{n}", f"herb{n}"]) for n, (p, g) in enumerate(PROTEINS * 2)]
    ranked = [item(r, 90 - n) for n, r in enumerate(recipes)]
    plan, checks, notes = weekly_plan(ranked, recipes, OBJECTIVES, cuisines=["European"])
    assert all(checks.values()), checks
    assert sum(len(day) for day in plan) == 14
    assert notes == {"step": 1, "repeated": [], "borrowed": {}}


def test_step2_repeats_a_favourite_but_not_on_consecutive_days():
    recipes = [recipe(n, p, g, [f"veg{n}"]) for n, (p, g) in enumerate(PROTEINS[:8])]
    ranked = [item(r, 90 - n) for n, r in enumerate(recipes)]
    plan, checks, notes = weekly_plan(ranked, recipes, OBJECTIVES, cuisines=["European"])
    assert notes["step"] == 2 and notes["repeated"] and not notes["borrowed"]
    assert all(checks.values()), checks
    days = {}
    for d, day in enumerate(plan):
        for m in day:
            days.setdefault(m["recipe_id"], []).append(d)
    assert all(len(ds) <= 2 and (len(ds) < 2 or ds[1] - ds[0] > 1) for ds in days.values())


def test_step3_borrows_only_from_declared_similar_cuisine_and_labels_it():
    own = [recipe(n, p, g, [f"veg{n}"]) for n, (p, g) in enumerate(PROTEINS[:3])]
    other = [recipe(100 + n, p, g, [f"veg{n}"], cuisine="Canadian") for n, (p, g) in enumerate(PROTEINS)]
    far = [recipe(200 + n, p, g, [f"veg{n}"], cuisine="Asian") for n, (p, g) in enumerate(PROTEINS)]
    recipes = own + other + far
    ranked = [item(r, 90 - n) for n, r in enumerate(recipes)]
    plan, checks, notes = weekly_plan(ranked, recipes, OBJECTIVES, cuisines=["European"],
                                      similar=[("Canadian", 0.9)])
    assert notes["step"] == 3 and all(checks.values()), checks
    assert set(notes["borrowed"].values()) == {"Canadian"}
    assert all(m["cuisine"] != "Asian" for day in plan for m in day)
    plan, checks, notes = weekly_plan(ranked, recipes, OBJECTIVES, cuisines=["European"])
    assert notes["step"] is None and not checks["complete"]


def test_similar_cuisines_by_shared_ingredients():
    def dish(rid, cuisine, names):
        return {"recipe_id": rid, "cuisine": cuisine,
                "ingredients": [{"name_lc": n, "predicted_category": "Vegetable"} for n in names]}
    recipes = ([dish(i, "A", ["corn", "bean", "chili"]) for i in range(20)]
               + [dish(i, "B", ["corn", "bean", "lime"]) for i in range(20)]
               + [dish(i, "C", ["rice", "soy", "ginger"]) for i in range(20)])
    profiles = cuisine_profiles(recipes)
    assert [c for c, _ in similar_cuisines("A", profiles, threshold=0.5)] == ["B"]
    assert similar_cuisines("Unknown", profiles) == []


def test_practical_limits():
    base = {"total_time": 45, "ingredients": [{}] * 10, "instructions": ["step"] * 8}
    assert practical(base)
    assert not practical({**base, "total_time": 90})
    assert not practical({**base, "instructions": ["step"] * 13})
    assert not practical({**base, "total_time": None})


def test_plan_checks_catch_violations():
    a = recipe(1, "beef", "Meat", ["x"])
    b = recipe(2, "beef steak", "Meat", ["y"], calories=700)   # same family, over 600 kcal
    checks = plan_checks([[a, b]], OBJECTIVES, cuisines=["Asian"])
    assert not checks["complete"] and not checks["varied_protein"]
    assert not checks["every_meal_meets_references"] and not checks["within_cuisines"]


def test_acceptability_shrinks_few_ratings():
    assert acceptability_score({"ratings": 5, "ratings_count": 3000}) > \
        acceptability_score({"ratings": 5, "ratings_count": 3}) > \
        acceptability_score({"ratings": 4.6, "ratings_count": 3000})
    assert acceptability_score({"ratings": None}) is None
