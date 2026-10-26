"""How typical a recipe's ingredients are of a cuisine, from ingredient co-occurrence (section 7.1).

For each cuisine with at least MIN_RECIPES labelled recipes in the corpus, pair_share[(a, b)] is the
share of that cuisine's recipes that use both ingredient a and ingredient b. A recipe's raw typicality
for the cuisine is the mean pair share over its own ingredient pairs: soy sauce with ginger is common
in Asian recipes, soy sauce with tortillas is not. The raw value is scaled so that the cuisine's own
recipes at the 75th percentile score 100 (capped at 100).

This lets the 71% of candidate meals labelled "Other" in RecipeDB be judged on what they contain.
Spices, herbs and condiments are kept, because they define cuisines (soy sauce, cumin, ginger); only
near-universal items (water, ice, salt, pepper, sugar) are dropped.
"""
import itertools
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

RECIPE_FILE = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'recipe_objects.json'
MIN_RECIPES = 20
SCALE_PERCENTILE = 75
UNLABELLED = {'', 'other', 'none', 'nan'}
UNIVERSAL = {'water', 'ice', 'salt', 'kosher salt', 'sea salt', 'pepper', 'black pepper', 'salt and pepper',
             'sugar', 'white sugar', 'granulated sugar'}


def cuisine_ingredients(recipe):
    names = set()
    for item in recipe.get('ingredients') or []:
        name = item.casefold() if isinstance(item, str) else (item.get('name_lc') or str(item.get('name', '')).casefold())
        if name and name not in UNIVERSAL:
            names.add(name.strip())
    return frozenset(names)


def _pairs(ingredients):
    return itertools.combinations(sorted(ingredients), 2)


def _raw(ingredients, shares):
    pairs = list(_pairs(ingredients))
    return float(np.mean([shares.get(p, 0.0) for p in pairs])) if pairs else None


class CuisineTypicality:
    def __init__(self, recipes):
        by_cuisine = {}
        for recipe in recipes:
            cuisine = str(recipe.get('cuisine') or '').strip()
            if cuisine.casefold() not in UNLABELLED:
                by_cuisine.setdefault(cuisine.casefold(), []).append(cuisine_ingredients(recipe))
        self.shares, self.scale = {}, {}
        for cuisine, sets in by_cuisine.items():
            if len(sets) < MIN_RECIPES:
                continue
            counts = Counter(p for s in sets for p in _pairs(s))
            shares = {p: n / len(sets) for p, n in counts.items()}
            raws = [r for r in (_raw(s, shares) for s in sets) if r is not None]
            self.shares[cuisine] = shares
            self.scale[cuisine] = float(np.percentile(raws, SCALE_PERCENTILE)) if raws else None

    def cuisines(self):
        return sorted(self.shares)

    def score(self, recipe, cuisine):
        """0-100 typicality of the recipe for the cuisine; None when it cannot be judged."""
        key = str(cuisine or '').strip().casefold()
        if key not in self.shares or not self.scale.get(key):
            return None
        raw = _raw(cuisine_ingredients(recipe), self.shares[key])
        return None if raw is None else round(min(100.0, 100 * raw / self.scale[key]), 2)


@lru_cache(maxsize=None)
def default_typicality(path=RECIPE_FILE):
    with open(path, encoding='utf-8') as f:
        return CuisineTypicality(json.load(f))
