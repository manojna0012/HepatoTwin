"""Degree of processing (FPro) and glycaemic load of each recipe (Weekly Plan, meeting 11).

Both are built from the recipe's ingredients (src/recipe_scoring/ingredient_amounts.py links each one to a
USDA SR Legacy food and converts its amount to grams):

- FPro: the energy-weighted mean of the ingredients' FPro (src/recipe_scoring/foodprox.py), the same
  weighting Menichetti et al. (2023) used to score whole diets. 0 means made from unprocessed foods,
  1 from ultra-processed ones.
- Glycaemic load per serving: the sum over ingredients of GI x available carbohydrate (g) / 100, divided by
  the servings (Atkinson et al., AJCN 2021: GL = GI x available carbohydrate / 100; for a mixed dish the
  carbohydrate-weighted GI of its parts). A serving is the part of the dish that carries RecipeDB's stated
  energy per serving, because RecipeDB's servings counts are sometimes wrong (a batch of 48 cookies listed
  as one serving) while its per-serving nutrition is what every other criterion uses. Each carbohydrate food takes the median GI of its matching
  entries in the 2021 international tables (data/evidence/glycaemic_index_links.csv; ISO-method Table 1
  where it has any, otherwise Table 2). Flour and other ingredients eaten cooked take the GI of the form
  in which they are eaten (wheat flour -> wheat bread).

A recipe is scored only when it can be trusted: at least MIN_LINE_SHARE of its ingredient lines are linked
with a known amount, and, for glycaemic load, at least MIN_CARB_SHARE of its available carbohydrate comes
from foods with a GI, and its carbohydrate per serving is within a factor of 3 of RecipeDB's stated value.
Otherwise the value is left blank ("not assessed").

    .venv/Scripts/python.exe -m src.recipe_scoring.processing_and_glycaemic_load
"""
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.recipe_scoring.foodprox import SR_FPRO_FILE
from src.recipe_scoring.ingredient_amounts import ingredient_grams, link_table, sr_foods

ROOT = Path(__file__).resolve().parents[2]
RECIPE_FILE = ROOT / 'data' / 'processed' / 'recipe_objects.json'
GI_LINKS_FILE = ROOT / 'data' / 'evidence' / 'glycaemic_index_links.csv'
GI_TABLE_FILE = ROOT / 'data' / 'raw_gi' / 'atkinson2021_gi.csv'
OUTPUT_DIR = ROOT / 'data' / 'results' / 'ingredient_scores'
SCORES_FILE = OUTPUT_DIR / 'recipe_processing_gl.csv'

MIN_LINE_SHARE = 0.8
MIN_CARB_SHARE = 0.8
MIN_CARB_FOR_GI = 1.0          # g available carbohydrate per 100 g below which a food needs no GI
CARB_AGREEMENT = 3.0


def non_capturing(pattern):
    """The patterns use groups only for alternatives: make them non-capturing (pandas warns otherwise)."""
    return re.sub(r'(?<!\\)\((?!\?)', '(?:', pattern)


def gi_groups(links_file=GI_LINKS_FILE, table_file=GI_TABLE_FILE):
    """Each USDA pattern with its GI: the median of the matching Atkinson 2021 entries."""
    table = pd.read_csv(table_file)
    table['item'] = table.item.fillna('')
    rows = []
    for r in pd.read_csv(links_file, keep_default_na=False).itertuples(index=False):
        sub = table if not r.atkinson_category else table[table.category.fillna('').str.startswith(r.atkinson_category)]
        hits = sub.item.str.contains(non_capturing(r.atkinson_pattern), case=False, regex=True)
        chosen = sub[hits & (sub.table == 1)]
        if chosen.empty:
            chosen = sub[hits & (sub.table == 2)]
        if chosen.empty:
            raise ValueError(f'No Atkinson 2021 entry matches {r.atkinson_pattern!r}')
        rows.append({'usda_pattern': r.usda_pattern, 'gi': float(chosen.gi.median()), 'entries': len(chosen),
                     'atkinson_table': int(chosen.table.iloc[0]),
                     'atkinson_numbers': ' '.join(str(n) for n in chosen.number.tolist()), 'note': r.note})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def food_values():
    """Per SR food: energy, available carbohydrate, FPro and GI (NaN where the food has no GI row)."""
    food = sr_foods().copy()
    food['fpro'] = pd.read_csv(SR_FPRO_FILE, index_col='fdc_id').fpro
    food['gi'] = np.nan
    unassigned = food.index
    for g in gi_groups().itertuples(index=False):
        hits = food.loc[unassigned, 'description'].str.contains(non_capturing(g.usda_pattern), case=False, regex=True)
        food.loc[hits[hits].index, 'gi'] = g.gi
        unassigned = unassigned.difference(hits[hits].index)
    return food


def score_recipe(recipe, links, food):
    """FPro, glycaemic load per serving and the coverage behind them, for one recipe."""
    lines = known = 0
    kcal = carb = carb_with_gi = load = fpro_kcal = 0.0
    for ingredient in recipe.get('ingredients') or []:
        name = ingredient.get('name_lc', '')
        how = links.how.get(name)
        if how == 'left out':
            continue
        lines += 1
        fdc_id = links.fdc_id.get(name)
        if fdc_id is None or pd.isna(fdc_id):
            continue
        fdc_id = int(fdc_id)
        grams = ingredient_grams(ingredient, fdc_id)
        if grams is None:
            continue
        known += 1
        f = food.loc[fdc_id]
        energy = grams / 100 * (f.kcal_100g if pd.notna(f.kcal_100g) else 0.0)
        kcal += energy
        fpro_kcal += energy * f.fpro
        available = grams / 100 * f.available_carbohydrate_100g
        carb += available
        if f.available_carbohydrate_100g < MIN_CARB_FOR_GI:
            carb_with_gi += available
        elif pd.notna(f.gi):
            carb_with_gi += available
            load += f.gi * available / 100
    servings = recipe.get('servings') or np.nan
    stated_kcal = (recipe.get('nutrition') or {}).get('calories')
    line_share = known / lines if lines else 0.0
    carb_share = carb_with_gi / carb if carb > 0 else 1.0
    trusted = line_share >= MIN_LINE_SHARE and kcal > 0
    # A serving is the share of the dish that carries RecipeDB's stated energy per serving, the basis of
    # every other recipe criterion; the servings count is used only when no energy is stated.
    share = stated_kcal / kcal if stated_kcal and kcal > 0 else (1 / servings if servings else np.nan)
    # One badly sized ingredient can dominate the carbohydrate; trust the load only while the summed
    # carbohydrate per serving stays within a factor CARB_AGREEMENT of RecipeDB's own figure.
    stated_carb = (recipe.get('nutrition') or {}).get('carbohydrates')
    agrees = not stated_carb or stated_carb < 5 or 1 / CARB_AGREEMENT <= carb * share / stated_carb <= CARB_AGREEMENT
    return {'recipe_id': recipe['recipe_id'], 'lines': lines, 'line_share': round(line_share, 3),
            'kcal_per_serving_from_ingredients': kcal / servings if servings else np.nan,
            'kcal_per_serving_recipedb': stated_kcal,
            'fpro': round(fpro_kcal / kcal, 3) if trusted else np.nan,
            'available_carbohydrate_g': round(carb * share, 1) if trusted else np.nan,
            'carb_share_with_gi': round(carb_share, 3),
            'glycaemic_load': round(load * share, 1)
            if trusted and carb_share >= MIN_CARB_SHARE and agrees else np.nan}


def main():
    recipes = json.load(open(RECIPE_FILE, encoding='utf-8'))
    uses = Counter(i['name_lc'] for r in recipes for i in r.get('ingredients') or [])
    links = link_table(list(uses))
    links['uses'] = pd.Series(uses)
    food = food_values()
    scores = pd.DataFrame([score_recipe(r, links, food) for r in recipes])

    full = scores[scores.line_share == 1].dropna(subset=['kcal_per_serving_from_ingredients', 'kcal_per_serving_recipedb'])
    full = full[full.kcal_per_serving_recipedb > 0]
    ratio = full.kcal_per_serving_from_ingredients / full.kcal_per_serving_recipedb
    summary = {
        'recipes': len(scores),
        'ingredient_uses_by_link': links.groupby('how').uses.sum().astype(int).to_dict(),
        'fpro_assessed': int(scores.fpro.notna().sum()),
        'glycaemic_load_assessed': int(scores.glycaemic_load.notna().sum()),
        'energy_check': {'recipes_fully_linked': len(full), 'median_ratio': round(float(ratio.median()), 3),
                         'ratio_iqr': [round(float(q), 3) for q in ratio.quantile([0.25, 0.75])],
                         'within_25_percent': round(float(ratio.between(0.75, 1.25).mean()), 3),
                         'spearman': round(float(spearmanr(full.kcal_per_serving_from_ingredients,
                                                           full.kcal_per_serving_recipedb).statistic), 3)},
        'fpro_quartiles': [round(float(q), 3) for q in scores.fpro.quantile([0.25, 0.5, 0.75])],
        'glycaemic_load_quartiles': [round(float(q), 1) for q in scores.glycaemic_load.quantile([0.25, 0.5, 0.75])],
        'rules': {'min_line_share': MIN_LINE_SHARE, 'min_carb_share': MIN_CARB_SHARE,
                  'min_carb_for_gi_g_per_100g': MIN_CARB_FOR_GI, 'carb_agreement_factor': CARB_AGREEMENT},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_FILE, index=False)
    links.reset_index().to_csv(OUTPUT_DIR / 'ingredient_links.csv', index=False)
    gi_groups().to_csv(OUTPUT_DIR / 'glycaemic_index_by_group.csv', index=False)
    json.dump(summary, open(OUTPUT_DIR / 'summary.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
