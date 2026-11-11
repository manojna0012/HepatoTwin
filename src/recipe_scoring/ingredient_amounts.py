"""Recipe ingredients linked to USDA SR Legacy foods and converted to grams (Weekly Plan, meeting 11).

RecipeDB gives each recipe's nutrition only per serving, but ultra-processing (FPro) and glycaemic
load are properties of the ingredients. Each ingredient name is therefore linked to one SR Legacy food:

- the 400 most frequent names (87% of ingredient uses) by hand, in
  data/evidence/ingredient_usda_manual.csv (a pattern for the USDA description, or blank for names that
  are not foods or too vague, such as "water" or "sauce");
- every other name automatically, to the generic SR food whose description shares its words, with the
  ingredient's last word (its noun) required and raw foods preferred. Links below a minimum score are left
  out.

Amounts become grams with USDA's portion weights for the linked food: cups, tablespoons and teaspoons from
its household measures (1 cup = 16 tablespoons = 48 teaspoons), counted items (a clove, a slice, a medium
onion) from the matching portion or else the first measure that names the item ("pepper", "tortilla"),
weights directly, whole spices without a weight (a bay leaf) as zero, and cans or packages from the size written in the
recipe ("1 (14 1/2 ounce) can"). A pinch is 1/16 teaspoon and a dash 1/8. Where the food has no volume
measure, 1 ml weighs 1 g. Amounts left blank ("salt to taste") count as zero.

The links and grams are checked against RecipeDB itself: energy summed over the ingredients, divided by the
servings, against the recipe's stated energy per serving.
"""
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SR_DIR = ROOT / 'data' / 'raw_usda' / 'FoodData_Central_sr_legacy_food_csv_2018-04'
MANUAL_FILE = ROOT / 'data' / 'evidence' / 'ingredient_usda_manual.csv'

DISTRIBUTION_NOTE = " (Includes foods for USDA's Food Distribution Program)"
GRAMS_PER_OUNCE, GRAMS_PER_POUND, ML_PER_CUP = 28.3495, 453.592, 236.588
CUPS = {'cup': 1, 'cups': 1, 'c': 1, 'tablespoon': 1 / 16, 'tablespoons': 1 / 16, 'tbsp': 1 / 16,
        'teaspoon': 1 / 48, 'teaspoons': 1 / 48, 'tsp': 1 / 48, 'pinch': 1 / 768, 'pinches': 1 / 768,
        'dash': 1 / 384, 'dashes': 1 / 384, 'drop': 1 / 3648, 'drops': 1 / 3648,
        'fluid ounce': 1 / 8, 'fluid ounces': 1 / 8, 'fl oz': 1 / 8, 'pint': 2, 'pints': 2, 'quart': 4,
        'quarts': 4, 'gallon': 16, 'gallons': 16, 'ml': 1 / ML_PER_CUP, 'l': 1000 / ML_PER_CUP,
        'liter': 1000 / ML_PER_CUP, 'litre': 1000 / ML_PER_CUP}
GRAMS = {'g': 1, 'gram': 1, 'grams': 1, 'kg': 1000, 'ounce': GRAMS_PER_OUNCE, 'ounces': GRAMS_PER_OUNCE,
         'oz': GRAMS_PER_OUNCE, 'lb': GRAMS_PER_POUND, 'lbs': GRAMS_PER_POUND, 'pound': GRAMS_PER_POUND,
         'pounds': GRAMS_PER_POUND}
CONTAINERS = {'can', 'cans', 'package', 'packages', 'box', 'boxes', 'jar', 'jars', 'bag', 'bags', 'envelope',
              'envelopes', 'container', 'containers', 'bottle', 'bottles', 'carton', 'cartons', 'packet',
              'packets'}
# Counted items, matched against the words of USDA's portion descriptions.
ITEM_WORDS = {'clove': 'clove', 'cloves': 'clove', 'slice': 'slice', 'slices': 'slice', 'stalk': 'stalk',
              'stalks': 'stalk', 'head': 'head', 'heads': 'head', 'bunch': 'bunch', 'bunches': 'bunch',
              'sprig': 'sprig', 'sprigs': 'sprig', 'ear': 'ear', 'ears': 'ear', 'stick': 'stick',
              'sticks': 'stick', 'piece': 'piece', 'pieces': 'piece', 'leaf': 'leaf', 'leaves': 'leaf'}
WHOLE_PORTIONS = ('medium', 'whole', 'large', 'each', 'item', 'fruit', 'small', 'serving')
MEASURE_WORDS = {'cup', 'cups', 'tbsp', 'tsp', 'tablespoon', 'teaspoon', 'oz', 'fl', 'lb', 'g', 'package',
                 'serving', 'pat', 'dash', 'pinch'}
ENERGY, CARBOHYDRATE, FIBRE = 1008, 1005, 1079

AUTO_STOP = {'fresh', 'chopped', 'large', 'small', 'medium', 'of', 'and', 'or', 'to', 'taste', 'for', 'ground',
             'dried', 'minced', 'sliced', 'diced', 'whole', 'finely', 'cut', 'into', 'pieces', 'a', 'the', 'frozen'}
AUTO_MIN_SCORE = 2.5
FOOD_GROUPS = {'spice', 'cheese', 'nut', 'bean', 'fish', 'oil', 'sugar', 'seed', 'pepper', 'beef', 'pork', 'chicken',
               'turkey', 'lamb', 'veal', 'crustacean', 'mollusk', 'lettuce', 'squash', 'cereal', 'syrup', 'alcoholic',
               'beverage', 'wheat', 'rice', 'vegetable', 'fruit', 'pasta', 'bread', 'cracker', 'cookie', 'candy',
               'candie', 'snack', 'sweetener', 'flour', 'salad', 'dressing', 'soup', 'milk', 'cream', 'egg', 'game',
               'meat', 'vinegar', 'juice', 'tomato', 'product', 'mushroom', 'onion', 'cabbage', 'potato', 'pea', 'and',
               'or', 'with', 'mature', 'dried', 'raw', 'fresh', 'canned', 'frozen'}
BRANDS = r"[A-Z]{3,}|Babyfood|Infant formula|Fast foods?|Restaurant|School Lunch|Pillsbury|McDonald|Burger King"


def number(text):
    """'1 1/2' -> 1.5, '1/3' -> 0.333; None when empty or not a number."""
    text = (text or '').strip()
    if not text:
        return None
    try:
        return float(sum(Fraction(part) for part in text.split()))
    except (ValueError, ZeroDivisionError):
        return None


def words(text):
    out = []
    for w in re.findall(r"[a-z]+", text.lower().replace('-', ' ')):
        if w.endswith('ies'):
            w = w[:-3] + 'y'
        elif w.endswith('oes'):
            w = w[:-2]
        elif w.endswith('s') and not w.endswith('ss') and len(w) > 3:
            w = w[:-1]
        out.append(w)
    return out


@lru_cache(maxsize=1)
def sr_foods(sr_dir=SR_DIR):
    """SR Legacy foods with energy, available carbohydrate and fibre per 100 g."""
    food = pd.read_csv(sr_dir / 'food.csv', usecols=['fdc_id', 'description'])
    food['description'] = food.description.str.replace(DISTRIBUTION_NOTE, '', regex=False)
    amounts = pd.read_csv(sr_dir / 'food_nutrient.csv', usecols=['fdc_id', 'nutrient_id', 'amount'])
    amounts = amounts[amounts.nutrient_id.isin([ENERGY, CARBOHYDRATE, FIBRE])]
    wide = amounts.pivot_table(index='fdc_id', columns='nutrient_id', values='amount', aggfunc='first')
    food = food.set_index('fdc_id').join(wide)
    food = food.rename(columns={ENERGY: 'kcal_100g', CARBOHYDRATE: 'carbohydrate_100g', FIBRE: 'fibre_100g'})
    food[['carbohydrate_100g', 'fibre_100g']] = food[['carbohydrate_100g', 'fibre_100g']].fillna(0)
    food['available_carbohydrate_100g'] = (food.carbohydrate_100g - food.fibre_100g).clip(lower=0)
    return food


@lru_cache(maxsize=1)
def sr_portions(sr_dir=SR_DIR):
    """USDA household measures: fdc_id -> list of (description words, amount, grams)."""
    p = pd.read_csv(sr_dir / 'food_portion.csv', usecols=['fdc_id', 'amount', 'modifier', 'gram_weight'])
    p['modifier'] = p.modifier.fillna('').str.lower()
    out = {}
    for fdc_id, amount, modifier, grams in p.itertuples(index=False):
        out.setdefault(fdc_id, []).append((modifier, amount or 1.0, grams))
    return out


def grams_per_cup(fdc_id):
    """Weight of one cup of the food from its household measures, or None."""
    for modifier, amount, grams in sr_portions().get(fdc_id, []):
        first = re.split(r"[ ,(]", modifier.strip())[0] if modifier.strip() else ''
        for unit, cups in (('cup', 1), ('tbsp', 1 / 16), ('tablespoon', 1 / 16), ('tsp', 1 / 48),
                           ('teaspoon', 1 / 48), ('fl', 1 / 8)):
            if first == unit:
                return grams / (amount * cups)
    return None


def grams_per_item(fdc_id, item=None):
    """Weight of one counted item ('clove', 'slice' or a whole medium one) from the household measures."""
    portions = sr_portions().get(fdc_id, [])
    wanted = [item] if item else []
    for word in wanted + list(WHOLE_PORTIONS):
        for modifier, amount, grams in portions:
            if re.search(rf"\b{word}", modifier):
                return grams / amount
    # Otherwise the first measure that names the item itself ("pepper", "thigh with skin", "tortilla").
    for modifier, amount, grams in portions:
        first = re.split(r"[ ,(]", modifier.strip())[0] if modifier.strip() else ''
        if first and first not in MEASURE_WORDS:
            return grams / amount
    return None


def container_ounces(phrase):
    """Size written in the recipe: '1 (14 1/2 ounce) can' -> 14.5 ounces."""
    m = re.search(r"\(\s*([\d/ .]+?)\s*(?:-\s*[\d/ .]+)?\s*(ounce|oz|g|gram)s?\b", phrase or '', re.I)
    if not m:
        return None
    size = number(m.group(1).replace('.', ' ').strip()) if '.' not in m.group(1) else float(m.group(1))
    if size is None:
        return None
    return size if m.group(2).lower() in ('ounce', 'oz') else size / GRAMS_PER_OUNCE


def ingredient_grams(ingredient, fdc_id):
    """Grams of one recipe ingredient line, or None when the amount cannot be worked out.

    A '|' separates alternative amounts ("2|540", "cups|ml": 2 cups or a 540 ml can), so only the
    first is used, except for a container whose size follows its count ("1|15", "ounce|can")."""
    units = [u.strip().lower() for u in (ingredient.get('unit') or '').split('|')]
    amounts = [number(q) for q in (ingredient.get('quantity') or '').split('|')]
    if all(a is None for a in amounts):
        return 0.0                                          # "salt to taste": no amount given
    first = amounts[0] if amounts[0] is not None else 1.0
    unit = units[0].replace(' whole', '')
    if unit in ('ounce', 'ounces', 'oz', 'g', 'lb') and len(units) > 1 and units[1].split()[0] in CONTAINERS:
        size = amounts[1] if len(amounts) > 1 and amounts[1] is not None else None
        if size is None:
            size = container_ounces(ingredient.get('ingredient_phrase'))
            return None if size is None else first * size * GRAMS_PER_OUNCE
        return first * size * GRAMS[unit]
    if unit in GRAMS:
        return first * GRAMS[unit]
    if unit in CUPS:
        return first * CUPS[unit] * (grams_per_cup(fdc_id) or ML_PER_CUP)
    if unit in CONTAINERS:
        size = container_ounces(ingredient.get('ingredient_phrase'))
        return None if size is None else first * size * GRAMS_PER_OUNCE
    item = grams_per_item(fdc_id, ITEM_WORDS.get(unit))
    if item is None and sr_foods().description.get(fdc_id, '').startswith('Spices,'):
        return 0.0                          # a bay leaf or a few whole cloves: no weight given, negligible energy
    return None if item is None else first * item


@lru_cache(maxsize=1)
def _candidates():
    food = sr_foods()
    c = food[~food.description.str.contains(BRANDS, regex=True)][['description']].copy()
    c['w'] = c.description.map(words)
    return list(zip(c.index, c.description, c.w))


@lru_cache(maxsize=None)
def auto_link(name):
    """(fdc_id, score) of the generic SR food that best shares the ingredient's words, or (None, score).

    The ingredient's last word (its noun) must appear in the description, and every word of the
    description's lead (before the first comma, such as "Sauce" in "Sauce, steak") must come from the
    ingredient or be a food-group word ("Beef", "Spices"), so "steaks" cannot become a steak sauce."""
    q = [w for w in words(name) if w not in AUTO_STOP]
    if not q:
        return None, 0.0
    best, best_score, qs = None, -1e9, set(q)
    for fdc_id, description, w in _candidates():
        ws = set(w)
        hit = len(qs & ws)
        if hit == 0 or q[-1] not in ws:
            continue
        lead = set(words(description.split(',')[0]))
        if not lead <= qs | FOOD_GROUPS:
            continue
        score = 3 * hit / len(qs) - 0.15 * len(w) + (2 if w.index(q[-1]) <= 1 else 0) + (0.6 if 'raw' in ws else 0)
        if score > best_score:
            best, best_score = fdc_id, score
    return (best, best_score) if best_score >= AUTO_MIN_SCORE else (None, best_score)


@lru_cache(maxsize=4)          # 650 regex searches over SR Legacy; the file does not change while running
def manual_links(path=MANUAL_FILE):
    """Hand-made links: ingredient name -> fdc_id (None for names deliberately left unlinked)."""
    food = sr_foods()
    out = {}
    for row in pd.read_csv(path, keep_default_na=False).itertuples(index=False):
        if not row.usda_pattern:
            out[row.ingredient] = None
            continue
        hits = food[food.description.str.contains(row.usda_pattern, case=False, regex=True)]
        if hits.empty:
            raise ValueError(f'No SR Legacy food matches {row.usda_pattern!r} for {row.ingredient!r}')
        out[row.ingredient] = int(hits.description.str.len().idxmin())
    return out


def link_table(names, path=MANUAL_FILE):
    """One row per ingredient name: fdc_id, description and how it was linked."""
    manual, food, rows = manual_links(path), sr_foods(), []
    for name in names:
        if name in manual:
            fdc_id, how, score = manual[name], 'manual' if manual[name] else 'left out', np.nan
        else:
            fdc_id, score = auto_link(name)
            how = 'automatic' if fdc_id else 'no match'
        rows.append({'ingredient': name, 'fdc_id': fdc_id, 'how': how, 'score': score,
                     'usda_description': food.description.get(fdc_id) if fdc_id else None})
    columns = ['ingredient', 'fdc_id', 'how', 'score', 'usda_description']   # no names: an empty table, not an error
    return pd.DataFrame(rows, columns=columns).set_index('ingredient')
