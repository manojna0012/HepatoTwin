"""Turn a food diary typed in words into the NHANES day-1 nutrient totals the twin reads.

NHANES participants come with their 24-hour recall already coded (DR1TOT). A new user writes what they
ate yesterday, for example "Breakfast: 2 eggs and toast with butter. Lunch: a turkey sandwich and
chips. Dinner: spaghetti with meat sauce, a glass of milk". Each item is

1. split from the text (meal labels removed; items separated by commas, full stops, new lines, "and",
   "with" and "plus");
2. read for an amount: a number, a fraction or a word ("a", "two", "half"), and a unit (cup, slice,
   tablespoon, glass = 1 cup, bowl = 1.5 cups, ounces, grams). Without an amount the item is one
   ordinary portion (USDA's medium or single item), and an item added "with" another (butter on toast,
   sugar in tea) is one tablespoon or one teaspoon for sugar;
3. linked to a USDA SR Legacy food: first the everyday foods of data/evidence/diary_food_links.csv
   (toast, cereal, pizza, soda; the longest matching phrase wins), then the recipe-ingredient links of
   ingredient_amounts.py (hand-made, then automatic);
4. converted to grams with USDA's portion weights (ingredient_amounts.ingredient_grams) and summed.

The totals use NHANES names (DR1TKCAL, DR1TPROT, DR1TCARB, DR1TTFAT, DR1TSFAT, DR1TMFAT, DR1TPFAT,
DR1TSUGR, DR1TFIBE, DR1TSODI). Added sugars (DR1T_ADD_SUGARS, teaspoon equivalents of 4.2 g) are
estimated as the sugars of foods that are sugars, syrups, sweets or sweetened drinks, since SR Legacy
does not separate added from natural sugars. The recall is marked reliable (DR1DRSTZ = 1) when at least
MIN_MATCHED of the items were understood, and as a usual day (DR1_300 = 2) unless the user says otherwise.
Items not understood are returned so the user can rephrase them. FPED food groups are not estimated, so
no HEI-2020 score is computed for a typed diary.
"""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.recipe_scoring.ingredient_amounts import (CONTAINERS, CUPS, GRAMS, ITEM_WORDS, SR_DIR, WHOLE_PORTIONS,
                                                   grams_per_cup, ingredient_grams, link_table, number, sr_foods,
                                                   sr_portions)

ROOT = Path(__file__).resolve().parents[2]
DIARY_LINKS = ROOT / 'data' / 'evidence' / 'diary_food_links.csv'
MIN_MATCHED = 0.8
GRAMS_PER_TSP_SUGAR = 4.2
DEFAULT_PORTION_G = 100
NUTRIENTS = {'DR1TKCAL': 1008, 'DR1TPROT': 1003, 'DR1TCARB': 1005, 'DR1TTFAT': 1004, 'DR1TSFAT': 1258,
             'DR1TMFAT': 1292, 'DR1TPFAT': 1293, 'DR1TSUGR': 2000, 'DR1TFIBE': 1079, 'DR1TSODI': 1093}
ADDED_SUGAR_FOODS = r'^(Sugars|Syrups|Honey|Molasses|Jams|Jellies|Marmalade|Candies|Frostings|Beverages, carbonated|' \
                    r'Cookies|Doughnuts|Ice creams|Cake|Pie|Puddings)'
WORD_NUMBERS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'half': 0.5,
                'a half': 0.5, 'couple': 2, 'a couple': 2, 'few': 3, 'a few': 3}
EXTRA_UNITS = {'glass': ('cup', 1.0), 'glasses': ('cup', 1.0), 'mug': ('cup', 1.0), 'mugs': ('cup', 1.0),
               'bowl': ('cup', 1.5), 'bowls': ('cup', 1.5), 'can': ('fluid ounces', 12.0), 'cans': ('fluid ounces', 12.0)}
MEAL_LABELS = r'\b(breakfast|lunch|dinner|supper|snacks?|brunch|dessert|drinks?)\s*:'
CONDIMENT = r'^(Butter|Margarine|Oil|Salad dressing|Mayonnaise|Sauce|Catsup|Mustard|Cream|Cheese|Jams|Honey|Syrups)'


@lru_cache(maxsize=1)
def nutrients_per_100g():
    amounts = pd.read_csv(SR_DIR / 'food_nutrient.csv', usecols=['fdc_id', 'nutrient_id', 'amount'])
    amounts = amounts[amounts.nutrient_id.isin(NUTRIENTS.values())]
    wide = amounts.pivot_table(index='fdc_id', columns='nutrient_id', values='amount', aggfunc='first')
    return wide.rename(columns={v: k for k, v in NUTRIENTS.items()}).fillna(0.0)


@lru_cache(maxsize=1)
def diary_links():
    food = sr_foods()
    out = {}
    for r in pd.read_csv(DIARY_LINKS).itertuples(index=False):
        hits = food[food.description.str.contains(r.usda_pattern, case=False, regex=True)]
        out[r.food] = int(hits.description.str.len().idxmin())
    return out


def split_items(text):
    """[(item text, added 'with' another item)] from free text."""
    text = re.sub(MEAL_LABELS, ',', text.lower())
    items = []
    dishes = [p for p in diary_links() if ' with ' in p]      # "spaghetti with meat sauce" is one dish
    for chunk in re.split(r'[,;.\n]|\band\b|\bplus\b|&', text):
        if any(d in chunk for d in dishes):
            parts = [chunk]
        else:
            parts = re.split(r'\bwith\b|\bin\b(?=\s+(?:sugar|milk|cream))', chunk)
        for n, part in enumerate(parts):
            part = part.strip()
            if part:
                items.append((part, n > 0))
    return items


def read_amount(item):
    """(quantity, unit, food name) from one item's words."""
    words = item.split()
    quantity, unit = None, ''
    two = ' '.join(words[:2])
    if two in WORD_NUMBERS:
        quantity, words = WORD_NUMBERS[two], words[2:]
    elif words and words[0] in WORD_NUMBERS:
        quantity, words = WORD_NUMBERS[words[0]], words[1:]
    elif words and re.fullmatch(r'\d+(?:\.\d+)?|\d+/\d+', words[0]):
        quantity = float(words[0]) if '.' in words[0] else number(words[0])
        words = words[1:]
        if words and re.fullmatch(r'\d+/\d+', words[0]):
            quantity += number(words[0])
            words = words[1:]
    known = set(CUPS) | set(GRAMS) | set(CONTAINERS) | set(ITEM_WORDS) | set(EXTRA_UNITS)
    if words and words[0] in known:
        unit, words = words[0], words[1:]
    elif len(words) > 1 and ' '.join(words[:2]) in known:
        unit, words = ' '.join(words[:2]), words[2:]
    if words and words[0] == 'of':
        words = words[1:]
    return quantity, unit, ' '.join(words).strip()


def link(name):
    """(fdc_id, how) for a food name, or (None, 'not understood')."""
    phrases = sorted((p for p in diary_links() if re.search(rf'\b{re.escape(p)}\b', name)), key=len, reverse=True)
    if phrases:
        return diary_links()[phrases[0]], 'everyday food'
    row = link_table([name]).iloc[0]
    if row.fdc_id is not None and row.fdc_id == row.fdc_id:
        return int(row.fdc_id), row.how
    singular = re.sub(r's\b', '', name)
    if singular != name:
        return link(singular)
    return None, 'not understood'


def ordinary_portion(fdc_id):
    """Grams of one ordinary portion when no amount is given: a whole or single item (a medium apple, a
    burger), else one cup (coffee, spaghetti), else one ounce (chips), else DEFAULT_PORTION_G."""
    for modifier, amount, grams in sr_portions().get(fdc_id, []):
        if modifier == 'inch sub':                 # a sandwich: the 6-inch sub, not one inch of it
            return grams
    for word in WHOLE_PORTIONS:
        for modifier, amount, grams in sr_portions().get(fdc_id, []):
            if re.search(rf'\b{word}', modifier):
                return grams / amount
    cup = grams_per_cup(fdc_id)
    if cup:
        return cup
    for modifier, amount, grams in sr_portions().get(fdc_id, []):
        if modifier.strip() == 'oz':
            return grams / amount
    return DEFAULT_PORTION_G


def item_grams(quantity, unit, fdc_id, added):
    description = sr_foods().description.get(fdc_id, '')
    if unit in EXTRA_UNITS:
        unit, factor = EXTRA_UNITS[unit]
        quantity = (quantity or 1) * factor
    if quantity is None and not unit:
        if added:
            unit, quantity = ('teaspoon', 1) if description.startswith('Sugars') else ('tablespoon', 1)
        elif re.match(CONDIMENT, description):
            unit, quantity = 'tablespoon', 1
        else:
            return ordinary_portion(fdc_id)
    if not unit:                                  # "a sandwich", "2 eggs": that many ordinary portions
        return quantity * ordinary_portion(fdc_id)
    line ={'quantity': '' if quantity is None else str(quantity), 'unit': unit, 'ingredient_phrase': ''}
    if quantity is None:
        line['quantity'] = '1'
    grams = ingredient_grams(line, fdc_id)
    return grams if grams else DEFAULT_PORTION_G * (quantity or 1)


def diary_totals(text, usual_day=True):
    """NHANES-coded day-1 totals from a typed diary, the items as understood, and those that were not."""
    table, food = nutrients_per_100g(), sr_foods()
    items, missed = [], []
    totals = dict.fromkeys(NUTRIENTS, 0.0)
    added_sugar_g = 0.0
    for raw, added in split_items(text):
        quantity, unit, name = read_amount(raw)
        if not name:
            continue
        fdc_id, how = link(name)
        if fdc_id is None or fdc_id not in table.index:
            missed.append(raw)
            continue
        grams = item_grams(quantity, unit, fdc_id, added)
        per = table.loc[fdc_id]
        for key in NUTRIENTS:
            totals[key] += grams / 100 * per[key]
        if re.match(ADDED_SUGAR_FOODS, food.description[fdc_id]):
            added_sugar_g += grams / 100 * per['DR1TSUGR']
        items.append({'text': raw, 'food': food.description[fdc_id], 'grams': round(grams),
                      'kcal': round(grams / 100 * per['DR1TKCAL']), 'linked_by': how})
    understood = len(items) / (len(items) + len(missed)) if items or missed else 0.0
    out = {k: round(float(v), 1) for k, v in totals.items()}
    out['DR1T_ADD_SUGARS'] = round(float(added_sugar_g) / GRAMS_PER_TSP_SUGAR, 2)
    out['DR1DRSTZ'] = 1.0 if items and understood >= MIN_MATCHED else 2.0
    out['DR1_300'] = 2.0 if usual_day else 1.0
    return {'measurements': out, 'items': items, 'not_understood': missed, 'share_understood': round(understood, 2)}
