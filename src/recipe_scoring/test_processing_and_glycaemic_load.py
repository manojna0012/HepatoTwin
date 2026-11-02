import numpy as np
import pandas as pd
import pytest

from src.recipe_scoring.foodprox import ABSENT, fpro, log_profile
from src.recipe_scoring.ingredient_amounts import (GRAMS_PER_OUNCE, container_ounces, grams_per_cup,
                                                   ingredient_grams, number, sr_foods)
from src.recipe_scoring.processing_and_glycaemic_load import gi_groups, score_recipe

ONION = int(sr_foods().index[sr_foods().description == 'Onions, raw'][0])
SUGAR = int(sr_foods().index[sr_foods().description == 'Sugars, granulated'][0])
BEEF = int(sr_foods().index[sr_foods().description == 'Beef, ground, 80% lean meat / 20% fat, raw'][0])


def line(quantity, unit, name='x', phrase=''):
    return {'quantity': quantity, 'unit': unit, 'name_lc': name, 'ingredient_phrase': phrase}


def test_fpro_runs_from_minimally_to_ultra_processed():
    assert fpro([[1, 0, 0, 0], [0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25]]) == pytest.approx([0, 1, 0.5])
    assert list(log_profile([1.0, 0.0, np.nan])) == [0.0, ABSENT, ABSENT]


def test_amounts_and_alternatives():
    assert number('1 1/2') == 1.5 and number('1/3') == pytest.approx(1 / 3) and number('') is None
    assert container_ounces('1 (14 1/2 ounce) can diced tomatoes') == 14.5
    cup = grams_per_cup(ONION)
    assert cup == pytest.approx(160)                                     # USDA: 1 cup chopped = 160 g
    assert ingredient_grams(line('2', 'tablespoons'), ONION) == pytest.approx(cup / 8)
    assert ingredient_grams(line('2|540', 'cups|ml'), ONION) == pytest.approx(2 * cup)   # alternatives, not a sum
    assert ingredient_grams(line('1', 'ounce|can', phrase='1 (15 ounce) can'), ONION) == pytest.approx(15 * GRAMS_PER_OUNCE)
    assert ingredient_grams(line('2', 'lbs'), BEEF) == pytest.approx(907.184)
    assert ingredient_grams(line('1', ''), ONION) == pytest.approx(110)  # a medium onion
    assert ingredient_grams(line('', ''), ONION) == 0.0                  # "to taste"


def test_gi_groups_use_the_atkinson_tables():
    groups = gi_groups().set_index('usda_pattern')
    assert groups.loc['^Sugars, (granulated|brown|powdered)', 'gi'] == pytest.approx(60.5)
    assert (groups.gi.between(10, 110)).all() and (groups.entries > 0).all()


def test_recipe_scores_weight_by_energy_and_scale_to_the_stated_serving():
    food = pd.DataFrame({'kcal_100g': [400.0, 40.0], 'available_carbohydrate_100g': [100.0, 8.0],
                         'fpro': [0.6, 0.1], 'gi': [60.0, np.nan]}, index=[1, 2])
    links = pd.DataFrame({'fdc_id': [1, 2, None], 'how': ['manual', 'manual', 'left out']},
                         index=['sugar', 'onion', 'water'])
    recipe = {'recipe_id': 'r', 'servings': 2, 'nutrition': {'calories': 220, 'carbohydrates': 54},
              'ingredients': [line('100', 'g', 'sugar'), line('100', 'g', 'onion'), line('1', 'cup', 'water')]}
    s = score_recipe(recipe, links, food)
    assert s['fpro'] == pytest.approx((400 * 0.6 + 40 * 0.1) / 440, abs=1e-3)
    # onion has 8 g carbohydrate per 100 g but no GI: 100 / 108 of the carbohydrate has a GI
    assert s['carb_share_with_gi'] == pytest.approx(100 / 108, abs=1e-3)
    assert s['glycaemic_load'] == pytest.approx(60 * 100 / 100 * 220 / 440, abs=0.1)    # half the dish
    s = score_recipe({**recipe, 'nutrition': {'calories': 220, 'carbohydrates': 5.5}}, links, food)
    assert np.isnan(s['glycaemic_load'])                                 # 54 g vs RecipeDB's 5.5 g: not trusted


def test_ingredient_measures_are_optional_criteria():
    from src.recipe_scoring.meal_plan import meets
    from src.recipe_scoring.recipe_criteria import CRITERIA, margin_score
    upf, gl = CRITERIA['limit_ultra_processed'], CRITERIA['lower_glycaemic_load']
    objectives = [{**upf, 'weight': 1}, {**gl, 'weight': 1}, {**CRITERIA['limit_saturated_fat'], 'weight': 1}]
    values = {'fpro': 0.25, 'glycaemic_load': None, 'saturated_fat_pct_energy': 5}
    assert meets(values, objectives)                       # glycaemic load not assessed: not a miss
    assert margin_score(values, objectives) == pytest.approx(50)      # fpro 0.25 and sat fat 5: half-way each
    assert meets({**values, 'glycaemic_load': 25}, objectives)      # outside the gate: shapes the score only
    assert margin_score({**values, 'glycaemic_load': 10}, objectives) == pytest.approx(50)
