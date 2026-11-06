import pytest

from src.recipe_scoring.food_diary import diary_totals, read_amount, split_items
from src.recipe_scoring.habitual_diet import habitual_intake

DIARY = ('Breakfast: 2 eggs and toast with butter, coffee with milk. Lunch: a turkey sandwich and chips, '
         'a can of coke. Dinner: spaghetti with meat sauce, a glass of milk, ice cream')


def test_items_and_amounts_are_read():
    items = [t for t, _ in split_items(DIARY)]
    assert items[:4] == ['2 eggs', 'toast', 'butter', 'coffee'] and 'spaghetti with meat sauce' in items
    assert ('butter', True) in split_items(DIARY)                      # added "with" toast
    assert read_amount('2 slices of pizza') == (2, 'slices', 'pizza')
    assert read_amount('a glass of milk') == (1, 'glass', 'milk')
    assert read_amount('1 1/2 cups rice') == (1.5, 'cups', 'rice')


def test_a_typed_diary_becomes_nhanes_recall_totals():
    out = diary_totals(DIARY)
    foods = {i['text']: i for i in out['items']}
    assert out['not_understood'] == [] and out['share_understood'] == 1.0
    assert foods['2 eggs']['grams'] == 92 and foods['a glass of milk']['grams'] == 244
    assert foods['a turkey sandwich']['food'].startswith('Fast foods, submarine sandwich, turkey')
    assert foods['butter']['grams'] == 14                               # one tablespoon on the toast
    m = out['measurements']
    assert 1100 < m['DR1TKCAL'] < 1600 and m['DR1TSFAT'] > 0 and m['DR1T_ADD_SUGARS'] > 5   # the coke
    assert m['DR1DRSTZ'] == 1 and m['DR1_300'] == 2
    intake = habitual_intake({**m, 'RIAGENDR': 2})                      # the twin can now use it
    assert intake['recall_reliable'] and intake['saturated_fat_pct_energy'] == pytest.approx(900 * m['DR1TSFAT'] / m['DR1TKCAL'])


def test_unknown_items_are_returned_and_lower_reliability():
    out = diary_totals('a zorblax and a frumple, 2 eggs')
    assert out['not_understood'] == ['a zorblax', 'a frumple'] and out['measurements']['DR1DRSTZ'] == 2
    assert diary_totals('2 eggs', usual_day=False)['measurements']['DR1_300'] == 1
