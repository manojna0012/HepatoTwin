import numpy as np

from src.recipe_scoring.personal_fit import dish_distance, fit_scores, personalise


def recipe(rid, kcal, ingredients):
    return {'recipe_id': rid, 'title': rid, 'ingredients': ingredients,
            'nutrition': {'calories': kcal, 'saturated_fat': 2, 'sugar': 5, 'fiber': 6, 'fat': 10, 'carbohydrates': 40, 'protein': 20}}


def item(rid, overall, gated=True):
    return {'recipe_id': rid, 'overall': overall, 'meets_all_references': gated, 'health_margin': 80 if gated else None,
            'scores': {'health': 100 if gated else 60, 'feasibility': 80}}


def test_favourite_dishes_move_similar_recipes_up_but_never_above_the_health_gate():
    recipes = [recipe('beans', 500, ['black beans', 'rice', 'onion']), recipe('soup', 500, ['carrot', 'celery', 'leek']),
               recipe('fries', 500, ['potato', 'oil'])]
    ranked = [item('soup', 90), item('beans', 89), item('fries', 99, gated=False)]
    favourite = [{'ingredients': ['black beans', 'rice', 'pepper']}]
    out = personalise(ranked, recipes, [], favourites=favourite)
    assert [i['recipe_id'] for i in out] == ['beans', 'soup', 'fries']      # the ungated recipe stays last
    assert out[0]['scores']['personal_fit'] > out[1]['scores']['personal_fit'] and out[0]['rank'] == 1


def test_without_food_answers_the_ranking_is_unchanged():
    ranked = [item('soup', 90), item('beans', 89)]
    recipes = [recipe('soup', 500, ['carrot']), recipe('beans', 500, ['beans'])]
    assert personalise(ranked, recipes, [], patient=None, favourites=()) == ranked
    assert fit_scores(recipes, [], None, ()) is None
    assert np.allclose(dish_distance(recipes, [{'ingredients': ['carrot']}]), [0, 1])
