import pytest

from src.recipe_scoring.food_preferences import (FoodPreferences, allowed_recipes, breaks, contains, liked_dish,
                                                 questions)


def recipe(*phrases, diet='non-vegetarian', minutes=30, groups=None):
    groups = groups or {}
    return {'recipe_id': phrases[0], 'diet': diet, 'total_time': minutes,
            'ingredients': [{'name_lc': p, 'ingredient_phrase': p, 'predicted_category': groups.get(p, '')}
                            for p in phrases]}


def test_look_alikes_are_not_matched():
    assert not contains(recipe('1 eggplant, diced'), 'egg')
    assert not contains(recipe('1 cup coconut milk', '2 tbsp peanut butter'), 'milk')
    assert not contains(recipe('1/2 tsp nutmeg', '1 butternut squash'), 'tree_nuts')
    assert not contains(recipe('2 tbsp red wine vinegar', '1 tbsp ginger'), 'alcohol')
    assert not contains(recipe('2 cups vegetable broth'), 'meat')
    assert not contains(recipe('1 cup rice flour', '8 corn tortillas'), 'wheat')
    assert contains(recipe('2 tbsp butter'), 'milk') and contains(recipe('1 tbsp worcestershire sauce'), 'fish')
    assert contains(recipe('2 tbsp soy sauce'), 'wheat') and contains(recipe('2 tbsp soy sauce'), 'soy')
    assert contains(recipe('1/2 cup chopped walnuts'), 'tree_nuts') and contains(recipe('3 eggs'), 'egg')


def test_diets_use_both_the_label_and_the_ingredients():
    beans = recipe('1 can black beans', '1 onion', diet='vegan')
    mislabelled = recipe('1 lb ground beef', diet='vegan', groups={'1 lb ground beef': 'Meat'})
    cheese = recipe('1 cup cheddar cheese', diet='vegetarian', groups={'1 cup cheddar cheese': 'Dairy'})
    salmon = recipe('2 salmon fillets', diet='pescetarian', groups={'2 salmon fillets': 'Fish'})
    omelette = recipe('3 eggs', '1 onion', diet='vegetarian')
    vegan = FoodPreferences(diet='vegan')
    assert not breaks(beans, vegan) and breaks(mislabelled, vegan) and breaks(cheese, vegan)
    assert not breaks(cheese, FoodPreferences(diet='vegetarian')) and breaks(salmon, FoodPreferences(diet='vegetarian'))
    assert not breaks(salmon, FoodPreferences(diet='pescatarian'))
    assert breaks(omelette, FoodPreferences(diet='eggless_vegetarian'))
    assert not breaks(omelette, FoodPreferences(diet='vegetarian'))


def test_allergies_avoid_dislikes_and_time_are_hard_rules():
    satay = recipe('2 tbsp peanut butter', '1 lb chicken breast', minutes=50)
    prefs = FoodPreferences.from_dict({'allergies': ['peanuts']})
    assert breaks(satay, prefs) == ['allergen: peanuts']
    assert breaks(recipe('2 slices bacon'), FoodPreferences(avoid=('pork',)))
    assert breaks(recipe('1 cup sliced mushrooms'), FoodPreferences(dislikes=('mushroom',)))
    assert breaks(satay, FoodPreferences(max_minutes=30)) == ['takes 50 minutes']
    assert breaks(recipe('1 cup pearl barley'), FoodPreferences(allergies=('gluten',)))
    assert not breaks(recipe('1 cup pearl barley'), FoodPreferences(allergies=('wheat',)))
    kept = allowed_recipes([satay, recipe('1 can chickpeas')], {'allergies': ['peanuts']})
    assert [r['recipe_id'] for r in kept] == ['1 can chickpeas']


def test_answers_are_checked_and_likes_only_rank():
    with pytest.raises(ValueError):
        FoodPreferences.from_dict({'diet': 'keto'})
    with pytest.raises(ValueError):
        FoodPreferences.from_dict({'allergies': ['strawberries']})
    prefs = FoodPreferences.from_dict({'likes': [' Beans ', 'beans', 'Chicken'], 'household': '3'})
    assert prefs.likes == ('beans', 'chicken') and prefs.household == 3 and prefs.is_default
    assert liked_dish(prefs)['ingredients'] == ['beans', 'chicken'] and liked_dish(FoodPreferences()) is None
    assert set(questions()) == {'diet', 'allergies', 'other_allergies', 'avoid', 'dislikes', 'likes', 'max_minutes', 'household'}


def test_an_allergy_typed_under_other_removes_recipes_that_list_it():
    from src.recipe_scoring.food_preferences import FoodPreferences, allowed_recipes, breaks
    kiwi = {'diet': 'vegan', 'ingredients': [{'name': 'kiwi', 'ingredient_phrase': '2 kiwis, sliced'}]}
    plain = {'diet': 'vegan', 'ingredients': [{'name': 'apple', 'ingredient_phrase': '1 apple'}]}
    prefs = FoodPreferences.from_dict({'other_allergies': ['Kiwi ']})
    assert prefs.other_allergies == ('kiwi',) and not prefs.is_default
    assert breaks(kiwi, prefs) == ['allergen: kiwi'] and allowed_recipes([kiwi, plain], prefs) == [plain]
    assert FoodPreferences.from_dict(prefs.as_dict()) == prefs
