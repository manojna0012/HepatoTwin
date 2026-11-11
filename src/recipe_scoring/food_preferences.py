"""Food preferences asked at sign-up and the recipes they allow (Weekly Plan, meeting 11).

The sign-up food screen asks:

- diet: anything, pescatarian (fish but no meat), vegetarian (eggs and dairy allowed), eggless vegetarian
  (dairy but no eggs) or vegan;
- allergies: the nine major food allergens named in US law (FALCPA 2004 plus sesame, FASTER Act 2021):
  milk, egg, fish, shellfish, tree nuts, peanuts, wheat, soy and sesame; gluten is offered too and also
  rules out barley and rye; "other" lets the patient type any further allergy (free text, e.g. "kiwi",
  "mustard"), matched on the ingredient words like a dislike;
- avoid: pork, beef or alcohol (religious or personal reasons);
- dislikes: ingredients the patient will not eat (free text, e.g. "mushroom", "cilantro");
- likes: ingredients the patient enjoys (free text) - these only move recipes up, never out;
- max_minutes: the most time the patient wants to spend on a meal;
- household: how many people the shopping list is for.

Diet, allergies (listed and other), avoid, dislikes and max_minutes are hard rules: a recipe that breaks one is never shown.
They are checked on each ingredient's name, its full phrase and RecipeDB's food group (Meat, Fish,
Seafood, Dairy, Beverage-Alcoholic), and for the diet also on RecipeDB's own diet label, so a recipe passes
only when both agree. Matching is deliberately cautious ("butter" counts as milk unless it is peanut or
almond butter), but it reads recipe text, not product labels: the app tells patients with allergies to
check the labels of what they buy.
"""
import re
from dataclasses import dataclass, field

DIETS = ('anything', 'pescatarian', 'vegetarian', 'eggless_vegetarian', 'vegan')
ALLERGENS = ('milk', 'egg', 'fish', 'shellfish', 'tree_nuts', 'peanuts', 'wheat', 'gluten', 'soy', 'sesame')
AVOIDABLE = ('pork', 'beef', 'alcohol')

# Words that mark each group, and look-alikes that do not (checked first).
WORDS = {
    'meat': ('beef', 'steak', 'veal', 'pork', 'bacon', 'ham', 'sausage', 'chorizo', 'salami', 'pepperoni', 'prosciutto',
             'pancetta', 'kielbasa', 'lamb', 'mutton', 'goat', 'chicken', 'turkey', 'duck', 'venison', 'bison', 'meat',
             'meatball', 'hot dog', 'frankfurter', 'bratwurst', 'lard', 'gelatin', 'jello', 'marshmallow', 'bouillon',
             'broth', 'stock', 'consomme', 'drippings', 'suet', 'liver', 'chuck', 'sirloin', 'brisket', 'tenderloin',
             'ribs', 'roast beef', 'corned', 'pastrami', 'giblet', 'anchovy paste'),
    'fish': ('fish', 'salmon', 'tuna', 'cod', 'tilapia', 'anchovy', 'anchovies', 'sardine', 'halibut', 'trout', 'mackerel',
             'haddock', 'catfish', 'snapper', 'bass', 'swordfish', 'mahi', 'flounder', 'sole', 'pollock', 'herring',
             'worcestershire', 'caesar dressing', 'bonito', 'dashi'),
    'shellfish': ('shrimp', 'prawn', 'crab', 'lobster', 'clam', 'mussel', 'oyster', 'scallop', 'crawfish', 'crayfish',
                  'squid', 'calamari', 'octopus', 'langoustine', 'oyster sauce'),
    'milk': ('milk', 'cheese', 'butter', 'cream', 'yogurt', 'yoghurt', 'whey', 'casein', 'ghee', 'buttermilk', 'parmesan',
             'mozzarella', 'cheddar', 'ricotta', 'feta', 'brie', 'paneer', 'half-and-half', 'half and half', 'ice cream',
             'custard', 'kefir', 'curd', 'margarine', 'alfredo', 'queso', 'mascarpone', 'gruyere', 'romano', 'velveeta',
             'whipped topping', 'pudding', 'chocolate chips', 'milk chocolate'),
    'egg': ('egg', 'eggs', 'mayonnaise', 'mayo', 'meringue', 'aioli', 'egg noodles', 'eggnog'),
    'tree_nuts': ('almond', 'walnut', 'pecan', 'cashew', 'pistachio', 'hazelnut', 'filbert', 'macadamia', 'pine nut',
                  'pignoli', 'brazil nut', 'nutella', 'praline', 'marzipan', 'amaretto', 'mixed nuts', 'nuts'),
    'peanuts': ('peanut', 'peanuts', 'groundnut', 'satay', 'mixed nuts', 'nuts'),
    'wheat': ('wheat', 'flour', 'bread', 'breadcrumb', 'bread crumbs', 'panko', 'pasta', 'spaghetti', 'macaroni',
              'noodle', 'lasagna', 'linguine', 'fettuccine', 'penne', 'rigatoni', 'orzo', 'couscous', 'bulgur', 'semolina',
              'durum', 'farro', 'spelt', 'seitan', 'tortilla', 'pita', 'bun', 'roll', 'bagel', 'croissant', 'crouton',
              'cracker', 'biscuit', 'pie crust', 'pastry', 'puff pastry', 'phyllo', 'cake mix', 'muffin', 'pancake mix',
              'bisquick', 'stuffing', 'soy sauce', 'teriyaki', 'hoisin', 'beer', 'cereal', 'graham', 'wonton', 'dumpling',
              'gnocchi', 'ravioli', 'tortellini', 'cookie', 'brownie mix', 'crescent', 'dinner rolls'),
    'gluten_extra': ('barley', 'rye', 'malt', 'pumpernickel'),
    'soy': ('soy', 'soya', 'tofu', 'edamame', 'tempeh', 'miso', 'tamari', 'teriyaki', 'hoisin', 'textured vegetable protein'),
    'sesame': ('sesame', 'tahini', 'halva', 'hummus', 'za\'atar'),
    'honey': ('honey',),
    'pork': ('pork', 'bacon', 'ham', 'sausage', 'chorizo', 'salami', 'pepperoni', 'prosciutto', 'pancetta', 'kielbasa',
             'lard', 'hot dog', 'bratwurst', 'gelatin', 'jello', 'marshmallow', 'spare ribs', 'baby back ribs'),
    'beef': ('beef', 'steak', 'veal', 'chuck', 'sirloin', 'brisket', 'corned', 'pastrami', 'roast beef', 'hamburger',
             'ground round', 'oxtail', 'short ribs', 'bouillon cube'),
    'alcohol': ('wine', 'beer', 'ale', 'lager', 'rum', 'vodka', 'gin', 'bourbon', 'whiskey', 'whisky', 'brandy', 'cognac',
                'sherry', 'liqueur', 'tequila', 'sake', 'mirin', 'vermouth', 'kahlua', 'amaretto', 'triple sec',
                'grand marnier', 'champagne', 'marsala', 'port', 'schnapps', 'cointreau', 'baileys'),
}
NOT = {
    'meat': ('vegetable broth', 'vegetable stock', 'veggie broth', 'mushroom broth', 'vegetable bouillon',
             'chicken-flavored vegetarian', 'meatless', 'vegetarian', 'vegan', 'fish stock', 'fish broth', 'chickpea',
             'chick pea', 'eggplant', 'hamburger buns', 'hamburger bun', 'steak fries', 'steak seasoning',
             'bacon tray'),
    'fish': ('fish sauce substitute', 'fishless', 'vegan worcestershire', 'sole meunière seasoning'),
    'milk': ('coconut milk', 'almond milk', 'soy milk', 'soymilk', 'rice milk', 'oat milk', 'coconut cream',
             'cocoa butter', 'peanut butter', 'almond butter', 'cashew butter', 'apple butter', 'nut butter',
             'cream of tartar', 'butternut', 'butter beans', 'buttercup squash', 'non-dairy', 'nondairy', 'dairy-free',
             'vegan butter', 'vegan cheese', 'milkweed', 'coconut yogurt', 'cream soda', 'ice cream cone',
             'dark chocolate chips', 'vegan chocolate', 'bean curd'),
    'egg': ('eggplant', 'egg substitute', 'egg replacer', 'egg roll wrappers'),
    'tree_nuts': ('nutmeg', 'butternut', 'water chestnut', 'chestnut mushroom', 'coconut', 'doughnut', 'donut',
                  'nut-free', 'peanut', 'peanuts', 'almond extract substitute'),
    'peanuts': ('peanut-free',),
    'wheat': ('buckwheat', 'rice flour', 'corn flour', 'cornflour', 'almond flour', 'coconut flour', 'oat flour',
              'chickpea flour', 'gram flour', 'tapioca flour', 'potato flour', 'gluten-free', 'gluten free',
              'rice noodle', 'rice noodles', 'glass noodles', 'cellophane noodles', 'soba', 'corn tortilla',
              'corn tortillas', 'rice paper', 'rice cracker', 'cauliflower', 'tamari', 'coconut aminos',
              'root beer', 'egg roll', 'cornbread mix substitute'),
    'gluten_extra': ('malt vinegar substitute',),
    'shellfish': (),
    'soy': ('soy-free',),
    'sesame': (),
    'honey': ('honeydew', 'honey substitute', 'agave'),
    'pork': ('turkey bacon', 'turkey sausage', 'chicken sausage', 'veggie sausage', 'vegetarian sausage', 'beef sausage',
             'turkey ham', 'eggplant', 'vegetarian bacon', 'hamburger', 'graham', 'gelatin-free'),
    'beef': ('beefsteak tomato', 'beefsteak tomatoes', 'vegetarian', 'hamburger buns', 'hamburger bun'),
    'alcohol': ('wine vinegar', 'rice vinegar', 'sherry vinegar', 'non-alcoholic', 'alcohol-free', 'ginger', 'ginger ale',
                'root beer', 'portobello', 'port wine cheese', 'rum extract', 'brandy extract', 'sparkling water',
                'gingerbread'),
}
GROUPS = {'meat': {'Meat'}, 'fish': {'Fish'}, 'shellfish': {'Seafood'}, 'milk': {'Dairy'},
          'alcohol': {'Beverage-Alcoholic'}}
RECIPEDB_DIETS = {'anything': None, 'pescatarian': {'pescetarian', 'pescatarian', 'vegetarian', 'vegan'},
                  'vegetarian': {'vegetarian', 'vegan'}, 'eggless_vegetarian': {'vegetarian', 'vegan'},
                  'vegan': {'vegan'}}
DIET_RULES = {'anything': (), 'pescatarian': ('meat',), 'vegetarian': ('meat', 'fish', 'shellfish'),
              'eggless_vegetarian': ('meat', 'fish', 'shellfish', 'egg'),
              'vegan': ('meat', 'fish', 'shellfish', 'egg', 'milk', 'honey')}
def _pattern(words):
    return re.compile(r'\b(' + '|'.join(re.escape(w) + r'(?:e?s)?' for w in sorted(words, key=len, reverse=True))
                      + r')\b')


PATTERNS = {g: _pattern(w) for g, w in WORDS.items()}
EXCEPTIONS = {g: _pattern(w) if w else None for g, w in NOT.items()}


@dataclass(frozen=True)
class FoodPreferences:
    diet: str = 'anything'
    allergies: tuple = ()
    other_allergies: tuple = ()
    avoid: tuple = ()
    dislikes: tuple = ()
    likes: tuple = ()
    max_minutes: int = None
    household: int = 1

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        clean = lambda xs: tuple(sorted({str(x).strip().casefold() for x in xs or () if str(x).strip()}))
        prefs = cls(diet=data.get('diet') or 'anything', allergies=clean(data.get('allergies')),
                    other_allergies=clean(data.get('other_allergies')),
                    avoid=clean(data.get('avoid')), dislikes=clean(data.get('dislikes')), likes=clean(data.get('likes')),
                    max_minutes=data.get('max_minutes'), household=int(data.get('household') or 1))
        if prefs.diet not in DIETS:
            raise ValueError(f'diet must be one of {", ".join(DIETS)}')
        bad = [a for a in prefs.allergies if a not in ALLERGENS] + [a for a in prefs.avoid if a not in AVOIDABLE]
        if bad:
            raise ValueError(f'unknown allergy or food to avoid: {", ".join(bad)}')
        return prefs

    def as_dict(self):
        return {'diet': self.diet, 'allergies': list(self.allergies), 'other_allergies': list(self.other_allergies), 'avoid': list(self.avoid),
                'dislikes': list(self.dislikes), 'likes': list(self.likes), 'max_minutes': self.max_minutes,
                'household': self.household}

    @property
    def is_default(self):
        """No hard rule set, so every recipe is allowed (likes and household size do not filter)."""
        return (self.diet == 'anything' and not (self.allergies or self.other_allergies or self.avoid or self.dislikes)
                and not self.max_minutes)


def _texts(ingredient):
    if isinstance(ingredient, str):
        return [ingredient.casefold()], ''
    name = ingredient.get('name_lc') or str(ingredient.get('name', '')).casefold()
    phrase = str(ingredient.get('ingredient_phrase', '')).casefold()
    return [name, phrase], ingredient.get('predicted_category', '') or ''


def _has(group, text):
    """Whether the text names the group, after removing its look-alikes ("eggplant" is not an egg)."""
    exceptions = EXCEPTIONS.get(group)
    return bool(PATTERNS[group].search(exceptions.sub(' ', text) if exceptions else text))


def contains(recipe, group):
    """Whether any ingredient of the recipe belongs to the group (by its food group or its words)."""
    for ingredient in recipe.get('ingredients') or []:
        texts, category = _texts(ingredient)
        exceptions = EXCEPTIONS.get(group)
        if category in GROUPS.get(group, ()) and not any(exceptions and exceptions.search(t) for t in texts):
            return True
        if any(_has(group, t) for t in texts if t):
            return True
    return False


def _word_in(word, recipe):
    pattern = _pattern([word])
    return any(pattern.search(t) for i in recipe.get('ingredients') or [] for t in _texts(i)[0] if t)


def _minutes(recipe):
    try:
        return float(recipe.get('total_time'))
    except (TypeError, ValueError):
        return None


def breaks(recipe, prefs):
    """The reasons a recipe is not allowed for these preferences (empty when allowed)."""
    reasons = []
    allowed = RECIPEDB_DIETS[prefs.diet]
    if allowed is not None and str(recipe.get('diet', '')).casefold() not in allowed:
        reasons.append(f"labelled {recipe.get('diet')}")
    reasons += [f'contains {g}' for g in DIET_RULES[prefs.diet] if contains(recipe, g)]
    for allergen in prefs.allergies:
        groups = ('wheat', 'gluten_extra') if allergen == 'gluten' else (allergen,)
        reasons += [f'allergen: {allergen}' for g in groups if contains(recipe, g)][:1]
    reasons += [f'allergen: {w}' for w in prefs.other_allergies if _word_in(w, recipe)]
    reasons += [f'avoid: {a}' for a in prefs.avoid if contains(recipe, a)]
    reasons += [f'disliked: {w}' for w in prefs.dislikes if _word_in(w, recipe)]
    minutes = _minutes(recipe)
    if prefs.max_minutes and minutes is not None and minutes > prefs.max_minutes:
        reasons.append(f'takes {minutes:.0f} minutes')
    return reasons


def allowed_recipes(recipes, prefs):
    """The recipes these preferences allow, in their original order."""
    prefs = prefs if isinstance(prefs, FoodPreferences) else FoodPreferences.from_dict(prefs)
    if prefs.is_default:
        return list(recipes)
    return [r for r in recipes if not breaks(r, prefs)]


def liked_dish(prefs):
    """The liked ingredients as a pseudo-dish, added to the favourite dishes so that recipes sharing them
    count as closer to what the patient enjoys (style_ladder.closeness); None without likes."""
    if not prefs.likes:
        return None
    return {'recipe_id': 'liked', 'title': 'Ingredients you like', 'ingredients': list(prefs.likes)}


def questions():
    """The sign-up food questions and their options, for the app to show."""
    return {
        'diet': {'question': 'How do you eat?', 'options': list(DIETS),
                 'labels': ['I eat everything', 'Fish but no meat', 'Vegetarian (eggs and dairy are fine)',
                            'Vegetarian without eggs', 'Vegan (no animal products)']},
        'allergies': {'question': 'Are you allergic to any of these?', 'options': list(ALLERGENS), 'multiple': True,
                      'note': 'We remove recipes that list these, but always check the labels of what you buy.'},
        'other_allergies': {'question': 'Other allergies', 'free_text': True, 'example': 'kiwi, mustard'},
        'avoid': {'question': 'Is there anything you never eat?', 'options': list(AVOIDABLE), 'multiple': True},
        'dislikes': {'question': 'Ingredients you dislike', 'free_text': True, 'example': 'mushroom, cilantro'},
        'likes': {'question': 'Ingredients you enjoy', 'free_text': True, 'example': 'beans, chicken, sweet potato'},
        'max_minutes': {'question': 'How long can you spend cooking a meal?', 'options': [20, 30, 45, 60, None],
                        'labels': ['20 minutes', '30 minutes', '45 minutes', 'An hour', 'No limit']},
        'household': {'question': 'How many people do you cook for?', 'options': [1, 2, 3, 4, 5, 6]},
    }
