"""Manual check against the live RecipeDB API: prints the nutrient profile for one recipe.

Not a pytest test (no assertions; needs RECIPEDB_BASE_URL in .env). Run from
src/recipes so the sibling modules import:

    ../../.venv/Scripts/python.exe check_live_nutrient_profile.py
"""
from recipedb_client import get_recipe
from recipe_parser import parse_recipe
from recipe_model import create_recipe
from nutrient_profile import create_nutrient_profile

#TESTS FOR ONE RECIPE

if __name__ == "__main__":

    recipe_data = get_recipe("00000001")
    parsed = parse_recipe(recipe_data)
    recipe = create_recipe(parsed)

    nutrient_profile = create_nutrient_profile(recipe)

    print("===== NUTRIENT PROFILE =====")

    for key, value in nutrient_profile.items():
        print(f"{key}: {value}")