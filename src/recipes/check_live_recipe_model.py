"""Manual check against the live RecipeDB API: prints the Recipe object for one recipe.

Not a pytest test (no assertions; needs RECIPEDB_BASE_URL in .env). Run from
src/recipes so the sibling modules import:

    ../../.venv/Scripts/python.exe check_live_recipe_model.py
"""
from recipedb_client import get_recipe
from recipe_parser import parse_recipe
from recipe_model import create_recipe

# TEST FOR ONE RECIPE


if __name__ == "__main__":

    recipe_data = get_recipe("00000001")

    parsed = parse_recipe(recipe_data)

    recipe = create_recipe(parsed)

    print("===== RECIPE OBJECT =====")
    print("ID:", recipe.recipe_id)
    print("Title:", recipe.title)
    print("Cuisine:", recipe.cuisine)
    print("Category:", recipe.category)
    print("Servings:", recipe.servings)

    print("\n===== PREPARATION =====")
    print("Prep time:", recipe.prep_time)
    print("Cook time:", recipe.cook_time)
    print("Total time:", recipe.total_time)

    print("\n===== NUTRITION =====")
    print(recipe.nutrition)

    print("\n===== INGREDIENTS =====")
    print("Number of ingredients:", len(recipe.ingredients))

    print("\n===== INSTRUCTIONS =====")
    print("Number of instructions:", len(recipe.instructions))
