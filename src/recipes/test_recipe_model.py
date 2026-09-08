from recipedb_client import get_recipe
from recipe_parser import parse_recipe
from recipe_model import create_recipe


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
