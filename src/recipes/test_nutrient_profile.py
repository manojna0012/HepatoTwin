from recipedb_client import get_recipe
from recipe_parser import parse_recipe
from recipe_model import create_recipe
from nutrient_profile import create_nutrient_profile


if __name__ == "__main__":

    recipe_data = get_recipe("00000001")
    parsed = parse_recipe(recipe_data)
    recipe = create_recipe(parsed)

    nutrient_profile = create_nutrient_profile(recipe)

    print("===== NUTRIENT PROFILE =====")

    for key, value in nutrient_profile.items():
        print(f"{key}: {value}")