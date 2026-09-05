import os

import requests
from dotenv import load_dotenv


load_dotenv()

BASE_URL = os.getenv("RECIPEDB_BASE_URL")


def get_recipe(recipe_id):
    """
    Fetch a recipe from RecipeDB using its Recipe_ID.
    """

    if not BASE_URL:
        raise ValueError("RECIPEDB_BASE_URL is not set")

    recipe_id = str(recipe_id).zfill(8)

    url = f"{BASE_URL}/search-recipe/{recipe_id}"

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    recipe = get_recipe("00000001")

    print("Recipe ID:", recipe.get("Recipe_ID"))
    print("Title:", recipe.get("Recipe_Title"))
    print("Cuisine:", recipe.get("Cuisine"))
    print("Category:", recipe.get("Category"))