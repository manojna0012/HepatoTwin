import os

import requests
from dotenv import load_dotenv
from recipe_parser import parse_recipe


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

    parsed = parse_recipe(recipe)

    print("\n--- METADATA ---")
    print(parsed["metadata"])

    print("\n--- PREPARATION ---")
    print(parsed["preparation"])

    print("\n--- NUTRITION ---")
    print(parsed["nutrition"])

    print("\n--- INGREDIENTS ---")
    for ingredient in parsed["ingredients"]:
        print(ingredient)

    print("\n--- INSTRUCTIONS ---")
    for number, instruction in enumerate(parsed["instructions"], start=1):
        print(f"{number}. {instruction}")