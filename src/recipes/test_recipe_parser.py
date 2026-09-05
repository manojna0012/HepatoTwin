from recipedb_client import get_recipe
from recipe_parser import parse_recipe


recipe_ids = [
    "00000001",
    "00000002",
    "00000003",
    "00000004",
    "00000005"
]


for recipe_id in recipe_ids:
    try:
        recipe = get_recipe(recipe_id)
        parsed = parse_recipe(recipe)

        print("\n" + "=" * 60)
        print(f"Recipe ID: {recipe_id}")
        print(f"Title: {parsed['metadata']['title']}")
        print(f"Cuisine: {parsed['metadata']['cuisine']}")
        print(f"Category: {parsed['metadata']['category']}")
        print(f"Ingredients: {len(parsed['ingredients'])}")
        print(f"Instructions: {len(parsed['instructions'])}")

    except Exception as e:
        print(f"\nRecipe {recipe_id} failed: {e}")