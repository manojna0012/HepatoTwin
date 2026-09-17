import json
from pathlib import Path
from dataclasses import asdict

from recipedb_client import get_recipe
from recipe_parser import parse_recipe
from recipe_model import create_recipe
from nutrient_profile import create_nutrient_profile


# INPUT: REAL RECIPE IDS COLLECTED FROM RECIPEDB
RECIPE_IDS_FILE = Path("data/processed/recipe_ids.json")

# OUTPUT FILES
OUTPUT_FILE = Path("data/processed/recipe_nutrient_profiles.json")
RECIPE_OBJECTS_FILE = Path("data/processed/recipe_objects.json")


def main():

    # Load real RecipeDB recipe IDs
    with open(RECIPE_IDS_FILE, "r", encoding="utf-8") as f:
        recipe_ids = json.load(f)

    print("Recipe IDs loaded:", len(recipe_ids))

    profiles = []
    recipes = []

    for index, recipe_id in enumerate(recipe_ids, start=1):

        print(f"Processing {index}/{len(recipe_ids)}: {recipe_id}")

        try:
            recipe_data = get_recipe(recipe_id)

            parsed = parse_recipe(recipe_data)

            recipe = create_recipe(parsed)

            recipes.append(asdict(recipe))

            profile = create_nutrient_profile(recipe)

            profiles.append(profile)

        except Exception as e:
            print(f"Failed for {recipe_id}: {e}")

    # Create output directory if needed
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save nutrient profiles
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            profiles,
            f,
            indent=2,
            ensure_ascii=False
        )

    # Save recipe objects
    with open(RECIPE_OBJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            recipes,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n===== COMPLETE =====")
    print("Recipe IDs loaded:", len(recipe_ids))
    print("Profiles created:", len(profiles))
    print("Recipe objects created:", len(recipes))
    print("Saved profiles to:", OUTPUT_FILE)
    print("Saved recipe objects to:", RECIPE_OBJECTS_FILE)


if __name__ == "__main__":
    main()