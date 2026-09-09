import json
from pathlib import Path

from recipedb_client import get_recipe
from recipe_parser import parse_recipe
from recipe_model import create_recipe
from nutrient_profile import create_nutrient_profile
from dataclasses import asdict

# SAVES 30 NUTRIENT PROFILES IN JSON FILE FOR 30 RECIPES

OUTPUT_FILE = Path("data/processed/recipe_nutrient_profiles.json")
RECIPE_OBJECTS_FILE = Path("data/processed/recipe_objects.json")

RECIPE_IDS = [
    f"{i:08d}" for i in range(1, 31)
]


def main():

    profiles = []
    recipes = []

    for recipe_id in RECIPE_IDS:

        print(f"Processing recipe {recipe_id}...")

        try:
            recipe_data = get_recipe(recipe_id)
            parsed = parse_recipe(recipe_data)
            recipe = create_recipe(parsed)
            recipes.append(asdict(recipe))

            profile = create_nutrient_profile(recipe)

            profiles.append(profile)

        except Exception as e:
            print(f"Failed for {recipe_id}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    
    with open(RECIPE_OBJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2)

    print("\n===== COMPLETE =====")
    print("Profiles created:", len(profiles))
    print("Saved to:", OUTPUT_FILE)
    print("Recipe objects created:", len(recipes))
    print("Saved to:", RECIPE_OBJECTS_FILE)


if __name__ == "__main__":
    main()