import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import os


load_dotenv()

BASE_URL = os.getenv("RECIPEDB_BASE_URL")

OUTPUT_FILE = Path("data/processed/recipe_ids.json")


def main():

    if not BASE_URL:
        raise ValueError("RECIPEDB_BASE_URL is not set")

    all_recipe_ids = []
    page = 1

    while True:

        print(f"Fetching page {page}...")

        url = f"{BASE_URL}/recipesinfo"

        response = requests.get(
            url,
            params={"page": page},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        for recipe in results:
            recipe_id = recipe.get("Recipe_ID")

            if recipe_id:
                all_recipe_ids.append(str(recipe_id))

        print(f"Recipes collected so far: {len(all_recipe_ids)}")

        if not data.get("hasNext", False):
            break

        page += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_recipe_ids, f, indent=2)

    print("\n===== COMPLETE =====")
    print("Pages processed:", page)
    print("Recipe IDs collected:", len(all_recipe_ids))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()