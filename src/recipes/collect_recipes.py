import csv
import os

from recipedb_client import get_recipe
from recipe_parser import parse_recipe


OUTPUT_DIR = "data/recipes"


def collect_recipes(recipe_ids):
    all_metadata = []
    all_preparation = []
    all_nutrition = []
    all_ingredients = []
    all_instructions = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for recipe_id in recipe_ids:
        try:
            recipe = get_recipe(recipe_id)
            parsed = parse_recipe(recipe)

            all_metadata.append(parsed["metadata"])
            all_preparation.append(parsed["preparation"])
            all_nutrition.append(parsed["nutrition"])

            all_ingredients.extend(parsed["ingredients"])

            for step_number, instruction in enumerate(
                parsed["instructions"], start=1
            ):
                all_instructions.append({
                    "recipe_id": recipe.get("Recipe_ID"),
                    "step_number": step_number,
                    "instruction": instruction
                })

            print(
                f"Processed {recipe_id}: "
                f"{recipe.get('Recipe_Title')}"
            )

        except Exception as e:
            print(f"Failed to process {recipe_id}: {e}")

    return (
        all_metadata,
        all_preparation,
        all_nutrition,
        all_ingredients,
        all_instructions
    )


def save_csv(data, filename):
    if not data:
        print(f"No data to save for {filename}")
        return

    filepath = os.path.join(OUTPUT_DIR, filename)

    fieldnames = data[0].keys()

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved: {filepath}")


if __name__ == "__main__":

    recipe_ids = [
        str(i).zfill(8)
        for i in range(1, 31)
    ]

    (
        metadata,
        preparation,
        nutrition,
        ingredients,
        instructions
    ) = collect_recipes(recipe_ids)

    save_csv(metadata, "recipe_metadata.csv")
    save_csv(preparation, "recipe_preparation.csv")
    save_csv(nutrition, "recipe_nutrition.csv")
    save_csv(ingredients, "recipe_ingredients.csv")
    save_csv(instructions, "recipe_instructions.csv")