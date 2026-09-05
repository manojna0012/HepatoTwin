import pandas as pd


DATA_DIR = "data/recipes"


def main():
    metadata = pd.read_csv(
        f"{DATA_DIR}/recipe_metadata.csv"
    )

    preparation = pd.read_csv(
        f"{DATA_DIR}/recipe_preparation.csv"
    )

    nutrition = pd.read_csv(
        f"{DATA_DIR}/recipe_nutrition.csv"
    )

    ingredients = pd.read_csv(
        f"{DATA_DIR}/recipe_ingredients.csv"
    )

    instructions = pd.read_csv(
        f"{DATA_DIR}/recipe_instructions.csv"
    )

    print("\n===== DATASET SIZES =====")

    print(f"Recipes:       {len(metadata)}")
    print(f"Preparation:   {len(preparation)}")
    print(f"Nutrition:     {len(nutrition)}")
    print(f"Ingredients:   {len(ingredients)}")
    print(f"Instructions:  {len(instructions)}")

    print("\n===== RECIPE ID CHECK =====")

    print(
        "Unique recipe IDs:",
        metadata["recipe_id"].nunique()
    )

    print(
        "Duplicate recipe IDs:",
        metadata["recipe_id"].duplicated().sum()
    )

    print("\n===== MISSING VALUES =====")

    print("\nMetadata:")
    print(metadata.isna().sum())

    print("\nPreparation:")
    print(preparation.isna().sum())

    print("\nNutrition:")
    print(nutrition.isna().sum())

    print("\nIngredients:")
    print(ingredients.isna().sum())

    print("\nInstructions:")
    print(instructions.isna().sum())

    print("\n===== INGREDIENTS PER RECIPE =====")

    ingredients_per_recipe = (
        ingredients.groupby("recipe_id")
        .size()
    )

    print(ingredients_per_recipe.describe())

    print("\n===== INSTRUCTIONS PER RECIPE =====")

    instructions_per_recipe = (
        instructions.groupby("recipe_id")
        .size()
    )

    print(instructions_per_recipe.describe())

    print("\n===== PREPARATION TIME CHECK =====")

    invalid_times = preparation[
        (preparation["prep_time"] < 0)
        | (preparation["cook_time"] < 0)
        | (preparation["total_time"] < 0)
    ]

    print(
        "Recipes with negative times:",
        len(invalid_times)
    )

    print("\n===== DATA CHECK COMPLETE =====")


if __name__ == "__main__":
    main()