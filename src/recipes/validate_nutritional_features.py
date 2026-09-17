import json
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/recipe_nutritional_features.json"
)

REQUIRED_FEATURES = [
    "recipe_id",
    "title",
    "calories",
    "protein",
    "carbohydrates",
    "fat",
    "saturated_fat",
    "cholesterol",
    "fiber",
    "sugar",
    "sodium",
]


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    print("===== NUTRITIONAL FEATURE VALIDATION =====")

    # 1. Number of profiles
    print("\nNumber of profiles:", len(profiles))

    # 2. Unique recipe IDs
    recipe_ids = [
        profile.get("recipe_id")
        for profile in profiles
    ]

    unique_ids = set(recipe_ids)

    print("Unique recipe IDs:", len(unique_ids))
    print("Duplicate recipe IDs:", len(recipe_ids) - len(unique_ids))

    # 3. Check required features
    missing_features = []

    for i, profile in enumerate(profiles):

        for feature in REQUIRED_FEATURES:

            if feature not in profile:
                missing_features.append(
                    (i, feature)
                )

    print(
        "Missing required fields:",
        len(missing_features)
    )

    if missing_features:
        print(missing_features)

    # 4. Check missing values
    numeric_features = [
        "calories",
        "protein",
        "carbohydrates",
        "fat",
        "saturated_fat",
        "cholesterol",
        "fiber",
        "sugar",
        "sodium",
    ]

    missing_values = 0
    negative_values = 0

    for profile in profiles:

        for feature in numeric_features:

            value = profile.get(feature)

            if value is None:
                missing_values += 1

            elif value < 0:
                negative_values += 1

    print("Missing nutritional values:", missing_values)
    print("Negative nutritional values:", negative_values)

    # 5. Nutritional ranges
    print("\n===== NUTRITIONAL RANGES =====")

    for feature in numeric_features:

        values = [
            profile[feature]
            for profile in profiles
            if profile.get(feature) is not None
        ]

        if values:
            print(
                f"{feature}: "
                f"{min(values)} -> {max(values)}"
            )

    print("\n===== VALIDATION COMPLETE =====")


if __name__ == "__main__":
    main()