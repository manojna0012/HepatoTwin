import json
from pathlib import Path

from nutritional_features import create_nutritional_features


INPUT_FILE = Path(
    "data/processed/recipe_nutrient_profiles.json"
)

OUTPUT_FILE = Path(
    "data/processed/recipe_nutritional_features.json"
)


def main():

    # Load existing nutrient profiles
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    nutritional_features = []

    # Create nutritional feature profile for each recipe
    for profile in profiles:

        features = create_nutritional_features(profile)

        nutritional_features.append(features)

    # Make sure output directory exists
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save nutritional feature profiles
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            nutritional_features,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("===== NUTRITIONAL FEATURE PROFILES =====")
    print("Number of recipes:", len(nutritional_features))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()