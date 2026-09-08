import json
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/recipe_nutrient_profiles.json"
)


NUTRIENT_FIELDS = [
    "calories",
    "fat",
    "saturated_fat",
    "cholesterol",
    "sodium",
    "carbohydrates",
    "fiber",
    "sugar",
    "protein",
]


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    print("===== NUTRIENT PROFILE VALIDATION =====")
    print("Number of profiles:", len(profiles))

    missing_profiles = 0
    missing_values = 0

    for profile in profiles:

        if "recipe_id" not in profile:
            missing_profiles += 1
            continue

        for nutrient in NUTRIENT_FIELDS:

            if profile.get(nutrient) is None:
                missing_values += 1
                print(
                    f"Missing {nutrient} "
                    f"for recipe {profile['recipe_id']}"
                )

    print("\nMissing recipe IDs:", missing_profiles)
    print("Missing nutrient values:", missing_values)

    print("\nFirst profile:")
    print(profiles[0])

    print("\n===== VALIDATION COMPLETE =====")


if __name__ == "__main__":
    main()