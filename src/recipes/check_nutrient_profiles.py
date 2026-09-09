import json
from pathlib import Path

# NEED TO LOAD JSON FILE AND CHECK IF ITS FINE (LIKE DOES IT HAVE NEGATIVE VALUES ETC)

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
    negative_values = 0

    # Check recipe IDs and missing nutrient values
    for profile in profiles:

        if "recipe_id" not in profile:
            missing_profiles += 1
            continue

        for nutrient in NUTRIENT_FIELDS:

            value = profile.get(nutrient)

            if value is None:
                missing_values += 1
                print(
                    f"Missing {nutrient} "
                    f"for recipe {profile['recipe_id']}"
                )

            elif value < 0:
                negative_values += 1
                print(
                    f"Negative {nutrient} "
                    f"for recipe {profile['recipe_id']}: {value}"
                )

    print("\nMissing recipe IDs:", missing_profiles)
    print("Missing nutrient values:", missing_values)
    print("Negative nutrient values:", negative_values)

    # Range checks
    print("\n===== NUTRIENT RANGES =====")

    for nutrient in NUTRIENT_FIELDS:

        values = [
            profile[nutrient]
            for profile in profiles
            if profile.get(nutrient) is not None
        ]

        if values:
            print(
                f"{nutrient}: "
                f"{min(values)} -> {max(values)}"
            )
        else:
            print(f"{nutrient}: No valid values")

    print("\nFirst profile:")
    print(profiles[0])

    print("\n===== VALIDATION COMPLETE =====")


if __name__ == "__main__":
    main()