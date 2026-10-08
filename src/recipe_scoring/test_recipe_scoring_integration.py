
import json
from pathlib import Path

from practical_feasibility import practical_feasibility_score
from cuisine_compatibility import cuisine_compatibility_score
from health_alignment import health_alignment_score


DATA_FILE = Path("data/processed/recipe_objects.json")

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    print("Recipes loaded:", len(recipes))

    objectives = [
        {"nutrient": "sodium", "direction": "lower",
         "target": 500, "tolerance": 1000, "weight": 1},
        {"nutrient": "fiber", "direction": "higher",
         "target": 10, "tolerance": 10, "weight": 1},
    ]

    score_data = {
        "feasibility": [],
        "cuisine": [],
        "health": [],
    }
    missing = {key: 0 for key in score_data}
    errors = []

    for recipe in recipes:
        try:
            results = {
                "feasibility": practical_feasibility_score(recipe),
                "cuisine": cuisine_compatibility_score(
                    recipe, recipe.get("cuisine")
                ),
                "health": health_alignment_score(recipe, objectives),
            }

            for name, result in results.items():
                score = result["score"]

                if score is None:
                    missing[name] += 1
                elif not isinstance(score, (int, float)) or not (
                    0 <= score <= 100
                ):
                    raise ValueError(f"{name} score out of range: {score}")
                else:
                    score_data[name].append(score)

        except Exception as exc:
            errors.append(
                f"{recipe.get('recipe_id', 'unknown')}: {exc}"
            )

    print("\nRecipes tested:", len(recipes))
    print("Errors:", len(errors))

    for name, scores in score_data.items():
        print(f"\n{name.upper()}")
        print("Scored:", len(scores))
        print("Missing:", missing[name])

        if scores:
            print(f"Minimum: {min(scores):.2f}")
            print(f"Maximum: {max(scores):.2f}")
            print(f"Mean: {sum(scores) / len(scores):.2f}")

    if errors:
        print("\nFirst errors:")
        for error in errors[:10]:
            print("-", error)

    assert not errors, "Integration errors found."
    print("\nFull-dataset validation passed.")


if __name__ == "__main__":
    main()
