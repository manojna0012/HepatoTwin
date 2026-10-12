from src.recipe_scoring.patient_objectives import build_patient_objectives
from src.recipe_scoring.health_alignment import health_alignment_score

def test_priority_weights_are_applied():
    profile = {
        "priorities": [
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "high",
                "target": 500,
                "tolerance": 1000,
            },
            {
                "nutrient": "fiber",
                "direction": "higher",
                "priority": "medium",
                "target": 10,
                "tolerance": 10,
            },
        ]
    }

    objectives = build_patient_objectives(profile)

    assert objectives[0]["weight"] == 3.0
    assert objectives[1]["weight"] == 2.0


def test_duplicate_nutrients_are_rejected():
    profile = {
        "priorities": [
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "high",
                "target": 500,
                "tolerance": 1000,
            },
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "low",
                "target": 400,
                "tolerance": 1000,
            },
        ]
    }

    try:
        build_patient_objectives(profile)
    except ValueError:
        pass
    else:
        assert False, "Duplicate nutrient should be rejected"

def test_patient_objectives_work_with_health_alignment():
    profile = {
        "priorities": [
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "high",
                "target": 500,
                "tolerance": 1000,
            },
            {
                "nutrient": "fiber",
                "direction": "higher",
                "priority": "medium",
                "target": 10,
                "tolerance": 10,
            },
        ]
    }

    recipe = {
        "nutrition": {
            "sodium": 500,
            "fiber": 10,
        }
    }

    objectives = build_patient_objectives(profile)
    result = health_alignment_score(recipe, objectives)

    assert result["score"] == 100
    assert result["coverage"] == 1.0
    assert result["warnings"] == []


def test_different_patients_can_rank_recipes_differently():
    patient_a = {
        "priorities": [
            {
                "nutrient": "sodium",
                "direction": "lower",
                "priority": "high",
                "target": 300,
                "tolerance": 1000,
            }
        ]
    }

    patient_b = {
        "priorities": [
            {
                "nutrient": "fiber",
                "direction": "higher",
                "priority": "high",
                "target": 10,
                "tolerance": 10,
            }
        ]
    }

    recipes = [
        {"nutrition": {"sodium": 200, "fiber": 2}},
        {"nutrition": {"sodium": 800, "fiber": 10}},
    ]

    objectives_a = build_patient_objectives(patient_a)
    objectives_b = build_patient_objectives(patient_b)

    scores_a = [
        health_alignment_score(recipe, objectives_a)["score"]
        for recipe in recipes
    ]
    scores_b = [
        health_alignment_score(recipe, objectives_b)["score"]
        for recipe in recipes
    ]

    assert scores_a[0] > scores_a[1]
    assert scores_b[1] > scores_b[0]

def test_unsupported_nutrient_is_rejected():
    profile = {
        "priorities": [
            {
                "nutrient": "unknown_nutrient",
                "direction": "lower",
                "priority": "high",
                "target": 10,
                "tolerance": 5,
            }
        ]
    }

    try:
        build_patient_objectives(profile)
    except ValueError as exc:
        assert "Unsupported nutrient" in str(exc)
    else:
        assert False, "Unsupported nutrient should be rejected"