
"""Conservative, evidence-linked candidate rules."""

EVIDENCE_RULES = [
    {
        "condition": {
            "field": "weight_management_appropriate",
            "operator": "eq",
            "value": 1,
        },
        "candidate": {
            "id": "weight_management",
            "title": "Consider evidence-informed weight management",
            "priority": "medium",
            "reason": (
                "Weight-loss intervention evidence supports consideration "
                "when appropriate; no individual calorie target is inferred."
            ),
            "evidence_id": "Vilar-Gomez-2015",
            "recipe_fields": ["calories", "serving_size", "nutrition"],
        },
    },
    {
        "condition": {
            "field": "diet_quality_review_requested",
            "operator": "eq",
            "value": 1,
        },
        "candidate": {
            "id": "diet_quality",
            "title": "Consider overall dietary quality",
            "priority": "medium",
            "reason": (
                "The project mapping references Mediterranean-style dietary "
                "patterns. One recipe cannot establish whole-diet outcomes."
            ),
            "evidence_id": "EASL-EASD-EASO-2024",
            "recipe_fields": [
                "saturated_fat",
                "sugar",
                "fiber",
                "protein",
                "carbohydrates",
                "ingredients",
            ],
        },
    },
]
