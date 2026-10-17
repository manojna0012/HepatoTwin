
"""Conservative, evidence-linked candidate rules.

Rules fire on explicit 0/1 flags. Callers can set them directly, or derive
them from Layer 2 outputs with src.recipe_scoring.patient_state. Every
evidence_ids entry is a row of data/evidence/intervention_evidence.csv.
Priority levels rank which objective matters more for this patient; they are
not dietary prescriptions and carry no numeric target.
"""


def _is(field):
    return {"field": field, "operator": "eq", "value": 1}


EVIDENCE_RULES = [
    {
        "condition": _is("weight_management_appropriate"),
        "priority_levels": [
            {"when": [_is("steatosis_moderate_severe_likely")], "priority": "high",
             "basis": "Probable moderate-to-severe steatosis (CAP >= 280 dB/m)"},
            {"when": [_is("obesity")], "priority": "high",
             "basis": "BMI in the obesity range"},
        ],
        "candidate": {
            "id": "weight_management",
            "title": "Consider evidence-informed weight management",
            "priority": "medium",
            "reason": (
                "Weight-loss intervention evidence supports consideration "
                "when appropriate; no individual calorie target is inferred."
            ),
            "evidence_id": "Vilar-Gomez-2015",
            "evidence_ids": ["VG2015-WL5", "VG2015-WL7-10", "VG2015-WL10-NASH", "HAI2022-CR"],
            "recipe_fields": ["calories", "serving_size", "nutrition"],
        },
    },
    {
        "condition": _is("diet_quality_review_requested"),
        "priority_levels": [
            {"when": [_is("steatosis_moderate_severe_likely")], "priority": "high",
             "basis": "Probable moderate-to-severe steatosis (CAP >= 280 dB/m)"},
        ],
        "candidate": {
            "id": "diet_quality",
            "title": "Consider overall dietary quality",
            "priority": "medium",
            "reason": (
                "The project mapping references Mediterranean-style dietary "
                "patterns. One recipe cannot establish whole-diet outcomes."
            ),
            "evidence_id": "EASL-EASD-EASO-2024",
            "evidence_ids": ["EASL2024-DIET", "STE2026-MEDDIET", "HAI2022-DIET",
                             "SAN2021-AST-GGT", "SAN2021-ALT", "XIO2024-MD-LFD"],
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
    {
        "condition": _is("steatosis_likely"),
        "priority_levels": [
            {"when": [_is("dyslipidaemia")], "priority": "high",
             "basis": "Raised triglycerides or low HDL"},
        ],
        "candidate": {
            "id": "limit_saturated_fat",
            "title": "Prefer recipes lower in saturated fat",
            "priority": "medium",
            "reason": (
                "The guideline advises limiting foods rich in saturated fat; "
                "in an overfeeding trial saturated fat raised liver fat more "
                "than unsaturated fat or sugars. No gram target is inferred."
            ),
            "evidence_id": "EASL-EASD-EASO-2024",
            "evidence_ids": ["EASL2024-DIET", "LUU2018-SAT"],
            "recipe_fields": ["saturated_fat", "fat"],
        },
    },
    {
        "condition": _is("steatosis_likely"),
        "priority_levels": [
            {"when": [_is("elevated_glycaemia")], "priority": "medium",
             "basis": "HbA1c at or above the prediabetes threshold"},
            {"when": [_is("weight_management_appropriate")], "priority": "medium",
             "basis": "Excess weight, where surplus energy from sugar matters most"},
        ],
        "candidate": {
            "id": "limit_excess_sugar",
            "title": "Prefer recipes lower in added sugars",
            "priority": "low",
            "reason": (
                "The guideline advises avoiding sugar-sweetened beverages and "
                "sugar-rich ultra-processed foods. Feeding trials show harm when "
                "sugar adds surplus energy, not when it replaces other "
                "carbohydrates, so the base priority is low."
            ),
            "evidence_id": "EASL-EASD-EASO-2024",
            "evidence_ids": ["EASL2024-DIET", "CHI2014-HYPER", "CHI2014-ISO"],
            "recipe_fields": ["sugar", "carbohydrates", "calories"],
        },
    },
]
