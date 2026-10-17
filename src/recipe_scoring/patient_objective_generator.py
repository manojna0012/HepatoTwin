"""Generate evidence-aware candidate nutrition priorities.

Candidates are not diagnoses or clinical prescriptions.
Numeric nutrient targets must be validated separately.
"""

from copy import deepcopy

from src.recipe_scoring.evidence_table import cite


SUPPORTED_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "present"}
PRIORITIES = ("low", "medium", "high")


def _matches_condition(patient_state, condition):
    field = condition.get("field")
    operator = condition.get("operator")

    if not field or operator not in SUPPORTED_OPERATORS:
        raise ValueError("Each condition needs a field and supported operator.")

    value = patient_state.get(field)

    if operator == "present":
        return value is not None

    if value is None or isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    threshold = condition.get("value")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Numeric conditions need a numeric value.")

    return {
        "gt": value > threshold,
        "gte": value >= threshold,
        "lt": value < threshold,
        "lte": value <= threshold,
        "eq": value == threshold,
    }[operator]


def _matches_all(patient_state, conditions):
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Conditions must be a non-empty list.")
    return all(_matches_condition(patient_state, c) for c in conditions)


def generate_patient_objectives(patient_state, evidence_rules):
    """Return documented candidate priorities; do not invent nutrient targets.

    A rule applies when its "condition" (or every entry of "conditions")
    holds. Optional "priority_levels" are checked in order; the first whose
    "when" conditions all hold replaces the candidate's base priority.
    """

    if not isinstance(patient_state, dict):
        raise ValueError("Patient state must be a dictionary.")

    candidates = []
    seen = set()

    for rule in evidence_rules:
        conditions = rule.get("conditions", [rule.get("condition")])
        candidate = rule.get("candidate")

        if not all(isinstance(c, dict) for c in conditions) or not isinstance(candidate, dict):
            raise ValueError("Each rule needs condition and candidate dictionaries.")

        if not _matches_all(patient_state, conditions):
            continue

        required = ("id", "title", "priority", "reason", "evidence_id")
        if any(not candidate.get(key) for key in required):
            raise ValueError(f"Candidate must contain: {required}")

        levels = rule.get("priority_levels", [])
        if any(level.get("priority") not in PRIORITIES for level in levels) \
                or candidate["priority"] not in PRIORITIES:
            raise ValueError("Priority must be low, medium, or high.")

        if candidate["id"] in seen:
            continue

        result = deepcopy(candidate)
        result["priority_basis"] = "base rule"
        for level in levels:
            if _matches_all(patient_state, level["when"]):
                result["priority"] = level["priority"]
                result["priority_basis"] = level["basis"]
                break
        if result.get("evidence_ids"):
            result["supporting_evidence"] = cite(result["evidence_ids"])
        result["status"] = "candidate_not_clinical_prescription"
        result["numeric_target"] = None
        result["requires_clinical_review"] = True
        candidates.append(result)
        seen.add(candidate["id"])

    return {"priorities": candidates}
