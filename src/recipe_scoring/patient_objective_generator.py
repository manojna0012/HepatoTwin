"""Generate evidence-aware candidate nutrition priorities.

Candidates are not diagnoses or clinical prescriptions.
Numeric nutrient targets must be validated separately.
"""

from copy import deepcopy


SUPPORTED_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "present"}


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


def generate_patient_objectives(patient_state, evidence_rules):
    """Return documented candidate priorities; do not invent nutrient targets."""

    if not isinstance(patient_state, dict):
        raise ValueError("Patient state must be a dictionary.")

    candidates = []
    seen = set()

    for rule in evidence_rules:
        condition = rule.get("condition")
        candidate = rule.get("candidate")

        if not isinstance(condition, dict) or not isinstance(candidate, dict):
            raise ValueError("Each rule needs condition and candidate dictionaries.")

        if not _matches_condition(patient_state, condition):
            continue

        required = ("id", "title", "priority", "reason", "evidence_id")
        if any(not candidate.get(key) for key in required):
            raise ValueError(f"Candidate must contain: {required}")

        if candidate["priority"] not in {"low", "medium", "high"}:
            raise ValueError("Priority must be low, medium, or high.")

        if candidate["id"] in seen:
            continue

        result = deepcopy(candidate)
        result["status"] = "candidate_not_clinical_prescription"
        result["numeric_target"] = None
        result["requires_clinical_review"] = True
        candidates.append(result)
        seen.add(candidate["id"])

    return {"priorities": candidates}
