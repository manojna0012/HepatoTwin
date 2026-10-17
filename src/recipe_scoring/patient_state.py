"""Derive the Layer 3 patient state from Layer 2 outputs and measurements.

The evidence rules fire only on explicit 0/1 flags. This module sets those
flags from the calibrated steatosis probabilities and from routine
measurements, using published thresholds:

- BMI: overweight >= 25 and obesity >= 30 kg/m2 (WHO); >= 23 and >= 27.5 for
  non-Hispanic Asian participants (WHO Expert Consultation, Lancet 2004).
- Waist: > 102 cm in men, > 88 cm in women (NCEP ATP III).
- HbA1c: >= 5.7% (ADA prediabetes threshold).
- Triglycerides: fasting >= 150 mg/dL (NCEP ATP III) or non-fasting
  >= 175 mg/dL (EAS/EFLM 2016); HDL < 40 mg/dL in men, < 50 in women.
- Steatosis: calibrated probability at or above the classifier's
  F1-selected threshold, saved with each Layer 2 classifier.

Flags describe which evidence rules apply; they are not diagnoses.
"""

ASIAN = 6        # RIDRETH3: non-Hispanic Asian
MALE = 1         # RIAGENDR

FLAG_FIELDS = (
    "steatosis_likely",
    "steatosis_moderate_severe_likely",
    "overweight",
    "obesity",
    "central_obesity",
    "elevated_glycaemia",
    "dyslipidaemia",
    "weight_management_appropriate",
    "diet_quality_review_requested",
)


def _number(measurements, name):
    try:
        value = float(measurements.get(name))
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # drop NaN


def _flag(value):
    return None if value is None else int(bool(value))


def derive_patient_state(measurements, probabilities, thresholds):
    """Return 0/1 flags (None when unknown) plus the values they came from.

    measurements: raw NHANES-coded values (RIAGENDR, RIDRETH3, BMXBMI,
        BMXWAIST, LBXGH, LBXTR or LBXSTR, LBDHDD).
    probabilities / thresholds: calibrated P(CAP >= 248) and P(CAP >= 280)
        keyed "cap248" and "cap280", and their decision thresholds.
    """
    if not isinstance(measurements, dict) or not isinstance(probabilities, dict):
        raise ValueError("Measurements and probabilities must be dictionaries.")

    sex = _number(measurements, "RIAGENDR")
    asian = _number(measurements, "RIDRETH3") == ASIAN
    bmi = _number(measurements, "BMXBMI")
    waist = _number(measurements, "BMXWAIST")
    hba1c = _number(measurements, "LBXGH")
    hdl = _number(measurements, "LBDHDD")
    fasting_tg = _number(measurements, "LBXTR")
    any_tg = _number(measurements, "LBXSTR")

    def likely(name):
        p = probabilities.get(name)
        if p is None:
            return None
        return p >= thresholds[name]

    overweight = None if bmi is None else bmi >= (23 if asian else 25)
    obesity = None if bmi is None else bmi >= (27.5 if asian else 30)
    central = None
    if waist is not None and sex is not None:
        central = waist > (102 if sex == MALE else 88)

    high_tg = None
    if fasting_tg is not None:
        high_tg = fasting_tg >= 150
    elif any_tg is not None:
        high_tg = any_tg >= 175
    low_hdl = None
    if hdl is not None and sex is not None:
        low_hdl = hdl < (40 if sex == MALE else 50)
    dyslipidaemia = None if high_tg is None and low_hdl is None else bool(high_tg or low_hdl)

    steatosis = likely("cap248")
    excess_weight = None if overweight is None and central is None else bool(overweight or central)

    state = {
        "steatosis_likely": _flag(steatosis),
        "steatosis_moderate_severe_likely": _flag(likely("cap280")),
        "overweight": _flag(overweight),
        "obesity": _flag(obesity),
        "central_obesity": _flag(central),
        "elevated_glycaemia": _flag(None if hba1c is None else hba1c >= 5.7),
        "dyslipidaemia": _flag(dyslipidaemia),
        "weight_management_appropriate": _flag(
            None if steatosis is None or excess_weight is None else steatosis and excess_weight),
        "diet_quality_review_requested": _flag(steatosis),
    }
    state["values"] = {
        "p_cap248": probabilities.get("cap248"),
        "p_cap280": probabilities.get("cap280"),
        "BMXBMI": bmi, "BMXWAIST": waist, "LBXGH": hba1c,
        "triglycerides": fasting_tg if fasting_tg is not None else any_tg,
        "LBDHDD": hdl,
    }
    return state
