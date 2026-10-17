"""Layer 5 scenario engine: "if I choose recipes A instead of B" (proposal section 8).

A scenario is a set of recipes that replace the patient's usual meals. For
each of the patient's recipe criteria (recipe_criteria.py) the engine
compares the scenario's mean composition with the patient's own 24-hour
recall and reports whether it improves (lower saturated fat, sugars or
energy; more fibre), worsens, or does not change meaningfully. Each criterion carries the evidence rows for that
pathway; the scenario's evidence label is the strongest label among the
criteria that improve. The result is a direction with an evidence label,
never a predicted CAP or probability: the Layer 2 model is not queried for
scenarios, because it was not trained on interventions.

Evidence links (rows of data/evidence/intervention_evidence.csv):
- energy: calorie restriction improved steatosis (HAI2022-CR, MODERATE);
  weight loss of 5% or more is the STRONG pathway (VG2015-WL5), but lower
  energy meals only lead there if sustained, so it is listed as conditional.
- saturated fat: saturated-fat overfeeding raised liver fat more than
  unsaturated fat (LUU2018-SAT, MODERATE) and the EASL guideline advises
  limiting it.
- sugars: sugar harms the liver when it adds surplus energy (CHI2014-HYPER,
  MODERATE) but not when exchanged isocalorically (CHI2014-ISO, WEAK), so the
  label is MODERATE only when the scenario also lowers energy.
- fibre, as a marker of a Mediterranean-style diet: MedDiet reduced MRI-PDFF
  (STE2026-MEDDIET, MODERATE). One recipe is not a dietary pattern, so every
  scenario is labelled as indirect evidence.

Usual meal energy is a third of the recall's daily energy. Without a
reliable recall, criteria are compared with the reference values instead.
Steatosis grades from CAP follow Karlas et al., J Hepatol 2017 (S1 >= 248,
S2 >= 268, S3 >= 280 dB/m).
"""

from src.recipe_scoring.evidence_table import cite
from src.recipe_scoring.habitual_diet import habitual_intake
from src.recipe_scoring.recipe_criteria import CRITERIA, derived_metrics

MEALS_PER_DAY = 3
# A change smaller than this share of the criterion's tolerance counts as no change.
MIN_CHANGE = 0.1
GRADES = ((280, "S3"), (268, "S2"), (248, "S1"))
STRENGTH_ORDER = ("STRONG", "MODERATE", "GUIDELINE RECOMMENDATION", "WEAK / INCONCLUSIVE")
LABEL = "Simulated projection, not a clinical forecast"
HORIZONS = ("3 months", "6 months", "1 year")

PATHWAYS = {
    "weight_management": {"habitual": "meal_energy_kcal", "evidence_ids": ["HAI2022-CR"],
                          "conditional": {"evidence_ids": ["VG2015-WL5"],
                                          "condition": "if sustained lower energy leads to >= 5% weight loss"}},
    "limit_saturated_fat": {"habitual": "saturated_fat_pct_energy",
                            "evidence_ids": ["LUU2018-SAT", "EASL2024-DIET"]},
    "limit_excess_sugar": {"habitual": "sugar_pct_energy", "evidence_ids": ["CHI2014-HYPER", "CHI2014-ISO"],
                           "needs_energy_reduction": True},
    "diet_quality": {"habitual": "fiber_per_1000kcal", "evidence_ids": ["STE2026-MEDDIET", "HAI2022-DIET"]},
}


def steatosis_grade(cap):
    for cutoff, grade in GRADES:
        if cap >= cutoff:
            return grade
    return "S0"


def composition(recipes):
    """Mean energy and composition measures over the scenario's recipes."""
    rows = []
    for recipe in recipes:
        nutrition = dict(recipe.get("nutrition") or {})
        rows.append({"calories": nutrition.get("calories"), **derived_metrics(nutrition)})
    keys = ["calories", "saturated_fat_pct_energy", "sugar_pct_energy", "fiber_per_1000kcal"]
    mean = {}
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        mean[key] = sum(values) / len(values) if values else None
    return mean


def _strongest(labels):
    ranked = [s for s in STRENGTH_ORDER if s in labels]
    return ranked[0] if ranked else "INSUFFICIENT"


def _usual(patient):
    intake = habitual_intake(patient or {})
    if not intake["recall_reliable"]:
        return None
    return {**intake, "meal_energy_kcal": intake["energy_kcal"] / MEALS_PER_DAY}


def evaluate_scenario(recipes, priorities, patient=None, name="Scenario"):
    """Evidence-graded projected direction for one scenario.

    recipes: recipe dicts with nutrition; priorities: Layer 3 priorities;
    patient: raw NHANES row including the recall columns (optional).
    """
    if not recipes:
        raise ValueError("A scenario needs at least one recipe.")
    usual = _usual(patient)
    scenario = composition(recipes)
    lower_energy = None
    if usual and scenario["calories"] is not None:
        lower_energy = scenario["calories"] < usual["meal_energy_kcal"] * (1 - MIN_CHANGE)

    criteria, shifts = [], []
    for item in priorities:
        criterion, pathway = CRITERIA.get(item["id"]), PATHWAYS.get(item["id"])
        if criterion is None or pathway is None:
            continue
        value = scenario[criterion["nutrient"]]
        baseline = usual[pathway["habitual"]] if usual else None
        compared_with = "usual diet" if baseline is not None else "reference value"
        if baseline is None:
            baseline = criterion["target"]
        if value is None:
            criteria.append({"priority_id": item["id"], "direction": "not assessed",
                             "evidence_strength": "INSUFFICIENT", "reason": "recipe value missing"})
            continue
        better = baseline - value if criterion["direction"] == "lower" else value - baseline
        change = better / criterion["tolerance"]
        shifts.append(abs(value - baseline) / criterion["tolerance"])
        direction = ("improves" if change >= MIN_CHANGE
                     else "worsens" if change <= -MIN_CHANGE else "no meaningful change")
        evidence = cite(pathway["evidence_ids"])
        strengths = [e["evidence_strength"] for e in evidence]
        if pathway.get("needs_energy_reduction") and not lower_energy:
            strengths = [s for s in strengths if s != "MODERATE"]
        result = {"priority_id": item["id"], "priority": item["priority"], "label": criterion["label"],
                  "scenario_value": round(value, 1), "baseline_value": round(baseline, 1),
                  "compared_with": compared_with, "direction": direction,
                  "evidence_strength": _strongest(strengths) if direction == "improves" else None,
                  "evidence": evidence}
        if "conditional" in pathway and direction == "improves":
            result["conditional"] = {"condition": pathway["conditional"]["condition"],
                                     "evidence": cite(pathway["conditional"]["evidence_ids"])}
        criteria.append(result)

    improving = [c for c in criteria if c["direction"] == "improves"]
    worsening = [c for c in criteria if c["direction"] == "worsens"]
    if not criteria or all(c["direction"] == "not assessed" for c in criteria):
        overall, strength = "insufficient evidence in current table to simulate this scenario", "INSUFFICIENT"
    elif improving and not worsening:
        overall, strength = "improvement", _strongest([c["evidence_strength"] for c in improving])
    elif improving and worsening:
        overall, strength = "mixed", _strongest([c["evidence_strength"] for c in improving])
    elif worsening:
        overall, strength = "worsening", None
    else:
        overall, strength = "no clear change", None
    shift = sum(shifts) / len(shifts) if shifts else None
    return {
        "name": name,
        "recipe_ids": [r.get("recipe_id") for r in recipes],
        "titles": [r.get("title") for r in recipes],
        "projected_direction": overall,
        "evidence_strength": strength,
        "shift_from_usual": None if shift is None or not usual else
        ("small" if shift < 0.5 else "moderate" if shift < 1 else "large"),
        "shift_score": None if shift is None else round(shift, 2),
        "criteria": criteria,
        "indirect_evidence": True,
        "label": LABEL,
    }


def current_state(layer2):
    cap = layer2["predicted_cap"]
    return {"predicted_cap": round(cap, 1), "predicted_grade": steatosis_grade(cap),
            "p_cap248": round(layer2["probabilities"]["cap248"], 3),
            "p_cap280": round(layer2["probabilities"]["cap280"], 3),
            "label": "Calibrated Layer 2 estimate from current measurements"}


def timeline(state, scenario):
    """Direction at each horizon for the Layer 6 slider; no grade or CAP is predicted."""
    direction = {"improvement": "towards a lower grade", "mixed": "uncertain",
                 "worsening": "towards a higher grade"}.get(scenario["projected_direction"], "unchanged")
    return [{"horizon": "today", "grade": state["predicted_grade"], "label": state["label"]}] + [
        {"horizon": h, "direction": direction, "evidence_strength": scenario["evidence_strength"],
         "assumes": "the scenario's recipes replace usual meals throughout", "label": LABEL}
        for h in HORIZONS]


def compare_scenarios(ranked, recipes, priorities, patient=None, size=3):
    """Scenario A (most health-aligned) and B (closest to the usual diet).

    ranked: Layer 4 output; both scenarios draw from recipes that meet every
    reference value, A by margin past the references, B by smallest shift
    from the patient's usual diet.
    """
    by_id = {str(r["recipe_id"]): r for r in recipes}
    passing = [item for item in ranked if item.get("meets_all_references")]
    if len(passing) < size:
        passing = list(ranked)
    pool = [by_id[str(item["recipe_id"])] for item in passing[:200]]
    a = sorted(passing[:200], key=lambda i: (-(i.get("health_margin") or 0), -i["overall"]))[:size]
    scored = [(evaluate_scenario([r], priorities, patient)["shift_score"] or 0, n, r) for n, r in enumerate(pool)]
    b = [r for _, _, r in sorted(scored)[:size]]
    scenario_a = evaluate_scenario([by_id[str(i["recipe_id"])] for i in a], priorities, patient,
                                   "Scenario A - max health")
    scenario_b = evaluate_scenario(b, priorities, patient, "Scenario B - closer to usual recipes")
    return [scenario_a, scenario_b]
