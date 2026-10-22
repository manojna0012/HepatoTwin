"""Habitual diet from the NHANES day-1 24-hour recall (proposal section 17.1).

Turns one participant's recall into the same composition measures used for
recipes (recipe_criteria.py), so a Layer 3 priority can be raised when the
patient's own intake misses the reference value and lowered when it is met.

Sources, all raw NHANES coding:
- DR1TOT nutrient totals (both FibroScan cycles): energy, saturated,
  monounsaturated and polyunsaturated fat, total sugars, fibre, sodium.
- USDA Food Patterns Equivalents (FPED) day-1 totals: added sugars and
  food-group amounts. USDA's totals for 2017-March 2020; for 2021-2023, which
  USDA has not released, totals derived from the individual foods
  (build_habitual_diet.py).

A recall is used only when it is reliable (DR1DRSTZ == 1), the day was not
much more or much less than usual (DR1_300 == 2), and energy is plausible
(500-3,500 kcal for women, 800-4,000 for men; Willett, Nutritional
Epidemiology, 3rd ed., 2013). One day is noisy for an individual, so values
adjust priorities by one level at most and are never used as a diagnosis.

HEI-2020 follows the simple scoring algorithm of the National Cancer
Institute (Shams-White et al., 2023): 13 components scored per 1,000 kcal
or as a percentage of energy, total 0-100. Legumes count towards both
vegetables and protein foods, as in the NCI code.
"""

MALE = 1
ENERGY_RANGE = {MALE: (800, 4000), 2: (500, 3500)}
KCAL_PER_TSP_SUGAR = 16      # FPED added sugars are in tsp equivalents; 16 kcal each (NCI HEI code)

# (name, max points, value giving 0 points, value giving max points)
HEI_ADEQUACY = [
    ("total_fruits", 5, 0, 0.8),
    ("whole_fruits", 5, 0, 0.4),
    ("total_vegetables", 5, 0, 1.1),
    ("greens_and_beans", 5, 0, 0.2),
    ("whole_grains", 10, 0, 1.5),
    ("dairy", 10, 0, 1.3),
    ("total_protein_foods", 5, 0, 2.5),
    ("seafood_plant_proteins", 5, 0, 0.8),
    ("fatty_acids", 10, 1.2, 2.5),
]
HEI_MODERATION = [
    ("refined_grains", 10, 4.3, 1.8),
    ("sodium", 10, 2.0, 1.1),
    ("added_sugars", 10, 26, 6.5),
    ("saturated_fats", 10, 16, 8),
]


def _number(row, name):
    try:
        value = float(row.get(name))
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _points(value, maximum, zero_at, full_at):
    fraction = (value - zero_at) / (full_at - zero_at)
    return maximum * min(max(fraction, 0.0), 1.0)


def recall_reliable(row):
    """True when the day-1 recall is complete, typical and plausible."""
    energy = _number(row, "DR1TKCAL")
    limits = ENERGY_RANGE.get(_number(row, "RIAGENDR"))
    return (_number(row, "DR1DRSTZ") == 1 and _number(row, "DR1_300") == 2
            and energy is not None and limits is not None and limits[0] <= energy <= limits[1])


def hei2020(row):
    """HEI-2020 total and components from DR1TOT and FPED totals; None without FPED."""
    energy = _number(row, "DR1TKCAL")
    names = ["DR1T_F_TOTAL", "DR1T_F_CITMLB", "DR1T_F_OTHER", "DR1T_V_TOTAL", "DR1T_V_LEGUMES",
             "DR1T_V_DRKGR", "DR1T_G_WHOLE", "DR1T_G_REFINED", "DR1T_D_TOTAL", "DR1T_PF_TOTAL",
             "DR1T_PF_LEGUMES", "DR1T_PF_SEAFD_HI", "DR1T_PF_SEAFD_LOW", "DR1T_PF_SOY",
             "DR1T_PF_NUTSDS", "DR1T_ADD_SUGARS", "DR1TSFAT", "DR1TMFAT", "DR1TPFAT", "DR1TSODI"]
    v = {name: _number(row, name) for name in names}
    if not energy or any(value is None for value in v.values()):
        return None
    per_1000 = 1000 / energy
    amounts = {
        "total_fruits": v["DR1T_F_TOTAL"] * per_1000,
        "whole_fruits": (v["DR1T_F_CITMLB"] + v["DR1T_F_OTHER"]) * per_1000,
        "total_vegetables": (v["DR1T_V_TOTAL"] + v["DR1T_V_LEGUMES"]) * per_1000,
        "greens_and_beans": (v["DR1T_V_DRKGR"] + v["DR1T_V_LEGUMES"]) * per_1000,
        "whole_grains": v["DR1T_G_WHOLE"] * per_1000,
        "dairy": v["DR1T_D_TOTAL"] * per_1000,
        "total_protein_foods": (v["DR1T_PF_TOTAL"] + v["DR1T_PF_LEGUMES"]) * per_1000,
        "seafood_plant_proteins": (v["DR1T_PF_SEAFD_HI"] + v["DR1T_PF_SEAFD_LOW"] + v["DR1T_PF_SOY"]
                                   + v["DR1T_PF_NUTSDS"] + v["DR1T_PF_LEGUMES"]) * per_1000,
        "refined_grains": v["DR1T_G_REFINED"] * per_1000,
        "sodium": v["DR1TSODI"] / 1000 * per_1000,
        "added_sugars": 100 * KCAL_PER_TSP_SUGAR * v["DR1T_ADD_SUGARS"] / energy,
        "saturated_fats": 100 * 9 * v["DR1TSFAT"] / energy,
    }
    if v["DR1TSFAT"] > 0:
        amounts["fatty_acids"] = (v["DR1TMFAT"] + v["DR1TPFAT"]) / v["DR1TSFAT"]
    else:
        amounts["fatty_acids"] = None
    components = {}
    for name, maximum, zero_at, full_at in HEI_ADEQUACY + HEI_MODERATION:
        value = amounts[name]
        # No saturated fat at all earns the full fatty-acid score (NCI convention).
        components[name] = maximum if value is None else _points(value, maximum, zero_at, full_at)
    return {"total": sum(components.values()), "components": components}


def habitual_intake(row):
    """Composition measures of the patient's recall, keyed as in recipe_criteria.

    Returns None for every measure when the recall is not reliable.
    """
    keys = ("energy_kcal", "saturated_fat_pct_energy", "sugar_pct_energy", "added_sugar_pct_energy",
            "fiber_per_1000kcal", "hei2020")
    if not recall_reliable(row):
        return {"recall_reliable": False, **{key: None for key in keys}}
    energy = _number(row, "DR1TKCAL")
    saturated, sugar, fiber = (_number(row, n) for n in ("DR1TSFAT", "DR1TSUGR", "DR1TFIBE"))
    added = _number(row, "DR1T_ADD_SUGARS")
    hei = hei2020(row)
    return {
        "recall_reliable": True,
        "energy_kcal": energy,
        "saturated_fat_pct_energy": None if saturated is None else 900 * saturated / energy,
        "sugar_pct_energy": None if sugar is None else 400 * sugar / energy,
        "added_sugar_pct_energy": None if added is None else 100 * KCAL_PER_TSP_SUGAR * added / energy,
        "fiber_per_1000kcal": None if fiber is None else 1000 * fiber / energy,
        "hei2020": None if hei is None else hei["total"],
    }
