"""Simulator inputs: literature values from the cited parameter table, plus the settings that
calibration will tune.

Literature values are read from data/evidence/simulator_parameters.csv by their param_id, so
every number in the simulation can be traced to a source row. Settings that no paper gives
(adherence patterns, person-to-person spread) live in Calibration; their defaults are starting
guesses, to be tuned against data/evidence/calibration_targets.csv (Weekly Plan, meeting 11).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
PARAMETER_FILE = ROOT / 'data' / 'evidence' / 'simulator_parameters.csv'
TARGET_FILE = ROOT / 'data' / 'evidence' / 'calibration_targets.csv'
KJ_PER_KCAL = 4.184


def load_table(path=PARAMETER_FILE):
    table = pd.read_csv(path)
    if not table.param_id.is_unique:
        raise ValueError(f'Duplicate param_id in {path}')
    return table.set_index('param_id')


def value(table, param_id):
    """Numeric value of one parameter row; raises if the row is missing or not numeric."""
    return float(table.loc[param_id, 'value'])


def weight_time_constant(half_months, ninety_five_months):
    """Time constant (months) of the single exponential closest to Hall's two rule-of-thumb points:
    half of the eventual change after half_months and 95% after ninety_five_months."""
    def loss(tau):
        return ((1 - np.exp(-half_months / tau) - 0.5) ** 2
                + (1 - np.exp(-ninety_five_months / tau) - 0.95) ** 2)
    return float(minimize_scalar(loss, bounds=(1, 60), method='bounded').x)


@dataclass(frozen=True)
class Evidence:
    """Literature values used by the simulator, each traceable to a param_id."""
    kg_per_100kj: float           # WT-EQ: eventual weight change per sustained 100 kJ/day
    weight_tau_months: float      # from WT-HALF and WT-95
    hall: dict                    # WT-RHO-F, WT-RHO-L, WT-FORBES, WT-GAMMA-F, WT-GAMMA-L, WT-DELTA, WT-FAT-REF
    circumference_cm_per_kg: dict  # WC-KG, HIP-KG, ARM-KG
    # % of the patient's own baseline per +1% weight change (weight gain positive), for patients with
    # diabetes (Look AHEAD) and without (Appuhamy 2014); blood pressure and insulin use one value for both.
    per_pct_diabetes: dict
    per_pct_no_diabetes: dict
    enzymes_per_pdff_point: dict  # LV-ALT, LV-AST, LV-GGT: U/L per point of liver-fat change
    crp_per_kg: float             # MK-CRP
    urate_per_kg: dict            # MK-UA, MK-UA-ND: mg/dL per +1 kg, with and without diabetes
    counts_pct_per_kg: dict       # CBC-WBC, CBC-LYMPCT, CBC-NEUPCT: % of baseline per +1 kg
    liver_fat_pct_per_pct: float  # LF-PER-PCT: relative liver-fat change per 1% weight change
    liver_fat_pct_per_deficit: float  # LF-DEFICIT: extra relative fall per unit energy-deficit fraction
    cap_per_pdff_point: float     # LF-CAP
    anchor_cap: float             # LF-CAP-ANCHOR
    anchor_pdff: float

    @classmethod
    def from_table(cls, table=None):
        table = load_table() if table is None else table
        anchor = str(table.loc['LF-CAP-ANCHOR', 'value'])
        cap, pdff = (float(part.split()[0].rstrip('%')) for part in anchor.split(';'))
        sbp, insulin = value(table, 'MK-SBP'), value(table, 'MK-INS')
        # Signs: weight loss lowers HbA1c, glucose, triglycerides, blood pressure and insulin, raises HDL.
        return cls(
            kg_per_100kj=value(table, 'WT-EQ'),
            weight_tau_months=weight_time_constant(value(table, 'WT-HALF'), value(table, 'WT-95')),
            hall={key: value(table, pid) for key, pid in [
                ('rho_fat', 'WT-RHO-F'), ('rho_lean', 'WT-RHO-L'), ('forbes', 'WT-FORBES'),
                ('gamma_fat', 'WT-GAMMA-F'), ('gamma_lean', 'WT-GAMMA-L'), ('activity', 'WT-DELTA'),
                ('fat_ref', 'WT-FAT-REF')]},
            circumference_cm_per_kg={'BMXWAIST': value(table, 'WC-KG'), 'BMXHIP': value(table, 'HIP-KG'),
                                     'BMXARMC': value(table, 'ARM-KG')},
            per_pct_diabetes={'LBXGH': value(table, 'MK-HBA1C'), 'glucose': value(table, 'MK-GLU'),
                              'LBDHDD': -value(table, 'MK-HDL'), 'LBXSTR': value(table, 'MK-TG'),
                              'sbp': sbp, 'LBXIN': insulin},
            per_pct_no_diabetes={'LBXGH': value(table, 'MK-HBA1C-ND'), 'glucose': value(table, 'MK-GLU-ND'),
                                 'LBDHDD': -value(table, 'MK-HDL-ND'), 'LBXSTR': value(table, 'MK-TG-ND'),
                                 'sbp': sbp, 'LBXIN': insulin},
            enzymes_per_pdff_point={'LBXSATSI': value(table, 'LV-ALT'), 'LBXSASSI': value(table, 'LV-AST'),
                                    'LBXSGTSI': value(table, 'LV-GGT')},
            crp_per_kg=value(table, 'MK-CRP'),
            urate_per_kg={True: value(table, 'MK-UA'), False: value(table, 'MK-UA-ND')},
            counts_pct_per_kg={'LBXWBCSI': value(table, 'CBC-WBC'), 'LBXLYPCT': value(table, 'CBC-LYMPCT'),
                               'LBXNEPCT': value(table, 'CBC-NEUPCT')},
            liver_fat_pct_per_pct=value(table, 'LF-PER-PCT'),
            liver_fat_pct_per_deficit=value(table, 'LF-DEFICIT'),
            cap_per_pdff_point=value(table, 'LF-CAP'),
            anchor_cap=cap, anchor_pdff=pdff)


def body_fat_kg(weight_kg, bmi, age, male):
    """Body fat mass from BMI, age and sex (Deurenberg et al. 1991; the child formula up to age 15)."""
    sex = 1.0 if male else 0.0
    if age <= 15:
        percent = 1.51 * bmi - 0.70 * age - 3.6 * sex + 1.4
    else:
        percent = 1.20 * bmi + 0.23 * age - 10.8 * sex - 5.4
    return weight_kg * float(np.clip(percent, 3, 60)) / 100


def hall_energetics(fat_kg, hall):
    """Effective energy density of weight change (kJ/kg) and energy expenditure change per kg of
    weight change (kJ/kg/day) for a person with this much body fat (Hall 2011 web appendix):
    fat makes up a share 1 / (1 + C/F) of each kg changed (Forbes), the rest is lean tissue."""
    fat_share = 1 / (1 + hall['forbes'] / max(fat_kg, 1.0))
    density = 1000 * (fat_share * hall['rho_fat'] + (1 - fat_share) * hall['rho_lean'])
    expenditure = fat_share * hall['gamma_fat'] + (1 - fat_share) * hall['gamma_lean'] + hall['activity']
    return density, expenditure


def personal_weight_response(fat_kg, ev):
    """Eventual-loss multiplier and time constant (months) for this person, relative to the rule of
    thumb, which is taken to describe a person with the reference body fat (WT-FAT-REF)."""
    density, expenditure = hall_energetics(fat_kg, ev.hall)
    ref_density, ref_expenditure = hall_energetics(ev.hall['fat_ref'], ev.hall)
    eventual = ref_expenditure / expenditure
    tau = ev.weight_tau_months * (density / expenditure) / (ref_density / ref_expenditure)
    return eventual, tau


@dataclass(frozen=True)
class Calibration:
    """Settings with no literature value. Defaults are starting guesses for calibration."""
    # Share of patients following each adherence pattern over the 12 months.
    pattern_shares: dict = field(default_factory=lambda: {
        'steady': 0.4, 'slipping': 0.3, 'regaining': 0.2, 'dropping_out': 0.1})
    adherence_mean: float = 0.7        # starting adherence, the share of the prescribed change achieved
    adherence_sd: float = 0.15
    slip_months: float = 6.0           # time constant of the decline for 'slipping'
    regain_month_range: tuple = (4, 9)   # month adherence stops for 'regaining'
    dropout_month_range: tuple = (2, 10)  # month a 'dropping_out' patient stops (and stops reporting)
    weight_response_sd: float = 0.2    # person-to-person spread of the weight response (log scale)
    marker_response_sd: float = 0.5    # spread of each marker's response around the published slope
    shared_response_corr: float = 0.5  # correlation of a person's responses across markers
    liver_response_sd: float = 0.5     # liver fat is only loosely tied to weight (LF-UNCOUPLED)
    max_liver_fat_fall: float = 0.9    # liver fat cannot fall by more than 90%
    min_pdff: float = 1.0              # % PDFF floor for the baseline liver-fat estimate
