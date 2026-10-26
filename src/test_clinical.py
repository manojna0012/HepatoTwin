import math

import numpy as np
import pytest

from src.fibrosis import assess, category, fib4
from src.liver_labels import alcohol_g_per_week, cardiometabolic_criteria, guardrails, liver_label

ADULT = {'RIDAGEYR': 45, 'RIAGENDR': 1, 'RIDRETH3': 3, 'BMXBMI': 24.0, 'BMXWAIST': 90.0, 'LBXGLU': 90.0,
         'LBXGH': 5.2, 'DIQ010': 2, 'DIQ050': 2, 'DIQ070': 2, 'BPXOSY1': 118, 'BPXOSY2': 120, 'BPXOSY3': 119,
         'BPXODI1': 75, 'BPXODI2': 76, 'BPXODI3': 74, 'BPQ020': 2, 'LBXSTR': 100.0, 'LBDHDD': 55.0,
         'ALQ111': 1, 'ALQ121': 5, 'ALQ130': 2, 'LBXSASSI': 22.0, 'LBXSATSI': 25.0, 'LBXPLTSI': 250.0}


def test_fib4_formula_and_cut_offs():
    assert fib4(50, 40, 25, 200) == pytest.approx(50 * 40 / (200 * 5))
    assert math.isnan(fib4(50, 40, 0, 200))
    assert category(1.29, 40) == 'low' and category(1.3, 40) == 'indeterminate'
    assert category(2.67, 40) == 'indeterminate' and category(2.68, 40) == 'high'
    assert category(3.0, 15) == 'not_applicable' and category(float('nan'), 40) == 'unknown'


def test_fibrosis_assessment_blocks_recipes_only_when_high():
    high = assess(ADULT | {'LBXSASSI': 90.0, 'LBXSATSI': 25.0, 'LBXPLTSI': 120.0})
    assert high['category'] == 'high' and high['refer_before_recipes']
    low = assess(ADULT)
    assert low['category'] == 'low' and not low['refer_before_recipes']
    teen = assess(ADULT | {'RIDAGEYR': 15})
    assert teen['fib4'] is None and teen['category'] == 'not_applicable'


def test_measured_stiffness_replaces_fib4_for_adults():
    scan = assess(ADULT | {'LUXSMED': 13.5, 'LUXSIQRM': 12})
    assert scan['method'] == 'fibroscan' and scan['category'] == 'high' and scan['refer_before_recipes']
    assert assess(ADULT | {'LUXSMED': 9.0})['category'] == 'indeterminate'          # spread not given
    assert assess(ADULT | {'LUXSMED': 5.1, 'LUXSIQRM': 10})['category'] == 'low'
    unreliable = assess(ADULT | {'LUXSMED': 13.5, 'LUXSIQRM': 45})
    assert unreliable['method'] == 'fib4' and 'not reliable' in unreliable['note']
    assert assess(ADULT | {'RIDAGEYR': 15, 'LUXSMED': 13.5})['category'] == 'not_applicable'


def test_low_fib4_under_35_carries_a_note():
    assert 'less reliable under age 35' in assess(ADULT | {'RIDAGEYR': 25})['note']
    assert assess(ADULT)['note'] is None                                              # age 45


def test_alcohol_grams():
    assert alcohol_g_per_week(ADULT) == pytest.approx(1 * 2 * 14)          # once a week, 2 drinks
    assert alcohol_g_per_week(ADULT | {'ALQ111': 2}) == 0
    assert alcohol_g_per_week(ADULT | {'ALQ121': 1, 'ALQ130': 3}) == pytest.approx(7 * 3 * 14)
    assert alcohol_g_per_week(ADULT | {'ALQ130': 999}) is None


def test_masld_needs_steatosis_and_a_cardiometabolic_risk_factor():
    healthy = liver_label(ADULT, steatosis=True)
    assert healthy['label'] == 'cryptogenic_SLD' and healthy['criteria_met'] == []
    assert liver_label(ADULT | {'BMXBMI': 28.0}, True)['label'] == 'MASLD'
    assert liver_label(ADULT | {'BMXBMI': 28.0}, False)['label'] == 'no_steatosis_indicated'
    asian = ADULT | {'RIDRETH3': 6, 'BMXBMI': 24.0}
    assert cardiometabolic_criteria(asian)['body_size'] is True


def test_metald_and_ald_bands_differ_by_sex():
    weekly_280 = {'ALQ121': 1, 'ALQ130': 20 / 7 * 1.0}      # 280 g a week
    man, woman = ADULT | {'BMXBMI': 28.0} | weekly_280, ADULT | {'BMXBMI': 28.0, 'RIAGENDR': 2} | weekly_280
    assert liver_label(man, True)['label'] == 'MetALD'
    assert liver_label(woman, True)['label'] == 'MetALD'
    heavy_woman = woman | {'ALQ130': 4}                    # 7 x 4 x 14 = 392 g: above 350 for women
    assert liver_label(heavy_woman, True)['label'] == 'ALD'
    assert liver_label(heavy_woman | {'RIAGENDR': 1}, True)['label'] == 'MetALD'


def test_adolescent_body_size_uses_bmi_for_age():
    teen = ADULT | {'RIDAGEYR': 15, 'BMXWAIST': None}
    assert cardiometabolic_criteria(teen | {'BMXBMI': 24.0})['body_size'] is True    # above the 85th centile
    assert cardiometabolic_criteria(teen | {'BMXBMI': 19.0})['body_size'] is False


def test_random_glucose_counts_for_adolescents_only():
    teen = ADULT | {'RIDAGEYR': 15}
    assert cardiometabolic_criteria(teen | {'LBXSGL': 210.0})['glycaemia'] is True
    assert cardiometabolic_criteria(teen | {'LBXSGL': 150.0})['glycaemia'] is False
    assert cardiometabolic_criteria(ADULT | {'LBXSGL': 210.0})['glycaemia'] is False   # adult criteria have no random glucose


def test_guardrails():
    assert guardrails(ADULT) == {'notices': [], 'block_recipes': False}
    ids = [n['id'] for n in guardrails(ADULT | {'RIDAGEYR': 16, 'DIQ070': 1, 'RIDEXPRG': 1})['notices']]
    assert ids == ['under_18', 'glucose_lowering_medicine', 'pregnant']
    blocked = guardrails(ADULT, fibrosis={'refer_before_recipes': True, 'message': 'see a clinician'})
    assert blocked['block_recipes'] and blocked['notices'][0]['id'] == 'fibrosis_referral'
