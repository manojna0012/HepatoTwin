"""End-to-end HepatoTwin pipeline for one patient.

Layer 2: predicted CAP and calibrated P(CAP >= 248) and P(CAP >= 280).
Layer 3: patient-state flags, then evidence-linked priorities with levels.
Layer 4: candidate recipes ranked on health, cuisine and feasibility.
Layer 5: Scenario A (max health) and B (closer to usual recipes), each an
evidence-graded projected direction with a timeline for Layer 6.

Alongside (Proposal sections 7.3, 17.2-17.4, 17.7): fibrosis risk (FIB-4) and its
referral, the MASLD/MetALD research label and safety notices, a per-patient
explanation of the predicted CAP, a 90% conformal CAP interval and probability
bands, knowledge-graph-traced explanations of the top recipes, and a grounded
plain-language summary. A high fibrosis risk withholds recipes until the patient
has had a clinical assessment.

The patient is a row of raw NHANES-coded inputs (src/combined_features.py).
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src import uncertainty
from src.explain_patient import explain_cap
from src.fibrosis import assess
from src.liver_labels import guardrails, liver_label
from src.predict_cap import MODEL_FILE, predict_patients, predict_steatosis
from src.recipe_scoring.recipe_explanation import explain_recipe
from src.twin_summary import summarise
from src.recipe_scoring.multi_criteria import DEFAULT_WEIGHTS
from src.recipe_scoring.patient_state import derive_patient_state
from src.recipe_scoring.patient_workflow import run_patient_workflow
from src.recipe_scoring.scenario_engine import compare_scenarios, current_state, timeline

ROOT = Path(__file__).resolve().parents[1]
RECIPE_FILE = ROOT / 'data' / 'processed' / 'recipe_objects.json'
CLASSIFIERS = ('cap248', 'cap280')
EXPLAINED_RECIPES = 3


@lru_cache(maxsize=None)
def load_recipes(path=RECIPE_FILE):
    with open(path, encoding='utf-8') as f:
        return tuple(json.load(f))


@lru_cache(maxsize=None)
def _classifier(path):
    return joblib.load(path)


def layer2(patient, model_file=MODEL_FILE):
    """Predicted CAP, calibrated probabilities and their decision thresholds."""
    frame = pd.DataFrame([patient])
    folder = Path(model_file).parent
    probabilities, thresholds, raw = {}, {}, {}
    for name in CLASSIFIERS:
        path = folder / f'steatosis_{name}.joblib'
        probabilities[name] = float(predict_steatosis(frame, path).iloc[0])
        thresholds[name] = float(_classifier(path).threshold)
        classifier = _classifier(path)
        raw[name] = float(classifier.pipeline.predict_proba(frame[list(classifier.feature_names_in_)])[0, 1])
    return {'predicted_cap': float(predict_patients(frame, model_file).iloc[0]),
            'probabilities': probabilities, 'thresholds': thresholds, 'raw_scores': raw}


def layer2_uncertainty(state_model):
    """90% conformal interval for CAP and bootstrap bands for the calibrated probabilities."""
    return {'cap_interval_90': uncertainty.cap_interval(state_model['predicted_cap']),
            'probability_bands': {name: uncertainty.probability_band(name, score)
                                  for name, score in state_model['raw_scores'].items()}}


def run_twin(patient, recipes=None, model_file=MODEL_FILE, weights=DEFAULT_WEIGHTS, preferences=None,
             summary_generator=None):
    """Run Layers 2-5 for one patient and return every intermediate result."""
    patient = dict(patient)
    state_model = layer2(patient, model_file)
    state = derive_patient_state(patient, state_model['probabilities'], state_model['thresholds'])
    fibrosis = assess(patient)
    steatosis = state.get('steatosis_likely')
    clinical = {'fibrosis': fibrosis,
                'liver_label': liver_label(patient, None if steatosis is None else bool(steatosis)),
                'guardrails': guardrails(patient, fibrosis)}
    recipes = load_recipes() if recipes is None else recipes
    workflow = run_patient_workflow(state, recipes, weights=weights, preferences=preferences)
    now = current_state(state_model)
    scenarios = []
    if workflow['ranked_recipes']:
        scenarios = compare_scenarios(workflow['ranked_recipes'], recipes,
                                      workflow['nutritional_priorities'], patient)
        for scenario in scenarios:
            scenario['timeline'] = timeline(now, scenario)
    by_id = {str(r.get('recipe_id')): r for r in recipes}
    explanations = [explain_recipe(item, by_id.get(str(item['recipe_id'])), item.get('cuisine_detail'))
                    for item in workflow['ranked_recipes'][:EXPLAINED_RECIPES]]
    twin = {'layer2': state_model, 'patient_state': state, **workflow,
            'current_state': now, 'scenarios': scenarios, 'clinical': clinical,
            'recipes_withheld': clinical['guardrails']['block_recipes'],
            'cap_explanation': explain_cap(patient, model_file),
            'uncertainty': layer2_uncertainty(state_model), 'recipe_explanations': explanations}
    twin['summary'] = summarise(twin, summary_generator)
    return twin
