"""End-to-end HepatoTwin pipeline for one patient.

Layer 2: predicted CAP and calibrated P(CAP >= 248) and P(CAP >= 280).
Layer 3: patient-state flags, then evidence-linked priorities with levels.
Layer 4: candidate recipes ranked on health, cuisine and feasibility.
Layer 5: Scenario A (max health) and B (closer to usual recipes), each an
evidence-graded projected direction with a timeline for Layer 6.

The patient is a row of raw NHANES-coded inputs (src/combined_features.py).
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.predict_cap import MODEL_FILE, predict_patients, predict_steatosis
from src.recipe_scoring.multi_criteria import DEFAULT_WEIGHTS
from src.recipe_scoring.patient_state import derive_patient_state
from src.recipe_scoring.patient_workflow import run_patient_workflow
from src.recipe_scoring.scenario_engine import compare_scenarios, current_state, timeline

ROOT = Path(__file__).resolve().parents[1]
RECIPE_FILE = ROOT / 'data' / 'processed' / 'recipe_objects.json'
CLASSIFIERS = ('cap248', 'cap280')


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
    probabilities, thresholds = {}, {}
    for name in CLASSIFIERS:
        path = folder / f'steatosis_{name}.joblib'
        probabilities[name] = float(predict_steatosis(frame, path).iloc[0])
        thresholds[name] = float(_classifier(path).threshold)
    return {'predicted_cap': float(predict_patients(frame, model_file).iloc[0]),
            'probabilities': probabilities, 'thresholds': thresholds}


def run_twin(patient, recipes=None, model_file=MODEL_FILE, weights=DEFAULT_WEIGHTS, preferences=None):
    """Run Layers 2-4 for one patient and return every intermediate result."""
    patient = dict(patient)
    state_model = layer2(patient, model_file)
    state = derive_patient_state(patient, state_model['probabilities'], state_model['thresholds'])
    recipes = load_recipes() if recipes is None else recipes
    workflow = run_patient_workflow(state, recipes, weights=weights, preferences=preferences)
    now = current_state(state_model)
    scenarios = []
    if workflow['ranked_recipes']:
        scenarios = compare_scenarios(workflow['ranked_recipes'], recipes,
                                      workflow['nutritional_priorities'], patient)
        for scenario in scenarios:
            scenario['timeline'] = timeline(now, scenario)
    return {'layer2': state_model, 'patient_state': state, **workflow,
            'current_state': now, 'scenarios': scenarios}
