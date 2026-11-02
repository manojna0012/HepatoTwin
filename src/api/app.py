"""HTTP backend wrapping the twin's Layers 2-5 (Proposal section 16).

Patients and their dated visits are kept in SQLite (src/api/db.py). Running the twin uses the
patient's measurements carried forward to the chosen visit and stores the full result, so later
visits can be compared with what the twin expected.

Two views of one twin run:

- patient: the plain-language summary, the chances of liver fat, the safety check, priorities,
  recipes with their reasons and the two scenarios. No CAP estimate, CAP range or CAP grade: at
  R^2 0.56 they would suggest more precision than the model has.
- clinician: the twin's full output, including the CAP estimate, its 90% range and the SHAP drivers.

    .venv/Scripts/python.exe -m uvicorn src.api.app:app --reload
"""
import math
import re
from datetime import date
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api.db import DEFAULT_DB, Store, current_inputs
from src.reassessment import WeightRule, projected_inputs, reassess
from src.twin_pipeline import layer2, load_recipes, run_twin

DATA_FILE = DEFAULT_DB.parents[1] / 'processed' / 'nhanes_combined.csv'
ID_COLUMNS = {'SEQN', 'cycle'}   # identifiers and the survey cycle, not measurements
CAP_KEY = re.compile(r'(^|_)cap(_|$)|predicted_grade|^grade$', re.IGNORECASE)


@lru_cache(maxsize=1)
def known_measurements():
    """NHANES-coded fields the twin understands (the processed dataset's columns, minus identifiers)."""
    return frozenset(pd.read_csv(DATA_FILE, nrows=0).columns) - ID_COLUMNS


@lru_cache(maxsize=1)
def recipes():
    return load_recipes()


def plain(value):
    """JSON-safe copy: numpy numbers become Python numbers, NaN and infinity become null."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def without_cap(value):
    """Drop every CAP estimate, range and grade, keeping the probabilities p_cap248 and p_cap280."""
    if isinstance(value, dict):
        return {k: without_cap(v) for k, v in value.items()
                if not (CAP_KEY.search(k) and not k.startswith('p_cap'))}
    if isinstance(value, list):
        return [without_cap(v) for v in value]
    return value


def patient_view(twin):
    """What a patient sees of one twin run."""
    state, clinical = twin['current_state'], twin['clinical']
    view = {
        'summary': twin['summary']['text'],
        'summary_checked': twin['summary']['check']['faithful'],
        'chance_some_fat_pct': round(100 * state['p_cap248']),
        'chance_a_lot_of_fat_pct': round(100 * state['p_cap280']),
        'liver_label': (clinical.get('liver_label') or {}).get('label'),
        'fibrosis': {k: clinical['fibrosis'].get(k) for k in ('method', 'fib4', 'liver_stiffness_kpa', 'category',
                                                              'message', 'note')},
        'notices': [n['message'] for n in clinical['guardrails']['notices']],
        'recipes_withheld': twin['recipes_withheld'],
        'priorities': [{'title': p['title'], 'priority': p['priority'],
                        'evidence': [e['evidence_id'] for e in p.get('supporting_evidence', [])]}
                       for p in twin.get('nutritional_priorities', [])],
        'recipes': [] if twin['recipes_withheld'] else twin['recipe_explanations'],
        'scenarios': [] if twin['recipes_withheld'] else twin['scenarios'],
    }
    return without_cap(view)


class Measurements(BaseModel):
    visit_date: date
    measurements: dict[str, Optional[float]] = Field(
        description='NHANES-coded values measured at this visit, for example {"BMXWT": 84.2, "LBXSATSI": 41}')

    @field_validator('measurements')
    @classmethod
    def only_known_fields(cls, values):
        unknown = sorted(set(values) - known_measurements())
        if unknown:
            raise ValueError(f'unknown measurement fields: {", ".join(unknown[:10])}')
        return values


class Plan(BaseModel):
    start_date: date
    energy_change_kcal: float = Field(ge=-1500, le=500, description='Change in daily intake, e.g. -500')


class NewPatient(Measurements):
    label: str = Field(min_length=1, max_length=80, description='A clinic reference, never a name')


app = FastAPI(title='HepatoTwin', version='0.1.0',
              description='Digital twin of liver fat with evidence-ranked recipes (research prototype).')


@lru_cache(maxsize=1)
def default_store():
    return Store(DEFAULT_DB)


def get_store():
    return default_store()


def require_patient(store, patient_id):
    patient = store.patient(patient_id)
    if patient is None:
        raise HTTPException(404, f'No patient {patient_id}.')
    return patient


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/patients', status_code=201)
def create_patient(body: NewPatient, store: Store = Depends(get_store)):
    patient_id = store.add_patient(body.label)
    visit_id = store.add_visit(patient_id, body.visit_date.isoformat(), body.measurements)
    return {'id': patient_id, 'visit_id': visit_id}


@app.get('/patients')
def list_patients(store: Store = Depends(get_store)):
    return store.patients()


@app.get('/patients/{patient_id}')
def get_patient(patient_id: int, store: Store = Depends(get_store)):
    patient = require_patient(store, patient_id)
    return {**patient, 'visits': store.visits(patient_id)}


@app.post('/patients/{patient_id}/visits', status_code=201)
def add_visit(patient_id: int, body: Measurements, store: Store = Depends(get_store)):
    require_patient(store, patient_id)
    return {'visit_id': store.add_visit(patient_id, body.visit_date.isoformat(), body.measurements)}


@app.put('/patients/{patient_id}/plan')
def set_plan(patient_id: int, body: Plan, store: Store = Depends(get_store)):
    require_patient(store, patient_id)
    store.set_plan(patient_id, body.start_date.isoformat(), body.energy_change_kcal)
    return store.plan(patient_id)


def full_inputs(inputs):
    """Fields never measured are passed as missing, which the Layer 2 model handles."""
    return dict.fromkeys(known_measurements(), float('nan')) | inputs


def liver_chances(inputs, clinician):
    state = layer2(full_inputs(inputs))
    out = {'chance_some_fat_pct': round(100 * state['probabilities']['cap248']),
           'chance_a_lot_of_fat_pct': round(100 * state['probabilities']['cap280'])}
    if clinician:
        out['predicted_cap'] = round(state['predicted_cap'], 1)
    return out


@app.get('/patients/{patient_id}/timeline')
def get_timeline(patient_id: int, view: Literal['patient', 'clinician'] = 'patient',
                 store: Store = Depends(get_store)):
    """Every visit with what was measured and the chances of liver fat at the time, the re-assessment
    loop on weight (expected, observed, verdict, what was learned) and projections at 3, 6 and 12
    months, both for weight and for the chances of liver fat."""
    require_patient(store, patient_id)
    clinician = view == 'clinician'
    history = current_inputs(store.visits(patient_id))
    if not history:
        return {'plan': None, 'visits': [], 'loop': None, 'projection': []}
    shown = ('BMXWT', 'BMXWAIST', 'LBXSATSI', 'LBXGH', 'DR1TKCAL') + (('LUXCAPM',) if clinician else ())
    visits = [{'visit_id': v['id'], 'visit_date': v['visit_date'],
               'measured': {k: v['measurements'][k] for k in shown if v['measurements'].get(k) is not None},
               **liver_chances(inputs, clinician)} for v, inputs, _ in history]
    plan = store.plan(patient_id)
    if plan is None:
        return plain({'plan': None, 'visits': visits, 'loop': None, 'projection': []})
    rule = WeightRule.from_tables()
    weights = [(v['visit_date'], v['measurements'].get('BMXWT')) for v, _, _ in history]
    loop = reassess(weights, plan['start_date'], plan['energy_change_kcal'], rule)
    latest = history[-1][1]
    projection = [{**p, **liver_chances(projected_inputs(latest, p['change_kg'], rule), clinician)}
                  for p in loop.pop('projection')]
    return plain({'plan': plan, 'visits': visits, 'loop': loop, 'projection': projection})


@app.post('/patients/{patient_id}/twin')
def run_for_visit(patient_id: int, view: Literal['patient', 'clinician'] = 'patient',
                  visit_id: Optional[int] = None, store: Store = Depends(get_store)):
    """Run the twin on the inputs carried forward to a visit (the latest visit by default) and store it."""
    require_patient(store, patient_id)
    history = current_inputs(store.visits(patient_id))
    if visit_id is not None:
        history = [h for h in history if h[0]['id'] == visit_id]
        if not history:
            raise HTTPException(404, f'Patient {patient_id} has no visit {visit_id}.')
    visit, inputs, measured = history[-1]
    twin = plain(run_twin(full_inputs(inputs), recipes=recipes()))
    run_id = store.add_run(patient_id, visit['id'], twin)
    body = twin if view == 'clinician' else patient_view(twin)
    return {'run_id': run_id, 'visit_id': visit['id'], 'visit_date': visit['visit_date'], 'view': view,
            'inputs_last_measured': measured if view == 'clinician' else None, 'twin': body}
