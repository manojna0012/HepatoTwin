"""HTTP backend wrapping the twin's Layers 2-5 (Proposal section 16).

Patients and their dated visits are kept in SQLite (src/api/db.py). Running the twin uses the
patient's measurements carried forward to the chosen visit and stores the full result, so later
visits can be compared with what the twin expected.

The patient's journey: sign in with a one-time code (src/api/auth.py) -> sign up with measurements,
cuisine and favourite dishes -> the twin runs -> the twin suggests a plan and the patient picks a pace
(a clinician approves it first when a safety notice applies; src/care_plan.py) -> a weekly meal plan
and shopping list at the chosen recipe-style step (src/recipe_scoring/style_ladder.py) -> check-ins
with weight and how much of the plan was followed -> verdict, cause and one tip (src/api/care.py).
Clinicians see their clinic's patients with alerts, set or approve plans and see the full twin.

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
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api import auth, care
from src.api.db import DEFAULT_DB, Store, current_inputs
from src.care_plan import PACES, MEAL_ANSWERS, WALK_ANSWERS, needs_approval, suggest
from src.reassessment import WeightRule, projected_inputs
from src.recipe_scoring.food_diary import diary_totals
from src.recipe_scoring.meal_plan import cuisine_profiles, similar_cuisines, weekly_plan
from src.recipe_scoring.recipe_criteria import criteria_for
from src.recipe_scoring.style_ladder import (STEP_LABELS, STEPS, favourite_choices, for_week, ladder_order,
                                             shopping_list)
from src.twin_pipeline import layer2, load_recipes, run_twin

DATA_FILE = DEFAULT_DB.parents[1] / 'processed' / 'nhanes_combined.csv'
DIET_FILE = DEFAULT_DB.parents[1] / 'processed' / 'nhanes_diet.csv'     # 24-hour recall totals (Layer 3)
ID_COLUMNS = {'SEQN', 'cycle'}   # identifiers and the survey cycle, not measurements
CAP_KEY = re.compile(r'(^|_)cap(_|$)|predicted_grade|^grade$', re.IGNORECASE)


@lru_cache(maxsize=1)
def known_measurements():
    """NHANES-coded fields the twin understands: the model's inputs and the 24-hour recall totals that
    personalise the priorities (build_habitual_diet.py), minus identifiers."""
    columns = set(pd.read_csv(DATA_FILE, nrows=0).columns)
    if DIET_FILE.exists():
        columns |= set(pd.read_csv(DIET_FILE, nrows=0).columns)
    return frozenset(columns) - ID_COLUMNS


@lru_cache(maxsize=1)
def recipes():
    return load_recipes()


@lru_cache(maxsize=1)
def recipes_by_id():
    return {str(r['recipe_id']): r for r in recipes()}


@lru_cache(maxsize=32)
def similar_to(cuisine):
    return tuple(similar_cuisines(cuisine, cuisine_profiles(recipes()))) if cuisine else ()


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


# ---------------- request bodies ----------------
class Followed(BaseModel):
    meals: Optional[Literal[tuple(MEAL_ANSWERS)]] = None
    walking: Optional[Literal[tuple(WALK_ANSWERS)]] = None


class Measurements(BaseModel):
    visit_date: date
    measurements: dict[str, Optional[float]] = Field(
        description='NHANES-coded values measured at this visit, for example {"BMXWT": 84.2, "LBXSATSI": 41}')
    followed: Optional[Followed] = Field(None, description='At a check-in: how much of the plan was followed')
    diary_text: Optional[str] = Field(None, max_length=3000,
                                      description='What was eaten yesterday, in words (converted to recall totals)')
    usual_day: bool = Field(True, description='Was yesterday a usual day of eating?')

    def all_measurements(self):
        """The measurements with a typed diary converted to NHANES recall totals (coded totals given
        directly take precedence), and the diary as understood."""
        if not self.diary_text:
            return self.measurements, None
        diary = diary_totals(self.diary_text, self.usual_day)
        return {**diary['measurements'], **self.measurements}, diary

    @field_validator('measurements')
    @classmethod
    def only_known_fields(cls, values):
        unknown = sorted(set(values) - known_measurements())
        if unknown:
            raise ValueError(f'unknown measurement fields: {", ".join(unknown[:10])}')
        return values


class NewPatient(Measurements):
    label: Optional[str] = Field(None, min_length=1, max_length=80, description='A clinic reference, never a name')
    cuisine: Optional[str] = None
    favourites: list[str] = Field(default_factory=list, description='Recipe ids of dishes the patient cooks often')
    clinic_code: Optional[str] = None


class Profile(BaseModel):
    cuisine: Optional[str] = None
    favourites: Optional[list[str]] = None


class Plan(BaseModel):
    start_date: date
    pace: Optional[Literal[PACES]] = Field(None, description="The twin's suggestion at this pace")
    energy_change_kcal: Optional[float] = Field(None, ge=-1500, le=500,
                                                description='Clinicians only: a custom daily calorie change')
    walking_min_per_week: Optional[float] = Field(None, ge=0, le=600)
    style_step: int = Field(0, ge=0, le=STEPS)
    next_checkin: Optional[date] = None
    note: Optional[str] = Field(None, max_length=500)


class Style(BaseModel):
    style_step: int = Field(ge=0, le=STEPS)


class DiaryText(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    usual_day: bool = True


class Email(BaseModel):
    email: str = Field(min_length=3, max_length=200, pattern=r'^[^@\s]+@[^@\s]+$')


class CodeLogin(Email):
    code: str = Field(min_length=6, max_length=6)


app = FastAPI(title='HepatoTwin', version='0.2.0',
              description='Digital twin of liver fat with evidence-ranked recipes (research prototype).')


@lru_cache(maxsize=1)
def default_store():
    return Store(DEFAULT_DB)


def get_store():
    return default_store()


def current_user(authorization: Optional[str] = Header(None), store: Store = Depends(get_store)):
    token = authorization.removeprefix('Bearer ').strip() if authorization else None
    return auth.user_for_token(store, token)


def require_patient(store, patient_id, user):
    patient = store.patient(patient_id)
    if patient is None or not auth.can_see(user, patient):
        raise HTTPException(404, f'No patient {patient_id}.')
    return patient


def check_view(view, user):
    if view == 'clinician':
        auth.require_clinician(user)
    return view == 'clinician'


# ---------------- sign-in ----------------
@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/auth/code')
def request_code(body: Email, store: Store = Depends(get_store)):
    code = auth.request_code(store, body.email)
    return {'sent_to': body.email, 'code': code,
            'delivery': 'Prototype: the code is shown here; a deployed service would email it.'}


@app.post('/auth/session')
def sign_in(body: CodeLogin, store: Store = Depends(get_store)):
    token, user = auth.start_session(store, body.email, body.code)
    return {'token': token, 'user': {k: user[k] for k in ('id', 'email', 'role', 'patient_id', 'clinic_id')}}


@app.post('/auth/logout')
def sign_out(authorization: Optional[str] = Header(None), store: Store = Depends(get_store)):
    if authorization:
        auth.end_session(store, authorization.removeprefix('Bearer ').strip())
    return {'signed_out': True}


@app.get('/me')
def me(user=Depends(current_user)):
    return {k: user[k] for k in ('id', 'email', 'role', 'patient_id', 'clinic_id')}


@app.post('/diary/preview')
def diary_preview(body: DiaryText, user=Depends(current_user)):
    """How a typed diary is understood, item by item, before it is saved."""
    return diary_totals(body.text, body.usual_day)


@app.get('/dishes')
def dishes(cuisine: str, user=Depends(current_user)):
    """Popular main meals of a cuisine, to pick favourite dishes from at sign-up."""
    return favourite_choices(recipes(), cuisine)


# ---------------- patients and visits ----------------
@app.post('/patients', status_code=201)
def create_patient(body: NewPatient, store: Store = Depends(get_store), user=Depends(current_user)):
    """Sign-up (a patient account without a record) or a clinician registering a patient of their clinic."""
    if user['role'] == 'patient':
        if user['patient_id'] is not None:
            raise HTTPException(409, 'You already have a record.')
        clinic = store.clinic_by_code(body.clinic_code) if body.clinic_code else None
        if body.clinic_code and clinic is None:
            raise HTTPException(400, 'Unknown clinic code.')
        clinic_id = clinic['id'] if clinic else None
    else:
        clinic_id = user['clinic_id']
    unknown = [f for f in body.favourites if f not in recipes_by_id()]
    if unknown:
        raise HTTPException(400, f'Unknown recipes: {", ".join(unknown[:5])}')
    patient_id = store.add_patient(body.label or 'pending', body.cuisine, body.favourites, clinic_id)
    if not body.label:
        store.conn.execute('UPDATE patients SET label = ? WHERE id = ?', (f'P{patient_id:04d}', patient_id))
        store.conn.commit()
    if user['role'] == 'patient':
        store.link_user(user['id'], patient_id=patient_id, clinic_id=clinic_id)
    measurements, diary = body.all_measurements()
    visit_id = store.add_visit(patient_id, body.visit_date.isoformat(), measurements)
    return {'id': patient_id, 'visit_id': visit_id, 'label': store.patient(patient_id)['label'],
            'diary': diary and {k: diary[k] for k in ('items', 'not_understood', 'share_understood')}}


@app.get('/patients')
def list_patients(on: Optional[date] = None, store: Store = Depends(get_store), user=Depends(current_user)):
    """A clinician's patients with their alerts, most urgent first; a patient gets their own record."""
    if user['role'] == 'patient':
        return [store.patient(user['patient_id'])] if user['patient_id'] else []
    out = []
    for p in store.patients(clinic_id=user['clinic_id']):
        status = patient_status(store, p["id"], on)
        out.append({**p, **status})
    return sorted(out, key=lambda p: (min([a['severity'] for a in p['alerts']] or [9]), p['id']))


@app.get('/patients/{patient_id}')
def get_patient(patient_id: int, store: Store = Depends(get_store), user=Depends(current_user)):
    patient = require_patient(store, patient_id, user)
    return {**patient, 'visits': store.visits(patient_id), 'plan': store.plan(patient_id)}


@app.put('/patients/{patient_id}/profile')
def set_profile(patient_id: int, body: Profile, store: Store = Depends(get_store), user=Depends(current_user)):
    require_patient(store, patient_id, user)
    unknown = [f for f in body.favourites or [] if f not in recipes_by_id()]
    if unknown:
        raise HTTPException(400, f'Unknown recipes: {", ".join(unknown[:5])}')
    store.update_profile(patient_id, body.cuisine, body.favourites)
    return store.patient(patient_id)


@app.post('/patients/{patient_id}/visits', status_code=201)
def add_visit(patient_id: int, body: Measurements, store: Store = Depends(get_store), user=Depends(current_user)):
    """A visit or check-in. Returns what changed: the verdict, its cause, the tip and the new chances."""
    require_patient(store, patient_id, user)
    before = current_inputs(store.visits(patient_id))
    followed = body.followed.model_dump(exclude_none=True) if body.followed else None
    measurements, _ = body.all_measurements()
    visit_id = store.add_visit(patient_id, body.visit_date.isoformat(), measurements, followed)
    visits = store.visits(patient_id)
    plan = store.plan(patient_id)
    if plan and plan['status'] == 'active':
        store.update_plan(plan['id'], next_checkin=care.next_checkin(body.visit_date))
        plan = store.plan(patient_id)
    result = care.loop(visits, plan, plans=store.plans(patient_id))
    after = current_inputs(visits)[-1][1]
    changed = {'visit_id': visit_id, 'tip': care.tip(result, plan),
               'check_in': (result or {}).get('visits', [None])[-1] if result else None,
               'chances_now': liver_chances(after, False),
               'chances_before': liver_chances(before[-1][1], False) if before else None,
               'next_checkin': (plan or {}).get('next_checkin')}
    return plain(changed)


# ---------------- plans ----------------
def latest_weight_and_age(store, patient_id):
    history = current_inputs(store.visits(patient_id))
    inputs = history[-1][1] if history else {}
    return inputs.get('BMXWT'), inputs.get('RIDAGEYR')


@app.get('/patients/{patient_id}/plan/suggestion')
def plan_suggestion(patient_id: int, store: Store = Depends(get_store), user=Depends(current_user)):
    require_patient(store, patient_id, user)
    weight, age = latest_weight_and_age(store, patient_id)
    if weight is None:
        raise HTTPException(400, 'A weight is needed to suggest a plan.')
    run = store.latest_run(patient_id)
    return {**suggest(weight, age), 'approval_needed_for': needs_approval(care.notices(run and run['result'])),
            'style_steps': list(STEP_LABELS)}


@app.put('/patients/{patient_id}/plan')
def set_plan(patient_id: int, body: Plan, store: Store = Depends(get_store), user=Depends(current_user)):
    """Patients choose a pace from the suggestion; clinicians may also set custom numbers. A patient's plan
    waits for a clinician when a safety notice applies; a clinician's plan is approved by them."""
    require_patient(store, patient_id, user)
    weight, age = latest_weight_and_age(store, patient_id)
    clinician = user['role'] == 'clinician'
    if body.energy_change_kcal is not None and not clinician:
        raise HTTPException(403, 'Patients choose a pace; only a clinician can set custom numbers.')
    if body.pace is not None:
        options = {o['pace']: o for o in suggest(weight, age)['options']} if weight else {}
        if body.pace not in options:
            raise HTTPException(400, 'No suggestion for this patient (weight missing or under 18).')
        diet, walk = options[body.pace]['diet_kcal'], options[body.pace]['walking_min_per_week']
    elif body.energy_change_kcal is not None:
        diet, walk = body.energy_change_kcal, body.walking_min_per_week or 0
    else:
        raise HTTPException(400, 'Give a pace or, for clinicians, the energy change.')
    if body.walking_min_per_week is not None:
        walk = body.walking_min_per_week
    run = store.latest_run(patient_id)
    blocking = needs_approval(care.notices(run and run['result']))
    status = 'active' if clinician or not blocking else 'awaiting_approval'
    store.set_plan(patient_id, body.start_date.isoformat(), diet, pace=body.pace, walking_min_per_week=walk,
                   style_step=body.style_step,
                   next_checkin=(body.next_checkin or date.fromisoformat(care.next_checkin(body.start_date))).isoformat(),
                   status=status, set_by=user['role'], approved_by=user['id'] if clinician else None, note=body.note)
    return {**store.plan(patient_id), 'approval_needed_for': blocking if status != 'active' else []}


@app.post('/patients/{patient_id}/plan/approve')
def approve_plan(patient_id: int, store: Store = Depends(get_store), user=Depends(current_user)):
    auth.require_clinician(user)
    require_patient(store, patient_id, user)
    plan = store.plan(patient_id)
    if plan is None:
        raise HTTPException(404, 'No plan to approve.')
    store.update_plan(plan['id'], status='active', approved_by=user['id'])
    return store.plan(patient_id)


@app.put('/patients/{patient_id}/plan/style')
def set_style(patient_id: int, body: Style, store: Store = Depends(get_store), user=Depends(current_user)):
    """Move along the recipe-style ladder (0 closest to usual food ... 3 healthiest)."""
    require_patient(store, patient_id, user)
    plan = store.plan(patient_id)
    if plan is None:
        raise HTTPException(404, 'No plan yet.')
    store.update_plan(plan['id'], style_step=body.style_step)
    return {**store.plan(patient_id), 'style': STEP_LABELS[body.style_step]}


# ---------------- twin, timeline, week, today ----------------
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
                 store: Store = Depends(get_store), user=Depends(current_user)):
    """Every visit with what was measured and the chances of liver fat at the time, the re-assessment
    loop on weight (expected, observed, verdict and its cause, what was learned) and projections at 3, 6
    and 12 months, both for weight and for the chances of liver fat."""
    require_patient(store, patient_id, user)
    clinician = check_view(view, user)
    raw = store.visits(patient_id)
    history = current_inputs(raw)
    if not history:
        return {'plan': None, 'visits': [], 'loop': None, 'projection': []}
    shown = ('BMXWT', 'BMXWAIST', 'LBXSATSI', 'LBXGH', 'DR1TKCAL') + (('LUXCAPM',) if clinician else ())
    visits = [{'visit_id': v['id'], 'visit_date': v['visit_date'], 'followed': v.get('followed'),
               'measured': {k: v['measurements'][k] for k in shown if v['measurements'].get(k) is not None},
               **liver_chances(inputs, clinician)} for v, inputs, _ in history]
    plan = store.plan(patient_id)
    loop = care.loop(raw, plan, plans=store.plans(patient_id))
    if loop is None:
        return plain({'plan': plan, 'visits': visits, 'loop': None, 'projection': []})
    rule = WeightRule.from_tables()
    latest = history[-1][1]
    projection = [{**p, **liver_chances(projected_inputs(latest, p['change_kg'], rule), clinician)}
                  for p in loop.pop('projection')]
    if not clinician:
        loop.pop('body_response', None)
    return plain({'plan': plan, 'visits': visits, 'loop': loop, 'projection': projection,
                  'tip': care.tip(loop, plan)})


def twin_inputs(store, patient_id, visit_id=None):
    history = current_inputs(store.visits(patient_id))
    if visit_id is not None:
        history = [h for h in history if h[0]['id'] == visit_id]
        if not history:
            raise HTTPException(404, f'Patient {patient_id} has no visit {visit_id}.')
    if not history:
        raise HTTPException(400, 'No visits yet.')
    return history[-1]


@app.post('/patients/{patient_id}/twin')
def run_for_visit(patient_id: int, view: Literal['patient', 'clinician'] = 'patient',
                  visit_id: Optional[int] = None, store: Store = Depends(get_store), user=Depends(current_user)):
    """Run the twin on the inputs carried forward to a visit (the latest visit by default) and store it."""
    patient = require_patient(store, patient_id, user)
    clinician = check_view(view, user)
    visit, inputs, measured = twin_inputs(store, patient_id, visit_id)
    favourites = [recipes_by_id()[f] for f in patient['favourites'] if f in recipes_by_id()]
    preferences = {'requested_cuisine': patient['cuisine']} if patient['cuisine'] else None
    twin = plain(run_twin(full_inputs(inputs), recipes=recipes(), preferences=preferences, favourites=favourites))
    run_id = store.add_run(patient_id, visit['id'], twin)
    body = twin if clinician else patient_view(twin)
    return {'run_id': run_id, 'visit_id': visit['id'], 'visit_date': visit['visit_date'], 'view': view,
            'inputs_last_measured': measured if clinician else None, 'twin': body}


def week_for(store, patient_id):
    """The seven-day plan at the plan's recipe-style step, from the latest twin run."""
    patient, plan, run = store.patient(patient_id), store.plan(patient_id), store.latest_run(patient_id)
    if run is None:
        raise HTTPException(400, 'Run the twin first.')
    twin = run['result']
    if twin.get('recipes_withheld'):
        return {'withheld': True, 'message': 'Recipes are paused until your clinician has assessed your liver.',
                'days': [], 'checks': {}, 'style': None}
    step = (plan or {}).get('style_step') or 0
    inputs = twin_inputs(store, patient_id)[1]
    favourites = [recipes_by_id()[f] for f in patient['favourites'] if f in recipes_by_id()]
    order, personal = ladder_order(twin['ranked_recipes'], recipes(), twin['nutritional_priorities'],
                                   full_inputs(inputs), favourites, step)
    objectives, _ = criteria_for(twin['nutritional_priorities'])
    cuisines = [patient['cuisine']] if patient['cuisine'] else None
    chosen = {str(i['recipe_id']) for i in order}
    rest = [i for i in twin['ranked_recipes'] if str(i['recipe_id']) not in chosen]   # to complete the week
    days, checks, notes = weekly_plan(for_week(order + rest), list(recipes()), objectives, cuisines,
                                      list(similar_to(patient['cuisine'])), pool=len(order) + len(rest))
    return {'withheld': False, 'style_step': step, 'style': STEP_LABELS[step] if personal else STEP_LABELS[STEPS],
            'personalised': personal, 'start_date': (plan or {}).get('start_date'),
            'days': [[{'recipe_id': m['recipe_id'], 'title': m['title'], 'cuisine': m.get('cuisine'),
                       'total_time': m.get('total_time'), 'ingredients': len(m.get('ingredients') or [])}
                      for m in day] for day in days],
            'checks': checks, 'notes': notes, '_meals': [m for day in days for m in day]}


@app.get('/patients/{patient_id}/week')
def get_week(patient_id: int, store: Store = Depends(get_store), user=Depends(current_user)):
    require_patient(store, patient_id, user)
    week = week_for(store, patient_id)
    week.pop('_meals', None)
    return plain(week)


@app.get('/patients/{patient_id}/week/shopping')
def get_shopping(patient_id: int, store: Store = Depends(get_store), user=Depends(current_user)):
    require_patient(store, patient_id, user)
    week = week_for(store, patient_id)
    return plain({'servings': 'one serving of each meal', 'groups': shopping_list(week.get('_meals', []))})


@app.get('/patients/{patient_id}/today')
def get_today(patient_id: int, on: Optional[date] = None, store: Store = Depends(get_store),
              user=Depends(current_user)):
    """The Today screen: one status line, the next step, today's two meals and the next check-in."""
    require_patient(store, patient_id, user)
    today = on or date.today()
    plan, run = store.plan(patient_id), store.latest_run(patient_id)
    twin = run['result'] if run else None
    loop = care.loop(store.visits(patient_id), plan, plans=store.plans(patient_id))
    last = [v for v in (loop or {}).get('visits', []) if v['verdict'] != 'start']
    status = None
    if twin:
        p = round(100 * twin['current_state']['p_cap280'])
        status = f'{"Fat in your liver is likely" if p >= 50 else "Some liver fat is possible"} ({p}% chance of a lot of fat).'
        if last:
            status += f' Your progress: {last[-1]["verdict"]}.'
    meals = []
    if plan and plan['status'] == 'active' and twin and not twin.get('recipes_withheld'):
        days = week_for(store, patient_id)['days']
        if days:
            meals = days[(today - date.fromisoformat(plan['start_date'])).days % len(days)]
    return plain({'status': status, 'next_step': care.next_step(plan, twin, today), 'meals_today': meals,
                  'next_checkin': (plan or {}).get('next_checkin'), 'tip': care.tip(loop, plan),
                  'plan_status': (plan or {}).get('status')})


# ---------------- clinician ----------------
def patient_status(store, patient_id, today=None):
    plan, run = store.plan(patient_id), store.latest_run(patient_id)
    twin = run['result'] if run else None
    loop = care.loop(store.visits(patient_id), plan, plans=store.plans(patient_id))
    last = [v for v in (loop or {}).get('visits', []) if v['verdict'] != 'start']
    return {'alerts': care.alerts(store.patient(patient_id), plan, twin, loop, today),
            'chance_a_lot_of_fat_pct': round(100 * twin['current_state']['p_cap280']) if twin else None,
            'scarring': care.scarring(twin), 'last_verdict': last[-1]['verdict'] if last else None,
            'plan_status': (plan or {}).get('status')}


@app.get('/alerts')
def clinic_alerts(on: Optional[date] = None, store: Store = Depends(get_store), user=Depends(current_user)):
    """Every alert for the clinician's patients, most urgent first."""
    auth.require_clinician(user)
    out = [{'patient_id': p['id'], 'label': p['label'], **a}
           for p in store.patients(clinic_id=user['clinic_id']) for a in patient_status(store, p["id"], on)["alerts"]]
    return sorted(out, key=lambda a: (a['severity'], a['patient_id']))
