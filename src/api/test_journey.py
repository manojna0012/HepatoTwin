"""The patient's journey through the API, with the clinician, using the real sign-in."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import DATA_FILE, app, get_store
from src.api.auth import create_clinician
from src.api.care import alerts, next_step, tip
from src.api.db import Store
from src.care_plan import needs_approval, reported_adherence, suggest


@pytest.fixture()
def setup():
    store = Store(':memory:')
    create_clinician(store, 'dr.rao@example.org', 'Test clinic', 'HT-4821')
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), store
    app.dependency_overrides.clear()


@pytest.fixture(scope='module')
def baseline():
    data = pd.read_csv(DATA_FILE)
    row = data[data.RIDAGEYR.between(30, 55) & (data.LUXCAPM > 300) & data.LBXSATSI.notna()
               & data.LBXSASSI.notna() & data.LBXPLTSI.notna()].iloc[0]
    out = {k: (None if pd.isna(v) else float(v)) for k, v in row.drop(['SEQN', 'cycle']).items()}
    out['DIQ070'] = 1.0          # takes diabetes pills: a safety notice
    return out


def sign_in(client, email):
    code = client.post('/auth/code', json={'email': email}).json()['code']
    token = client.post('/auth/session', json={'email': email, 'code': code}).json()['token']
    return {'Authorization': f'Bearer {token}'}


def test_sign_in_needs_the_right_code(setup):
    client, _ = setup
    assert client.get('/me').status_code == 401
    client.post('/auth/code', json={'email': 'asha@example.org'})
    assert client.post('/auth/session', json={'email': 'asha@example.org', 'code': '000000'}).status_code == 401
    me = client.get('/me', headers=sign_in(client, 'asha@example.org')).json()
    assert me['role'] == 'patient' and me['patient_id'] is None


def test_journey(setup, baseline):
    client, store = setup
    patient = sign_in(client, 'asha@example.org')
    clinician = sign_in(client, 'dr.rao@example.org')

    # Sign-up: cuisine, favourite dishes and the clinic code
    dishes = client.get('/dishes', params={'cuisine': 'European'}, headers=patient).json()
    assert len(dishes) >= 3
    made = client.post('/patients', headers=patient, json={
        'visit_date': '2026-01-05', 'measurements': baseline, 'cuisine': 'European',
        'favourites': [d['recipe_id'] for d in dishes[:3]], 'clinic_code': 'HT-4821'}).json()
    pid = made['id']
    assert made['label'] == f'P{pid:04d}'
    assert client.post(f'/patients/{pid}/twin', headers=patient).status_code == 200

    # The twin suggests a plan; a safety notice means the clinician approves it first
    suggestion = client.get(f'/patients/{pid}/plan/suggestion', headers=patient).json()
    assert [o['pace'] for o in suggestion['options']] == ['standard', 'gentle', 'ambitious']
    assert suggestion['approval_needed_for'] == ['glucose_lowering_medicine']
    assert client.put(f'/patients/{pid}/plan', headers=patient,
                      json={'start_date': '2026-01-05', 'energy_change_kcal': -400}).status_code == 403
    plan = client.put(f'/patients/{pid}/plan', headers=patient,
                      json={'start_date': '2026-01-05', 'pace': 'standard'}).json()
    assert plan['status'] == 'awaiting_approval' and plan['next_checkin'] == '2026-02-02'
    today = client.get(f'/patients/{pid}/today', params={'on': '2026-01-06'}, headers=patient).json()
    assert today['next_step']['id'] == 'awaiting_approval' and today['meals_today'] == []

    # The clinician sees the alert and approves
    assert client.get('/alerts', headers=patient).status_code == 403
    listed = client.get('/alerts', headers=clinician).json()
    assert listed[0]['patient_id'] == pid and listed[0]['id'] == 'awaiting_approval'
    assert client.post(f'/patients/{pid}/plan/approve', headers=clinician).json()['status'] == 'active'

    # Meals for the week at the chosen style, and the shopping list
    week = client.get(f'/patients/{pid}/week', headers=patient).json()
    assert week['personalised'] and week['style_step'] == 0 and len(week['days']) == 7
    assert all(len(day) == 2 for day in week['days'])
    shopping = client.get(f'/patients/{pid}/week/shopping', headers=patient).json()['groups']
    assert sum(len(v) for v in shopping.values()) > 10
    today = client.get(f'/patients/{pid}/today', params={'on': '2026-01-06'}, headers=patient).json()
    assert today['next_step']['id'] == 'cook' and len(today['meals_today']) == 2
    assert client.put(f'/patients/{pid}/plan/style', headers=patient, json={'style_step': 3}).json()['style'] == 'healthiest'

    # Two check-ins behind plan while following it: slow response, shared with the clinician
    w = baseline['BMXWT']
    for when, kg in (('2026-02-02', w + 2.5), ('2026-03-02', w + 5.0)):
        changed = client.post(f'/patients/{pid}/visits', headers=patient, json={
            'visit_date': when, 'measurements': {'BMXWT': kg},
            'followed': {'meals': 'most', 'walking': 'as_planned'}}).json()
        assert changed['check_in']['progress'] == 'behind plan'
        assert changed['check_in']['cause'] == 'slow response'
    assert changed['tip']['id'] == 'shared_with_clinician' and changed['next_checkin'] == '2026-03-30'
    ids = [a['id'] for a in client.get('/alerts', headers=clinician).json()]
    assert 'behind_twice' in ids
    # The clinician moves the patient to another pace from today; earlier months keep the plan they had
    before = client.get(f'/patients/{pid}/timeline', params={'view': 'clinician'}, headers=clinician).json()
    client.put(f'/patients/{pid}/plan', headers=clinician, json={'start_date': '2026-03-02', 'pace': 'gentle',
                                                                 'note': 'Slow response; easier pace.'})
    after = client.get(f'/patients/{pid}/timeline', params={'view': 'clinician'}, headers=clinician).json()
    assert [c['start_date'] for c in after['loop']['plan_changes']] == ['2026-01-05', '2026-03-02']
    assert [v['progress'] for v in after['loop']['visits']] == [v['progress'] for v in before['loop']['visits']]
    assert after['projection'][0]['change_kg'] != before['projection'][0]['change_kg']
    timeline = client.get(f'/patients/{pid}/timeline', headers=patient).json()
    assert 'body_response' not in timeline['loop'] and timeline['visits'][1]['followed']['meals'] == 'most'
    assert client.get(f'/patients/{pid}/timeline', params={'view': 'clinician'}, headers=patient).status_code == 403
    assert 'body_response' in client.get(f'/patients/{pid}/timeline', params={'view': 'clinician'},
                                         headers=clinician).json()['loop']

    # A patient cannot see someone else's record
    other = sign_in(client, 'other@example.org')
    assert client.get(f'/patients/{pid}', headers=other).status_code == 404


def test_plan_rules():
    options = {o['pace']: o for o in suggest(90.0)['options']}
    assert options['standard']['diet_kcal'] == -667 and options['standard']['walking_min_per_week'] == 150
    assert options['gentle']['diet_kcal'] == -334 or options['gentle']['diet_kcal'] == -333
    assert options['ambitious']['diet_kcal'] < options['standard']['diet_kcal'] < options['gentle']['diet_kcal']
    assert options['standard']['expected_change_12m_kg'] == pytest.approx(options['standard']['if_followed_change_12m_kg'] / 2,
                                                                          abs=0.06)
    assert suggest(70.0, age=15)['options'] == []
    assert needs_approval(['alcohol', 'something_else']) == ['alcohol']
    assert reported_adherence({'meals': 'most'}, -600, -100) == pytest.approx(0.9)
    assert reported_adherence({'meals': 'few', 'walking': 'as_planned'}, -600, -200) == pytest.approx((600 * 0.15 + 200 * 0.9) / 800)
    assert reported_adherence(None, -600, -100) is None


def test_tips_and_next_step():
    def loop(*visits):
        return {'visits': [{'verdict': 'start'}] + [dict(verdict='on track', progress=v, cause=c, adherence=a)
                                                    for v, c, a in visits]}
    assert tip(loop(('behind plan', 'plan not followed', 0.15)), {'style_step': 2})['style_step'] == 1
    assert tip(loop(('behind plan', 'plan not followed', 0.15)), {'style_step': 0})['id'] == 'easier_walks'
    assert tip(loop(('on plan', None, 0.9), ('on plan', None, 0.9)), {'style_step': 1})['style_step'] == 2
    assert tip(loop(('behind plan', 'slow response', 0.9)), {})['id'] == 'keep_going'
    assert tip({'visits': [{'verdict': 'start'}]}, {}) is None
    run = {'clinical': {'fibrosis': {'category': 'unknown'}, 'guardrails': {'notices': []}}}
    plan = {'status': 'active', 'next_checkin': '2026-02-02', 'style_step': 0}
    assert next_step(plan, run, '2026-01-10')['id'] == 'add_bloods'
    assert next_step(plan, run, '2026-02-02')['id'] == 'check_in'
    assert next_step(None, run)['id'] == 'choose_plan'
    found = [a['id'] for a in alerts({}, plan, run, None, '2026-03-01')]
    assert found == ['checkin_overdue', 'scarring_check_missing']
