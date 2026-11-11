import re
import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import CAP_KEY, DATA_FILE, app, current_user, get_store, plain, without_cap
from src.api.auth import create_clinician
from src.api.db import Store, current_inputs


@pytest.fixture()
def client():
    """A signed-in clinician of a test clinic."""
    store = Store(':memory:')
    clinician = store.user(create_clinician(store, 'dr@example.org', 'Test clinic', 'HT-TEST'))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[current_user] = lambda: clinician
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(scope='module')
def baseline():
    """A real NHANES participant with fatty liver, as NHANES-coded values (identifier removed)."""
    data = pd.read_csv(DATA_FILE)
    row = data[data.RIDAGEYR.between(30, 55) & (data.LUXCAPM > 300)].iloc[0]
    return {k: (None if pd.isna(v) else float(v)) for k, v in row.drop(['SEQN', 'cycle']).items()}


def keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


def test_patients_and_visits_are_stored_in_date_order(client, baseline):
    made = client.post('/patients', json={'label': 'clinic-001', 'visit_date': '2026-09-12', 'measurements': baseline})
    assert made.status_code == 201
    pid = made.json()['id']
    assert client.post(f'/patients/{pid}/visits', json={'visit_date': '2026-11-12',
                                                       'measurements': {'BMXWT': 80.0}}).status_code == 201
    assert client.post(f'/patients/{pid}/visits', json={'visit_date': '2026-10-12',
                                                       'measurements': {'BMXWT': 82.0}}).status_code == 201
    visits = client.get(f'/patients/{pid}').json()['visits']
    assert [v['visit_date'] for v in visits] == ['2026-09-12', '2026-10-12', '2026-11-12']
    listed = client.get('/patients').json()
    assert listed[0]['label'] == 'clinic-001' and listed[0]['visits'] == 3 and listed[0]['last_visit'] == '2026-11-12'


def test_bad_requests_are_refused(client, baseline):
    bad = client.post('/patients', json={'label': 'x', 'visit_date': '2026-09-12', 'measurements': {'NAME': 1}})
    assert bad.status_code == 422 and 'unknown measurement fields: NAME' in bad.text
    assert client.post('/patients', json={'label': '', 'visit_date': '2026-09-12', 'measurements': {}}).status_code == 422
    assert client.get('/patients/99').status_code == 404
    assert client.post('/patients/99/twin').status_code == 404


def test_patient_view_shows_chances_but_no_cap(client, baseline):
    pid = client.post('/patients', json={'label': 'p', 'visit_date': '2026-09-12', 'measurements': baseline}).json()['id']
    out = client.post(f'/patients/{pid}/twin').json()
    twin = out['twin']
    assert out['view'] == 'patient' and out['inputs_last_measured'] is None
    assert 0 <= twin['chance_some_fat_pct'] <= 100 and twin['summary_checked']
    # only the published cut-offs (248 and 280 dB/m) may appear, never the patient's own CAP estimate
    assert twin['summary'].startswith('Your results')
    assert set(re.findall(r'(\d+) dB/m', json.dumps(twin))) <= {'248', '280'}
    assert not [k for k in keys(twin) if CAP_KEY.search(k)]
    clinician = client.post(f'/patients/{pid}/twin', params={'view': 'clinician'}).json()['twin']
    assert clinician['current_state']['predicted_cap'] > 0 and len(clinician['uncertainty']['cap_interval_90']) == 2


def test_later_visits_carry_earlier_measurements_forward(client, baseline):
    pid = client.post('/patients', json={'label': 'p', 'visit_date': '2026-09-12', 'measurements': baseline}).json()['id']
    later = client.post(f'/patients/{pid}/visits', json={'visit_date': '2026-10-12',
                                                         'measurements': {'BMXWT': baseline['BMXWT'] - 4}}).json()
    out = client.post(f'/patients/{pid}/twin', params={'view': 'clinician'}).json()
    assert out['visit_id'] == later['visit_id']
    assert out['inputs_last_measured']['BMXWT'] == '2026-10-12'
    assert out['inputs_last_measured']['LBXSATSI'] == '2026-09-12'
    first = client.post(f'/patients/{pid}/twin', params={'visit_id': later['visit_id'] - 1}).json()
    assert first['visit_date'] == '2026-09-12'
    assert client.post(f'/patients/{pid}/twin', params={'visit_id': 999}).status_code == 404


def test_timeline_runs_the_loop_and_projects_the_liver(client, baseline):
    pid = client.post('/patients', json={'label': 'p', 'visit_date': '2026-01-01', 'measurements': baseline}).json()['id']
    empty = client.get(f'/patients/{pid}/timeline').json()
    assert empty['plan'] is None and empty['loop'] is None and len(empty['visits']) == 1
    assert client.put(f'/patients/{pid}/plan', json={'start_date': '2026-01-01', 'energy_change_kcal': -5000}).status_code == 422
    assert client.put(f'/patients/{pid}/plan', json={'start_date': '2026-01-01',
                                                    'energy_change_kcal': -500}).json()['energy_change_kcal'] == -500
    for month, kg in ((1, -1.0), (2, -1.8), (3, -2.5)):
        client.post(f'/patients/{pid}/visits', json={'visit_date': f'2026-0{month + 1}-01',
                                                    'measurements': {'BMXWT': baseline['BMXWT'] + kg}})
    out = client.get(f'/patients/{pid}/timeline').json()
    assert [v['verdict'] for v in out['loop']['visits']] == ['start', 'on track', 'on track', 'on track']
    assert [p['months_ahead'] for p in out['projection']] == [3, 6, 12]
    assert out['projection'][-1]['chance_a_lot_of_fat_pct'] <= out['visits'][0]['chance_a_lot_of_fat_pct']
    assert not [k for k in keys(out) if CAP_KEY.search(k)]
    clinician = client.get(f'/patients/{pid}/timeline', params={'view': 'clinician'}).json()
    assert clinician['visits'][0]['predicted_cap'] > 0 and 'LUXCAPM' in clinician['visits'][0]['measured']


def test_current_inputs_and_json_helpers():
    visits = [{'id': 1, 'visit_date': '2026-01-01', 'measurements': {'BMXWT': 90, 'LBXSATSI': 40}},
              {'id': 2, 'visit_date': '2026-02-01', 'measurements': {'BMXWT': 88, 'LBXSATSI': None}}]
    (_, first, _), (_, second, when) = current_inputs(visits)
    assert first == {'BMXWT': 90, 'LBXSATSI': 40} and second == {'BMXWT': 88, 'LBXSATSI': 40}
    assert when == {'BMXWT': '2026-02-01', 'LBXSATSI': '2026-01-01'}
    cleaned = plain({'a': np.float32(1.5), 'b': [np.int64(2), float('nan')], 'c': (np.inf,)})
    assert cleaned == {'a': 1.5, 'b': [2, None], 'c': [None]} and not math.isnan(cleaned['a'])
    assert without_cap({'p_cap248': 0.9, 'predicted_cap': 300, 'cap_interval_90': [1, 2],
                        'nested': [{'grade': 'S3', 'direction': 'down'}]}) == \
        {'p_cap248': 0.9, 'nested': [{'direction': 'down'}]}


def test_saved_twin_is_returned_until_a_newer_visit(client, baseline):
    pid = client.post('/patients', json={'label': 's', 'visit_date': '2026-09-12', 'measurements': baseline}).json()['id']
    first = client.get(f'/patients/{pid}/twin').json()                    # no run yet: runs the twin
    again = client.get(f'/patients/{pid}/twin').json()                    # saved: same run, no rerun
    assert again['run_id'] == first['run_id'] and 'chance_a_lot_of_fat_pct' in again['twin']
    client.post(f'/patients/{pid}/visits', json={'visit_date': '2026-10-12', 'measurements': {'BMXWT': baseline['BMXWT'] - 2}})
    newer = client.get(f'/patients/{pid}/twin').json()                    # a newer visit: runs again
    assert newer['run_id'] != first['run_id'] and newer['visit_date'] == '2026-10-12'


def test_clinician_page_has_the_full_twin_but_not_the_whole_ranking(client, baseline):
    pid = client.post('/patients', json={'label': 'c', 'visit_date': '2026-09-12', 'measurements': baseline}).json()['id']
    page = client.get(f'/patients/{pid}/clinical').json()
    twin = page['twin']
    assert twin['current_state']['predicted_grade'] in ('S0', 'S1', 'S2', 'S3')
    assert twin['cap_explanation']['raising'] and len(twin['scenarios']) == 2
    assert 'ranked_recipes' not in twin and 0 < len(twin['top_recipes']) <= 10
    assert page['timeline']['visits'][0]['predicted_cap'] > 0 and 'alerts' in page
    patient = dict(app.dependency_overrides[current_user](), role='patient')
    app.dependency_overrides[current_user] = lambda: patient                 # patients cannot open it
    assert client.get(f'/patients/{pid}/clinical').status_code == 403
