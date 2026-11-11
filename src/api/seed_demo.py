"""Demo accounts for the dashboard: a clinic, its clinician and four patients who are real held-out NHANES
participants (never used to train the model), signed up through the API exactly as a new user would be.

    .venv/Scripts/python.exe -m src.api.seed_demo            # create the accounts that do not exist yet
    .venv/Scripts/python.exe -m src.api.seed_demo --reset    # start the five demo patients again from sign-up

Sign in with any of the emails below. They use the reserved example.org domain, so the sign-in code is
shown on screen and no email is sent (src/api/mailer.py). Running the script again skips accounts that
already exist; --reset first deletes the five demo patients (plans, check-ins, twin runs) so a rehearsal
starts from fresh sign-ups. Dr. Rao, the clinic and every other account are kept.

    dr.rao@example.org          clinician, Demo clinic (clinic code HT-4821)
    demo.patient@example.org    NHANES 111588: 49-year-old man, diabetes pills (his plan needs approval)
    patient2@example.org        NHANES 116375, Mexican cuisine
    patient3@example.org        NHANES 120969, European cuisine, vegetarian
    patient4@example.org        NHANES 135329, American cuisine
    patient5@example.org        NHANES 115065: 46-year-old woman with little sign of liver fat (measured CAP 203)
"""
import argparse
from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

from src.api import auth
from src.api.app import DIET_FILE, DATA_FILE, app, get_store, known_measurements
from src.api.remove_test_accounts import remove_account

CLINIC = ('Demo clinic', 'HT-4821')
CLINICIAN = 'dr.rao@example.org'
PATIENTS = [
    ('demo.patient@example.org', 111588, 'American', {'diet': 'anything', 'likes': ['beans', 'chicken', 'peppers'],
                                                      'dislikes': ['mushroom'], 'household': 1}),
    ('patient2@example.org', 116375, 'Mexican', {'diet': 'anything', 'avoid': ['pork'], 'household': 3}),
    ('patient3@example.org', 120969, 'European', {'diet': 'vegetarian', 'allergies': ['tree_nuts'], 'household': 2}),
    ('patient4@example.org', 135329, 'American', {'diet': 'anything', 'max_minutes': 45, 'household': 1}),
    ('patient5@example.org', 115065, 'Asian', {'diet': 'pescatarian', 'household': 2}),
]


def nhanes_row(seqn, data, diet):
    row = data[data.SEQN == seqn].merge(diet, on='SEQN', how='left', suffixes=('', '_diet')).iloc[0]
    known = known_measurements()
    return {k: float(v) for k, v in row.items() if k in known and not k.endswith('_diet') and not isinstance(v, str)
            and pd.notna(v)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reset', action='store_true', help='delete the five demo patients first, then create them again')
    args = parser.parse_args()
    store = get_store()
    if args.reset:
        for email, *_ in PATIENTS:
            user = store.user(email=email)
            if user:
                remove_account(store, user)
                print(f'{email:28s} removed')
    auth.create_clinician(store, CLINICIAN, *CLINIC)
    client = TestClient(app)

    def sign_in(email):
        code = client.post('/auth/code', json={'email': email}).json()['code']
        token = client.post('/auth/session', json={'email': email, 'code': code}).json()['token']
        return {'Authorization': f'Bearer {token}'}

    data = pd.read_csv(DATA_FILE)
    diet = pd.read_csv(DIET_FILE) if DIET_FILE.exists() else pd.DataFrame({'SEQN': data.SEQN})
    today = date.today().isoformat()
    print(f'{CLINICIAN:28s} clinician, {CLINIC[0]} (code {CLINIC[1]})')
    for email, seqn, cuisine, food in PATIENTS:
        existing = store.user(email=email)
        if existing and existing['patient_id']:
            print(f'{email:28s} already exists (patient {existing["patient_id"]})')
            continue
        me = sign_in(email)
        dishes = client.get('/dishes', params={'cuisine': cuisine, 'diet': food.get('diet', 'anything'),
                                               'allergies': ','.join(food.get('allergies', []))}, headers=me).json()
        made = client.post('/patients', headers=me, json={
            'visit_date': today, 'measurements': nhanes_row(seqn, data, diet), 'cuisine': cuisine,
            'favourites': [d['recipe_id'] for d in dishes[:3]], 'food': food, 'clinic_code': CLINIC[1]})
        made.raise_for_status()
        pid = made.json()['id']
        twin = client.post(f'/patients/{pid}/twin', headers=me)
        twin.raise_for_status()
        chance = twin.json()['twin']['chance_a_lot_of_fat_pct']
        print(f'{email:28s} NHANES {seqn}, {cuisine}: chance of a lot of liver fat {chance}%')


if __name__ == '__main__':
    main()
