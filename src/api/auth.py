"""Sign-in without passwords, and who may see which patient.

Sign-in is a one-time code: POST /auth/code with an email creates a six-digit code valid for
CODE_MINUTES; POST /auth/session with the email and code returns a bearer token valid for
SESSION_DAYS. Only hashes of codes and tokens are stored. Sending the code by email is outside this
prototype: the code is returned in the response (marked as such), as a demo would show it.

Roles: a patient sees only their own record; a clinician sees the patients of their clinic. Patients
join a clinic with the clinic code their clinician gives them; clinician accounts are created with
create_clinician (there is no self-sign-up for clinicians).
"""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

CODE_MINUTES = 10
SESSION_DAYS = 30


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _later(**delta):
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat(timespec='seconds')


def _expired(when):
    return datetime.fromisoformat(when) < datetime.now(timezone.utc)


def request_code(store, email):
    code = f'{secrets.randbelow(10 ** 6):06d}'
    store.save_login_code(email, _hash(code), _later(minutes=CODE_MINUTES))
    return code


def start_session(store, email, code):
    """A new session token for a valid code; the account is created as a patient on first sign-in."""
    saved = store.take_login_code(email)
    if saved is None or _expired(saved['expires_at']) or not hmac.compare_digest(saved['code_hash'], _hash(code)):
        raise HTTPException(401, 'The code is wrong or has expired.')
    user = store.user(email=email) or store.user(store.add_user(email, 'patient'))
    token = secrets.token_urlsafe(32)
    store.save_session(_hash(token), user['id'], _later(days=SESSION_DAYS))
    return token, user


def user_for_token(store, token):
    session = store.session(_hash(token)) if token else None
    if session is None or _expired(session['expires_at']):
        raise HTTPException(401, 'Please sign in.')
    return store.user(session['user_id'])


def end_session(store, token):
    store.end_session(_hash(token))


def create_clinician(store, email, clinic_name, clinic_code):
    """A clinician account and its clinic (made if new). Returns the user id."""
    clinic = store.clinic_by_code(clinic_code)
    clinic_id = clinic['id'] if clinic else store.add_clinic(clinic_name, clinic_code)
    existing = store.user(email=email)
    if existing:
        return existing['id']
    return store.add_user(email, 'clinician', clinic_id=clinic_id)


def can_see(user, patient):
    if user['role'] == 'clinician':
        return patient['clinic_id'] is not None and patient['clinic_id'] == user['clinic_id']
    return user['patient_id'] == patient['id']


def require_clinician(user):
    if user['role'] != 'clinician':
        raise HTTPException(403, 'Only a clinician can do this.')
