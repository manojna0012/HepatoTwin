"""SQLite storage for the twin's patients, their dated visits and each twin run (Proposal section 16).

Tables:

- patients: one row per patient (a label chosen by the clinic, never a name), with the cuisine they
  cook, the favourite dishes picked at sign-up (recipe ids) and their clinic.
- visits: one row per dated visit, holding whatever was measured that day as NHANES-coded values
  (a later visit may hold only a weight), and at check-ins what the patient says they followed. The
  twin's current inputs are the baseline visit with each later measurement carried forward from the
  most recent visit that recorded it.
- twin_runs: the twin's full output each time it is run, so its predictions can later be compared
  with what was observed (the re-assessment loop, Proposal section 17.5).
- plans: every plan set, newest last: pace, the daily calorie change and walking minutes, the recipe
  style step, the next check-in, who set it and whether it awaits a clinician's approval.
- clinics, users, login_codes, sessions: sign-in (src/api/auth.py).

Values are stored as JSON text; the standard-library sqlite3 module is enough at this scale. Columns
added after the first version are added to an existing database when it is opened.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / 'data' / 'app' / 'hepatotwin.sqlite'

SCHEMA = """
CREATE TABLE IF NOT EXISTS clinics (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    code TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    visit_date TEXT NOT NULL,
    measurements TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS twin_runs (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    visit_id INTEGER NOT NULL REFERENCES visits(id) ON DELETE CASCADE,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    start_date TEXT NOT NULL,
    energy_change_kcal REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('patient', 'clinician')),
    patient_id INTEGER REFERENCES patients(id) ON DELETE SET NULL,
    clinic_id INTEGER REFERENCES clinics(id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS login_codes (
    email TEXT PRIMARY KEY,
    code_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS visits_by_patient ON visits(patient_id, visit_date);
CREATE INDEX IF NOT EXISTS runs_by_patient ON twin_runs(patient_id, visit_id);
"""
# Columns added after the first version: (table, column, definition).
ADDED_COLUMNS = [
    ('patients', 'cuisine', 'TEXT'),
    ('patients', 'favourites', "TEXT NOT NULL DEFAULT '[]'"),
    ('patients', 'clinic_id', 'INTEGER REFERENCES clinics(id)'),
    ('patients', 'food', "TEXT NOT NULL DEFAULT '{}'"),
    ('visits', 'followed', 'TEXT'),
    ('plans', 'pace', 'TEXT'),
    ('plans', 'walking_min_per_week', 'REAL NOT NULL DEFAULT 0'),
    ('plans', 'style_step', 'INTEGER NOT NULL DEFAULT 0'),
    ('plans', 'next_checkin', 'TEXT'),
    ('plans', 'status', "TEXT NOT NULL DEFAULT 'active'"),
    ('plans', 'set_by', "TEXT NOT NULL DEFAULT 'clinician'"),
    ('plans', 'approved_by', 'INTEGER REFERENCES users(id)'),
    ('plans', 'note', 'TEXT'),
]
PLAN_FIELDS = ('id', 'start_date', 'energy_change_kcal', 'pace', 'walking_min_per_week', 'style_step',
               'next_checkin', 'status', 'set_by', 'approved_by', 'note', 'created_at')


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Store:
    def __init__(self, path=DEFAULT_DB):
        path = str(path)
        if path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys = ON')
        self.conn.executescript(SCHEMA)
        for table, column, definition in ADDED_COLUMNS:
            existing = {r['name'] for r in self.conn.execute(f'PRAGMA table_info({table})')}
            if column not in existing:
                self.conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
        self.conn.commit()

    # ---------------- patients ----------------
    def add_patient(self, label, cuisine=None, favourites=(), clinic_id=None, food=None):
        cur = self.conn.execute(
            'INSERT INTO patients (label, created_at, cuisine, favourites, clinic_id, food) VALUES (?, ?, ?, ?, ?, ?)',
            (label, _now(), cuisine, json.dumps([str(f) for f in favourites]), clinic_id, json.dumps(food or {})))
        self.conn.commit()
        return cur.lastrowid

    def update_profile(self, patient_id, cuisine=None, favourites=None, food=None):
        if food is not None:
            self.conn.execute('UPDATE patients SET food = ? WHERE id = ?', (json.dumps(food), patient_id))
        if cuisine is not None:
            self.conn.execute('UPDATE patients SET cuisine = ? WHERE id = ?', (cuisine, patient_id))
        if favourites is not None:
            self.conn.execute('UPDATE patients SET favourites = ? WHERE id = ?',
                              (json.dumps([str(f) for f in favourites]), patient_id))
        self.conn.commit()

    def patients(self, clinic_id=None):
        where, args = ('WHERE p.clinic_id = ?', (clinic_id,)) if clinic_id is not None else ('', ())
        rows = self.conn.execute(
            'SELECT p.id, p.label, p.created_at, p.cuisine, p.clinic_id, COUNT(v.id) AS visits, '
            f'MAX(v.visit_date) AS last_visit FROM patients p LEFT JOIN visits v ON v.patient_id = p.id {where} '
            'GROUP BY p.id ORDER BY p.id', args).fetchall()
        return [dict(r) for r in rows]

    def patient(self, patient_id):
        row = self.conn.execute('SELECT id, label, created_at, cuisine, favourites, clinic_id, food FROM patients '
                                'WHERE id = ?', (patient_id,)).fetchone()
        if row is None:
            return None
        return {**dict(row), 'favourites': json.loads(row['favourites'] or '[]'), 'food': json.loads(row['food'] or '{}')}

    # ---------------- visits ----------------
    def add_visit(self, patient_id, visit_date, measurements, followed=None):
        cur = self.conn.execute(
            'INSERT INTO visits (patient_id, visit_date, measurements, created_at, followed) VALUES (?, ?, ?, ?, ?)',
            (patient_id, visit_date, json.dumps(measurements), _now(), json.dumps(followed) if followed else None))
        self.conn.commit()
        return cur.lastrowid

    def visits(self, patient_id):
        """Visits in date order (ties in the order they were entered)."""
        rows = self.conn.execute(
            'SELECT id, visit_date, measurements, followed FROM visits WHERE patient_id = ? ORDER BY visit_date, id',
            (patient_id,)).fetchall()
        return [{'id': r['id'], 'visit_date': r['visit_date'], 'measurements': json.loads(r['measurements']),
                 'followed': json.loads(r['followed']) if r['followed'] else None} for r in rows]

    # ---------------- plans ----------------
    def set_plan(self, patient_id, start_date, energy_change_kcal, pace=None, walking_min_per_week=0,
                 style_step=0, next_checkin=None, status='active', set_by='clinician', approved_by=None, note=None):
        cur = self.conn.execute(
            'INSERT INTO plans (patient_id, start_date, energy_change_kcal, created_at, pace, walking_min_per_week, '
            'style_step, next_checkin, status, set_by, approved_by, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (patient_id, start_date, energy_change_kcal, _now(), pace, walking_min_per_week, style_step,
             next_checkin, status, set_by, approved_by, note))
        self.conn.commit()
        return cur.lastrowid

    def plan(self, patient_id):
        """The patient's current plan (the most recently set), or None."""
        row = self.conn.execute(
            f'SELECT {", ".join(PLAN_FIELDS)} FROM plans WHERE patient_id = ? ORDER BY id DESC LIMIT 1',
            (patient_id,)).fetchone()
        return None if row is None else dict(row)

    def plans(self, patient_id):
        rows = self.conn.execute(f'SELECT {", ".join(PLAN_FIELDS)} FROM plans WHERE patient_id = ? ORDER BY id',
                                 (patient_id,)).fetchall()
        return [dict(r) for r in rows]

    def update_plan(self, plan_id, **fields):
        allowed = {k: v for k, v in fields.items() if k in PLAN_FIELDS and k not in ('id', 'created_at')}
        if allowed:
            sets = ', '.join(f'{k} = ?' for k in allowed)
            self.conn.execute(f'UPDATE plans SET {sets} WHERE id = ?', (*allowed.values(), plan_id))
            self.conn.commit()

    # ---------------- twin runs ----------------
    def add_run(self, patient_id, visit_id, result):
        cur = self.conn.execute(
            'INSERT INTO twin_runs (patient_id, visit_id, result, created_at) VALUES (?, ?, ?, ?)',
            (patient_id, visit_id, json.dumps(result), _now()))
        self.conn.commit()
        return cur.lastrowid

    def runs(self, patient_id):
        """The latest twin run for each visit, in visit order."""
        rows = self.conn.execute(
            'SELECT r.id, r.visit_id, r.result, r.created_at FROM twin_runs r JOIN visits v ON v.id = r.visit_id '
            'WHERE r.patient_id = ? AND r.id = (SELECT MAX(id) FROM twin_runs WHERE visit_id = r.visit_id) '
            'ORDER BY v.visit_date, v.id', (patient_id,)).fetchall()
        return [{'id': r['id'], 'visit_id': r['visit_id'], 'created_at': r['created_at'],
                 'result': json.loads(r['result'])} for r in rows]

    def latest_run(self, patient_id):
        runs = self.runs(patient_id)
        return runs[-1] if runs else None

    # ---------------- clinics and users ----------------
    def add_clinic(self, name, code):
        cur = self.conn.execute('INSERT INTO clinics (name, code) VALUES (?, ?)', (name, code))
        self.conn.commit()
        return cur.lastrowid

    def clinic_by_code(self, code):
        row = self.conn.execute('SELECT id, name, code FROM clinics WHERE code = ?', (code,)).fetchone()
        return None if row is None else dict(row)

    def add_user(self, email, role, patient_id=None, clinic_id=None):
        cur = self.conn.execute(
            'INSERT INTO users (email, role, patient_id, clinic_id, created_at) VALUES (?, ?, ?, ?, ?)',
            (email.lower(), role, patient_id, clinic_id, _now()))
        self.conn.commit()
        return cur.lastrowid

    def user(self, user_id=None, email=None):
        row = (self.conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)) if user_id is not None else
               self.conn.execute('SELECT * FROM users WHERE email = ?', (email.lower(),))).fetchone()
        return None if row is None else dict(row)

    def link_user(self, user_id, patient_id=None, clinic_id=None):
        self.conn.execute('UPDATE users SET patient_id = COALESCE(?, patient_id), clinic_id = COALESCE(?, clinic_id) '
                          'WHERE id = ?', (patient_id, clinic_id, user_id))
        self.conn.commit()

    def save_login_code(self, email, code_hash, expires_at):
        self.conn.execute('INSERT OR REPLACE INTO login_codes (email, code_hash, expires_at) VALUES (?, ?, ?)',
                          (email.lower(), code_hash, expires_at))
        self.conn.commit()

    def take_login_code(self, email):
        row = self.conn.execute('SELECT code_hash, expires_at FROM login_codes WHERE email = ?',
                                (email.lower(),)).fetchone()
        self.conn.execute('DELETE FROM login_codes WHERE email = ?', (email.lower(),))
        self.conn.commit()
        return None if row is None else dict(row)

    def save_session(self, token_hash, user_id, expires_at):
        self.conn.execute('INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)',
                          (token_hash, user_id, expires_at))
        self.conn.commit()

    def session(self, token_hash):
        row = self.conn.execute('SELECT user_id, expires_at FROM sessions WHERE token_hash = ?',
                                (token_hash,)).fetchone()
        return None if row is None else dict(row)

    def end_session(self, token_hash):
        self.conn.execute('DELETE FROM sessions WHERE token_hash = ?', (token_hash,))
        self.conn.commit()


def current_inputs(visits):
    """Twin inputs at each visit: every measurement carried forward from the most recent visit that
    recorded it. Returns (visit, inputs, last_measured) per visit, last_measured giving each field's date."""
    inputs, measured, out = {}, {}, []
    for visit in visits:
        for name, value in visit['measurements'].items():
            if value is not None:
                inputs[name] = value
                measured[name] = visit['visit_date']
        out.append((visit, dict(inputs), dict(measured)))
    return out
