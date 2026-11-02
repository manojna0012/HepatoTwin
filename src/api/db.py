"""SQLite storage for the twin's patients, their dated visits and each twin run (Proposal section 16).

Three tables:

- patients: one row per patient (a label chosen by the clinic, never a name).
- visits: one row per dated visit, holding whatever was measured that day as NHANES-coded values
  (a later visit may hold only a weight and a food log). The twin's current inputs are the
  baseline visit with each later measurement carried forward from the most recent visit that
  recorded it.
- twin_runs: the twin's full output each time it is run, so its predictions can later be compared
  with what was observed (the re-assessment loop, Proposal section 17.5).
- plans: the energy change (kcal/day) the patient is following and from when; the latest applies.

Values are stored as JSON text; the standard-library sqlite3 module is enough at this scale.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / 'data' / 'app' / 'hepatotwin.sqlite'

SCHEMA = """
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
CREATE INDEX IF NOT EXISTS visits_by_patient ON visits(patient_id, visit_date);
CREATE INDEX IF NOT EXISTS runs_by_patient ON twin_runs(patient_id, visit_id);
"""


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

    def add_patient(self, label):
        cur = self.conn.execute('INSERT INTO patients (label, created_at) VALUES (?, ?)', (label, _now()))
        self.conn.commit()
        return cur.lastrowid

    def patients(self):
        rows = self.conn.execute(
            'SELECT p.id, p.label, p.created_at, COUNT(v.id) AS visits, MAX(v.visit_date) AS last_visit '
            'FROM patients p LEFT JOIN visits v ON v.patient_id = p.id GROUP BY p.id ORDER BY p.id').fetchall()
        return [dict(r) for r in rows]

    def patient(self, patient_id):
        row = self.conn.execute('SELECT id, label, created_at FROM patients WHERE id = ?', (patient_id,)).fetchone()
        return None if row is None else dict(row)

    def add_visit(self, patient_id, visit_date, measurements):
        cur = self.conn.execute(
            'INSERT INTO visits (patient_id, visit_date, measurements, created_at) VALUES (?, ?, ?, ?)',
            (patient_id, visit_date, json.dumps(measurements), _now()))
        self.conn.commit()
        return cur.lastrowid

    def visits(self, patient_id):
        """Visits in date order (ties in the order they were entered)."""
        rows = self.conn.execute(
            'SELECT id, visit_date, measurements FROM visits WHERE patient_id = ? ORDER BY visit_date, id',
            (patient_id,)).fetchall()
        return [{'id': r['id'], 'visit_date': r['visit_date'], 'measurements': json.loads(r['measurements'])}
                for r in rows]

    def set_plan(self, patient_id, start_date, energy_change_kcal):
        cur = self.conn.execute(
            'INSERT INTO plans (patient_id, start_date, energy_change_kcal, created_at) VALUES (?, ?, ?, ?)',
            (patient_id, start_date, energy_change_kcal, _now()))
        self.conn.commit()
        return cur.lastrowid

    def plan(self, patient_id):
        """The patient's current plan (the most recently set), or None."""
        row = self.conn.execute(
            'SELECT start_date, energy_change_kcal FROM plans WHERE patient_id = ? ORDER BY id DESC LIMIT 1',
            (patient_id,)).fetchone()
        return None if row is None else dict(row)

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
