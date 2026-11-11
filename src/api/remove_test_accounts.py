"""Remove the accounts made by the automated browser tests (frontend/e2e), so the demo clinic shows only the
demo patients and real users.

    .venv/Scripts/python.exe -m src.api.remove_test_accounts          # list what would go
    .venv/Scripts/python.exe -m src.api.remove_test_accounts --yes    # remove it

Test accounts are <name>.<13-digit timestamp>@example.org (patient., newpatient., plan.). Demo accounts
(src/api/seed_demo.py) and every other address are never touched. A test patient's visits, plans and
twin runs go with it.
"""
import argparse
import re

from src.api.app import get_store

TEST_EMAIL = re.compile(r'^(patient|newpatient|plan)\.\d{13}@example\.org$')


def remove_account(store, account):
    """Delete one account ({id, email, patient_id}) with its sessions, codes and, for a patient, the record."""
    conn = store.conn
    conn.execute('DELETE FROM sessions WHERE user_id = ?', (account['id'],))
    conn.execute('DELETE FROM login_codes WHERE email = ?', (account['email'],))
    conn.execute('DELETE FROM users WHERE id = ?', (account['id'],))
    if account['patient_id'] is not None:
        for table in ('twin_runs', 'plans', 'visits'):
            conn.execute(f'DELETE FROM {table} WHERE patient_id = ?', (account['patient_id'],))
        conn.execute('DELETE FROM patients WHERE id = ?', (account['patient_id'],))
    conn.commit()


def test_accounts(store):
    rows = store.conn.execute('SELECT id, email, patient_id FROM users').fetchall()
    return [dict(r) for r in rows if TEST_EMAIL.match(r['email'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--yes', action='store_true', help='remove them (otherwise only list)')
    args = parser.parse_args()
    store = get_store()
    accounts = test_accounts(store)
    for a in accounts:
        print(f"{a['email']:40s} patient {a['patient_id']}")
    if not args.yes:
        print(f'{len(accounts)} test accounts; run with --yes to remove them.')
        return
    for a in accounts:
        remove_account(store, a)
    print(f'Removed {len(accounts)} test accounts.')


if __name__ == '__main__':
    main()
