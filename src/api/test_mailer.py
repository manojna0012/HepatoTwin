import smtplib

import pytest
from fastapi.testclient import TestClient

from src.api import mailer
from src.api.app import app, get_store
from src.api.db import Store

SETTINGS = {'host': 'smtp.test', 'port': 587, 'user': 'sender@example.org', 'password': 'app-pass',
            'sender': 'HepatoTwin <sender@example.org>'}


class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        if password != 'app-pass':
            raise smtplib.SMTPAuthenticationError(535, b'bad')

    def send_message(self, message):
        FakeSMTP.sent.append(message)


@pytest.fixture
def client(tmp_path):
    store = Store(tmp_path / 'test.sqlite')
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_with_email_set_up_the_code_is_emailed_and_not_returned(client, monkeypatch):  # FakeSMTP: nothing leaves
    monkeypatch.setattr(mailer, 'settings', lambda: SETTINGS)
    monkeypatch.setattr(mailer.smtplib, 'SMTP', FakeSMTP)
    FakeSMTP.sent.clear()
    out = client.post('/auth/code', json={'email': 'asha@clinic-mail.net'}).json()
    assert out['delivery'] == 'email' and 'code' not in out
    message = FakeSMTP.sent[-1]
    assert message['To'] == 'asha@clinic-mail.net' and message['From'] == SETTINGS['sender']
    code = message['Subject'].split()[-1]
    assert len(code) == 6 and code in message.get_body(('plain',)).get_content()
    assert client.post('/auth/session', json={'email': 'asha@clinic-mail.net', 'code': code}).status_code == 200


def test_a_refused_login_is_reported_plainly(client, monkeypatch):
    monkeypatch.setattr(mailer, 'settings', lambda: {**SETTINGS, 'password': 'wrong'})
    monkeypatch.setattr(mailer.smtplib, 'SMTP', FakeSMTP)
    response = client.post('/auth/code', json={'email': 'asha@clinic-mail.net'})
    assert response.status_code == 502 and 'app password' in response.json()['detail']


def test_without_email_the_code_is_shown(client):
    out = client.post('/auth/code', json={'email': 'asha@example.org'}).json()
    assert out['delivery'] == 'screen' and len(out['code']) == 6


def test_reserved_test_addresses_are_never_emailed(client, monkeypatch):
    monkeypatch.setattr(mailer, 'settings', lambda: SETTINGS)
    monkeypatch.setattr(mailer.smtplib, 'SMTP', FakeSMTP)
    FakeSMTP.sent.clear()
    out = client.post('/auth/code', json={'email': 'robot@example.org'}).json()
    assert out['delivery'] == 'screen' and len(out['code']) == 6 and FakeSMTP.sent == []
    assert mailer.is_test_address('a@sub.example.com') and mailer.is_test_address('a@demo.test')
    assert not mailer.is_test_address('someone@gmail.com')
