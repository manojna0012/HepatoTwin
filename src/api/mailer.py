"""Send the one-time sign-in code by email (SMTP), when an account is set up in .env.

Settings (all in .env, never in the code):

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=<an app password, not the account password>
    SMTP_FROM=HepatoTwin <you@gmail.com>      (optional; defaults to SMTP_USER)

For Gmail the password must be an app password (Google Account -> Security -> 2-Step Verification ->
App passwords), because Google refuses the normal password over SMTP. Without SMTP_HOST the backend
keeps the prototype behaviour: the code is returned to the screen and marked as such.
"""
import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

TIMEOUT_SECONDS = 20


def settings():
    host = os.getenv('SMTP_HOST')
    if not host:
        return None
    user = os.getenv('SMTP_USER') or ''
    return {'host': host, 'port': int(os.getenv('SMTP_PORT') or 587), 'user': user,
            # Google shows app passwords in groups of four with spaces; the spaces are not part of it
            'password': (os.getenv('SMTP_PASSWORD') or '').replace(' ', ''), 'sender': os.getenv('SMTP_FROM') or user}


def configured():
    return settings() is not None


# Reserved for testing and documentation (RFC 2606); mail to them can never be delivered, so their codes are
# shown on screen instead, which keeps automated tests and demos from sending real email.
TEST_DOMAINS = ('example.com', 'example.org', 'example.net')


def is_test_address(email):
    domain = email.rsplit('@', 1)[-1].casefold()
    return domain in TEST_DOMAINS or domain.endswith(tuple('.' + d for d in TEST_DOMAINS)) or domain.endswith('.test')


def code_message(to, code, minutes, sender):
    message = EmailMessage()
    message['Subject'] = f'Your HepatoTwin sign-in code: {code}'
    message['From'] = sender
    message['To'] = to
    message.set_content(
        f'Your HepatoTwin sign-in code is {code}.\n\n'
        f'It works once and expires in {minutes} minutes. If you did not ask to sign in, you can ignore this email.\n\n'
        'HepatoTwin is a research prototype; it supports, and does not replace, care from your clinician.\n')
    message.add_alternative(
        '<div style="font-family:Arial,sans-serif;color:#111317;max-width:480px">'
        '<p style="font-size:15px;font-weight:600;margin:0 0 16px">HepatoTwin</p>'
        '<p style="margin:0 0 8px">Your sign-in code is</p>'
        f'<p style="font-size:28px;font-weight:700;letter-spacing:6px;margin:0 0 16px">{code}</p>'
        f'<p style="color:#5d626b;margin:0 0 8px">It works once and expires in {minutes} minutes. '
        'If you did not ask to sign in, you can ignore this email.</p>'
        '<p style="color:#8b9099;font-size:12px;margin:16px 0 0">HepatoTwin is a research prototype; it supports, '
        'and does not replace, care from your clinician.</p></div>', subtype='html')
    return message


def send_code(to, code, minutes):
    """Email the code; raises RuntimeError with a plain reason when it cannot be sent."""
    s = settings()
    if s is None:
        raise RuntimeError('Email is not set up.')
    try:
        with smtplib.SMTP(s['host'], s['port'], timeout=TIMEOUT_SECONDS) as server:
            server.starttls()
            if s['user']:
                server.login(s['user'], s['password'])
            server.send_message(code_message(to, code, minutes, s['sender']))
    except smtplib.SMTPAuthenticationError as error:
        raise RuntimeError('The email account refused the login; check SMTP_USER and the app password.') from error
    except (smtplib.SMTPException, OSError) as error:
        raise RuntimeError(f'The code could not be emailed ({error.__class__.__name__}).') from error
