import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { auth, type User } from '../api'
import { useAuth } from '../auth'
import { Icon } from '../components/Icon'

export function homeFor(user: User) {
  if (user.role === 'clinician') return '/patients'
  return user.patient_id ? '/today' : '/signup'
}

const POINTS = [
  ['Liver fat from a routine check-up', 'Your weight, waist and standard blood tests give the chance of liver fat, checked against FibroScan results.'],
  ['Recipes chosen by the evidence', 'Meals in your own cuisine that meet every nutrition target for you, with the studies behind each one.'],
  ['Followed month by month', 'Each check-in updates your twin, and your clinician sees the same results.'],
]
const RESEND_SECONDS = 30

export function SignIn() {
  const { signIn } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [sent, setSent] = useState<null | { delivery: 'email' | 'screen'; code?: string; minutes: number }>(null)
  const [wait, setWait] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const codeInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (wait <= 0) return
    const timer = setTimeout(() => setWait(w => w - 1), 1000)
    return () => clearTimeout(timer)
  }, [wait])

  const askForCode = async (e?: FormEvent) => {
    e?.preventDefault()
    setBusy(true); setError(null)
    try {
      const out = await auth.requestCode(email.trim())
      setSent({ delivery: out.delivery, code: out.code, minutes: out.expires_in_minutes })
      setWait(RESEND_SECONDS); setCode('')
      setTimeout(() => codeInput.current?.focus(), 0)
    } catch (err) { setError((err as Error).message) }
    setBusy(false)
  }
  const enter = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true); setError(null)
    try { navigate(homeFor(await signIn(email.trim(), code.trim()))) } catch (err) { setError((err as Error).message); setBusy(false) }
  }

  return (
    <div className="signin-page">
      <section className="signin-story">
        <div className="wordmark">HepatoTwin</div>
        <div className="signin-copy">
          <p className="eyebrow blue">A digital twin of your liver</p>
          <h1 className="signin-title">Know your liver. Change what you eat. See it work.</h1>
          <ul className="signin-points">
            {POINTS.map(([title, text]) => (
              <li key={title}><span className="tick"><Icon name="check" /></span><div><strong>{title}</strong><p className="muted">{text}</p></div></li>
            ))}
          </ul>
        </div>
        <p className="small muted signin-foot">Research prototype. It supports, and does not replace, care from your clinician.</p>
      </section>

      <section className="signin-panel">
        <div className="signin-form">
          {sent === null ? (
            <form className="page" onSubmit={askForCode}>
              <div><h1>Sign in</h1><p className="muted">No password. We email you a 6-digit code each time you sign in.</p></div>
              <label className="field"><span>Email</span>
                <input className="input lg" type="email" required value={email} onChange={e => setEmail(e.target.value)}
                       autoComplete="email" placeholder="you@example.com" autoFocus />
              </label>
              {error && <p className="error">{error}</p>}
              <button className="btn primary lg" disabled={busy}>{busy ? 'Sending…' : 'Send me a code'}</button>
              <p className="small muted">New here? Use the email your clinic knows; you will set up your account next.</p>
            </form>
          ) : (
            <form className="page" onSubmit={enter}>
              <div>
                <h1>{sent.delivery === 'email' ? 'Check your email' : 'Enter your code'}</h1>
                <p className="muted">
                  {sent.delivery === 'email'
                    ? <>We sent a 6-digit code to <strong>{email}</strong>. It expires in {sent.minutes} minutes; check your spam folder if it is not there.</>
                    : <>Enter the code for <strong>{email}</strong>.</>}
                </p>
              </div>
              {sent.delivery === 'screen' && (
                <div className="notice">
                  <p className="eyebrow blue">Email is not set up</p>
                  <p className="muted small">This copy of HepatoTwin cannot send email yet, so the code is shown here.</p>
                  <p className="code">{sent.code}</p>
                </div>
              )}
              <label className="field"><span>Code</span>
                <input ref={codeInput} className="input lg code-input" inputMode="numeric" autoComplete="one-time-code"
                       pattern="\d{6}" maxLength={6} required value={code} onChange={e => setCode(e.target.value.replace(/\D/g, ''))} />
              </label>
              {error && <p className="error">{error}</p>}
              <button className="btn primary lg" disabled={busy || code.length !== 6}>{busy ? 'Signing in…' : 'Sign in'}</button>
              <div className="row small">
                <button type="button" className="link" disabled={wait > 0 || busy} onClick={() => askForCode()}>
                  {wait > 0 ? `Send a new code in ${wait}s` : 'Send a new code'}
                </button>
                <span className="muted">·</span>
                <button type="button" className="link" onClick={() => { setSent(null); setCode(''); setError(null) }}>Use another email</button>
              </div>
            </form>
          )}
        </div>
      </section>
    </div>
  )
}
