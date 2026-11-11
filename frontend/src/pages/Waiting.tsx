import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { Shell } from '../components/Shell'
import { getPatient, type PatientRecord } from '../patient'

const POLL_MS = 15000

export function Waiting() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [patient, setPatient] = useState<PatientRecord | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    if (user?.patient_id) getPatient(user.patient_id).then(setPatient).catch(e => setError(e.message))
  }, [user?.patient_id])
  useEffect(() => { load(); const timer = setInterval(load, POLL_MS); return () => clearInterval(timer) }, [load])

  if (error) return <Shell title="Your plan"><p className="error">{error}</p></Shell>
  if (!patient) return <Shell title="Your plan"><div className="card narrow"><p className="muted">Loading…</p></div></Shell>
  const plan = patient.plan
  if (!plan) return (
    <Shell title="Your plan"><div className="page narrow"><p className="muted">You have not chosen a plan yet.</p>
      <div className="row"><button className="btn primary" onClick={() => navigate('/choose-plan')}>Choose my plan</button></div></div></Shell>
  )
  const kcal = Math.abs(Math.round(plan.energy_change_kcal))

  if (plan.status === 'active') return (
    <Shell title="Your plan is ready">
      <div className="page narrow">
        <section className="notice">
          <p className="eyebrow blue">{plan.set_by === 'clinician' ? 'New from your clinician' : 'Ready'}</p>
          <h2>Your plan is ready</h2>
          <p>{plan.set_by === 'clinician' ? 'Your clinician approved' : 'You chose'} <strong>{kcal} kcal less a day</strong> and <strong>{plan.walking_min_per_week} minutes of walking</strong> a week{plan.pace ? ` (the ${plan.pace} pace)` : ''}.</p>
          {plan.note && <p className="muted">“{plan.note}”</p>}
        </section>
        <div className="row"><button className="btn primary" onClick={() => navigate('/choose-recipes')}>Choose my recipes</button></div>
      </div>
    </Shell>
  )

  return (
    <Shell title="Your plan" subtitle="Your clinician is checking your plan">
      <div className="page narrow">
        {patient.clinic_id === null && (
          <section className="notice"><p className="eyebrow blue">No clinic linked</p>
            <p>Your plan needs a clinician, but your account has no clinic code. Ask your clinic for its code.</p></section>
        )}
        <section className="card">
          <h2>What happens next</h2>
          <ol className="steps">
            <li className="done">You chose the {plan.pace ?? 'suggested'} pace: {kcal} kcal less a day and {plan.walking_min_per_week} minutes of walking a week.</li>
            <li className="now">Your clinician reviews it, usually within a day. They may keep it or suggest a gentler start.</li>
            <li>We let you know here when it is ready, and you choose your recipes.</li>
          </ol>
        </section>
        <div className="row">
          <button className="btn" onClick={load}>Check again</button>
          <button className="btn" onClick={() => navigate('/liver')}>Back to my results</button>
        </div>
        <p className="small muted">This page checks by itself every 15 seconds.</p>
      </div>
    </Shell>
  )
}
