import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { Shell } from '../components/Shell'
import type { Suggestion } from '../patient'
import { PACE_WORDS, shortDate, usePatientGet, type Timeline } from '../progress'

const STEP_WORDS = ['Closest to what you eat now', 'A small step healthier', 'A bigger step healthier', 'The healthiest']

export function MyPlan() {
  const navigate = useNavigate()
  const line = usePatientGet<Timeline>('/timeline')
  const sugg = usePatientGet<Suggestion>('/plan/suggestion')
  const [busy, setBusy] = useState(false)
  const plan = line.data?.plan

  if (line.error) return <Shell title="My plan"><p className="error">{line.error}</p></Shell>
  if (!line.data) return <Shell title="My plan"><div className="card narrow"><p className="muted">Loading your plan…</p></div></Shell>
  if (!plan) return <Shell title="My plan"><div className="page narrow"><section className="notice"><p>You have not chosen a plan yet.</p>
    <div><button className="btn primary" onClick={() => navigate('/choose-plan')}>Choose my plan</button></div></section></div></Shell>

  const kcal = Math.round(-plan.energy_change_kcal)
  const option = sugg.data?.options.find(o => o.pace === plan.pace)
  const changes = line.data.loop?.plan_changes ?? []
  const setStep = async (step: number) => {
    setBusy(true)
    try { await api(`/patients/${line.pid}/plan/style`, { method: 'PUT', body: { style_step: step } }); line.reload() } finally { setBusy(false) }
  }

  return (
    <Shell title="My plan" subtitle={plan.status === 'awaiting_approval' ? 'Waiting for your clinician to approve it.' : `Started ${shortDate(plan.start_date)}.`}>
      <div className="page">
        <div className="stat-row">
          <div className="card stat"><span className="eyebrow">Pace</span><strong>{plan.pace ? PACE_WORDS[plan.pace] ?? plan.pace : 'Set by your clinician'}</strong>
            <span className="muted small">{plan.set_by === 'clinician' ? 'Set by your clinician' : plan.status === 'active' ? 'Chosen by you' : 'Waiting for approval'}</span></div>
          <div className="card stat"><span className="eyebrow">Eat each day</span><strong>{kcal > 0 ? `${kcal} kcal less` : kcal < 0 ? `${-kcal} kcal more` : 'As now'}</strong>
            <span className="muted small">Than you need to keep your weight</span></div>
          <div className="card stat"><span className="eyebrow">Walk each week</span><strong>{Math.round(plan.walking_min_per_week)} min</strong>
            <span className="muted small">About {Math.round(plan.walking_min_per_week / 5)} min on 5 days</span></div>
          <div className="card stat"><span className="eyebrow">In 12 months</span>
            <strong>{option ? `${Math.abs(option.expected_change_12m_kg).toFixed(1)} kg ${option.expected_change_12m_kg <= 0 ? 'lighter' : 'heavier'}` : '—'}</strong>
            <span className="muted small">{option ? `Up to ${Math.abs(option.if_followed_change_12m_kg).toFixed(1)} kg if you follow it fully` : 'Expected if you follow most of it'}</span></div>
        </div>

        {plan.note && <section className="notice"><span className="eyebrow">Note from your clinician</span><p>{plan.note}</p></section>}

        <div className="grid2">
          <section className="card">
            <div className="card-head"><h2>Recipe style</h2></div>
            <p className="muted small">Move up when the meals feel easy, back down when they feel hard. The twin also suggests this after check-ins.</p>
            <ol className="ladder">
              {STEP_WORDS.map((w, i) => (
                <li key={i} className={i === plan.style_step ? 'now' : i < plan.style_step ? 'done' : ''}>
                  <span className="ladder-dot" /><span className="grow">{w}</span>
                  {i === plan.style_step ? <span className="mark strong">Now</span>
                    : <button className="link small" disabled={busy || plan.status !== 'active'} onClick={() => setStep(i)}>Use this</button>}
                </li>))}
            </ol>
          </section>
          <section className="card">
            <div className="card-head"><h2>How the plan has changed</h2></div>
            <ol className="changes">
              {changes.length ? changes.map((c, i) => (
                <li key={i}><span className="muted small">{shortDate(c.start_date)}</span>
                  <span>{i === 0 ? 'Plan started' : 'Adjusted after a check-in'}: about {Math.round(-c.energy_change_kcal)} kcal a day from eating less and walking together</span></li>))
                : <li><span className="muted small">{shortDate(plan.start_date)}</span><span>Plan started</span></li>}
            </ol>
            <p className="muted small">Next check-in: {shortDate(plan.next_checkin)}.</p>
            <div className="row"><button className="btn" onClick={() => navigate('/choose-plan')}>Change my pace</button></div>
          </section>
        </div>
      </div>
    </Shell>
  )
}
