import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'
import { ScaleBar } from '../components/Charts'
import { Shell } from '../components/Shell'
import { getPatient, latestMeasurements, type PatientRecord } from '../patient'
import type { Timeline } from '../progress'
import { LiverView } from './LiverView'
import { pct, plainPriority, type TwinRun } from '../twin'


// published cut-offs, as used by the twin (src/recipe_scoring/patient_state.py)
function numbers(m: Record<string, number>) {
  const man = m.RIAGENDR === 1, asian = m.RIDRETH3 === 6
  const rows: { label: string; value: string; note?: string }[] = []
  const add = (label: string, v: number | undefined, unit: string, note?: string | false) => {
    if (v !== undefined) rows.push({ label, value: `${+v.toFixed(1)} ${unit}`.trim(), note: note || undefined })
  }
  add('Weight', m.BMXWT, 'kg')
  add('BMI', m.BMXBMI, '', m.BMXBMI >= (asian ? 27.5 : 30) ? 'obesity range' : m.BMXBMI >= (asian ? 23 : 25) && 'overweight range')
  add('Waist', m.BMXWAIST, 'cm', m.BMXWAIST > (man ? 102 : 88) && 'above the usual cut-off')
  add('HbA1c', m.LBXGH, '%', m.LBXGH >= 5.7 && 'raised')
  add('Triglycerides', m.LBXSTR, 'mg/dL', m.LBXSTR >= 175 && 'raised')
  add('HDL cholesterol', m.LBDHDD, 'mg/dL', m.LBDHDD < (man ? 40 : 50) && 'low')
  add('ALT', m.LBXSATSI, 'U/L')
  add('AST', m.LBXSASSI, 'U/L')
  return rows
}

const LEVEL_ORDER: Record<string, number> = { high: 0, medium: 1, low: 2 }

export function Results() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [run, setRun] = useState<TwinRun | null>(null)
  const [patient, setPatient] = useState<PatientRecord | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [timeline, setTimeline] = useState<Timeline | null>(null)

  useEffect(() => {
    if (!user?.patient_id) return
    getPatient(user.patient_id).then(setPatient).catch(e => setError(e.message))
    api<TwinRun>(`/patients/${user.patient_id}/twin`).then(setRun).catch(e => setError(e.message))   // the saved run: instant
    api<Timeline>(`/patients/${user.patient_id}/timeline`).then(setTimeline).catch(() => {})       // for the 12-month projection
  }, [user?.patient_id])

  if (error) return <Shell title="My liver" crumb="My liver"><p className="error">{error}</p></Shell>
  if (!run || !patient) return <Shell title="My liver" crumb="My liver"><div className="card narrow"><p className="muted">Working out your results…</p></div></Shell>
  const t = run.twin
  const fib = t.fibrosis
  const m = latestMeasurements(patient.visits)
  const plan = patient.plan
  const priorities = [...t.priorities].sort((a, b) => (LEVEL_ORDER[a.priority] ?? 3) - (LEVEL_ORDER[b.priority] ?? 3))
  const scan = fib.method === 'fibroscan'
  const score = scan ? fib.liver_stiffness_kpa : fib.fib4
  const shown = (v: number) => (scan ? v.toFixed(1) : v.toFixed(2))
  const fatLikely = t.chance_some_fat_pct >= 43

  return (
    <Shell title="My liver" crumb="My liver" subtitle={`From your measurements on ${new Date(run.visit_date + 'T12:00').toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}`}>
      <div className="page">
        <LiverView twin={t} timeline={timeline} />

        <section className="card">
          <h2>At a glance</h2>
          <dl className="glance">
            <dt>Liver fat</dt>
            <dd><strong>{t.chance_a_lot_of_fat_pct >= 37 ? 'Likely a lot' : t.chance_some_fat_pct >= 43 ? 'Likely some' : 'Little sign'}</strong>
              <span className="muted"> · {pct(t.chance_a_lot_of_fat_pct)} chance of a lot</span></dd>
            {t.liver_label && <><dt>Pattern</dt><dd>{t.liver_label === 'no_steatosis_indicated' ? 'No sign of fatty liver' : t.liver_label}
              <span className="muted"> · research label, not a diagnosis</span></dd></>}
            <dt>Scarring</dt>
            <dd>{score === null || fib.category === 'not_applicable' || fib.category === 'unknown' ? <span className="muted">Not worked out</span>
              : <><strong>{fib.category[0].toUpperCase() + fib.category.slice(1)} risk</strong><span className="muted"> · {scan ? 'stiffness' : 'FIB-4'} {shown(score)}</span></>}</dd>
            {t.notices.length > 0 && <><dt>Before you start</dt><dd>Talk to your clinician before changing your diet</dd></>}
            <dt>Focus on</dt>
            <dd>{priorities.length ? priorities.slice(0, 3).map(p => plainPriority(p.title).split(',')[0].split(' (')[0]).join(' · ') : <span className="muted">No diet change needed</span>}</dd>
            {!t.recipes_withheld && t.recipes.length > 0 && <><dt>Recipes to try</dt><dd>{t.recipes.slice(0, 3).map(x => x.title).join(' · ')}</dd></>}
          </dl>
        </section>

        {t.notices.map(n => (
          <section key={n} className="notice"><p className="eyebrow blue">Before you start</p><p>{n}</p></section>
        ))}

        <div className="grid2">
          <section className="card">
            <div className="card-head"><h2>Liver scarring check</h2>{score !== null && fib.category !== 'not_applicable' && <span className="mark">{fib.category} risk</span>}</div>
            <p className="muted small">Liver scarring (fibrosis) is long-term damage that makes the liver stiffer.</p>
            {score === null || fib.category === 'not_applicable' || fib.category === 'unknown' ? <p>{fib.message}</p> : <>
              <p><span className="metric">{shown(score)}</span> <span className="muted">{scan ? 'kPa, FibroScan liver stiffness' : 'scarring score (FIB-4)'}</span></p>
              <ScaleBar value={score} max={scan ? 20 : 4} format={shown}
                        bands={scan ? [{ upTo: 8, label: 'Low' }, { upTo: 12, label: 'Indeterminate' }, { upTo: 20, label: 'High' }]
                                    : [{ upTo: 1.3, label: 'Low' }, { upTo: 2.67, label: 'Indeterminate' }, { upTo: 4, label: 'High' }]} />
              <p className="small muted">Cut-offs from the EASL-EASD-EASO 2024 guideline. {fib.message}</p>
              {fib.note && fatLikely && <p className="small muted">{fib.note}</p>}
            </>}
          </section>

          <section className="card">
            <h2>Your numbers</h2>
            <ul className="plain-list numbers">
              {numbers(m).map(r => (
                <li key={r.label}><span>{r.label}</span><span><strong>{r.value}</strong>{r.note && <span className="mark tag">{r.note}</span>}</span></li>
              ))}
            </ul>
          </section>
        </div>

        <section className="card">
          <div className="card-head"><h2>What matters most for you</h2><span className="muted small">From your results and the evidence</span></div>
          {priorities.length === 0 ? <p className="muted">No diet change is needed for your liver right now.</p> : (
            <div className="priority-grid">
              {priorities.map(p => (
                <div key={p.title} className={`priority level-${p.priority}`}>
                  <span className="level"><i />{p.priority[0].toUpperCase() + p.priority.slice(1)} priority</span>
                  <strong>{plainPriority(p.title)}</strong>
                  <span className="muted small">{p.evidence.length ? `Backed by ${p.evidence.length} ${p.evidence.length === 1 ? 'study' : 'studies'}` : 'Guideline advice'}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        {t.recipes_withheld ? (
          <section className="notice"><p className="eyebrow blue">Recipes paused</p><p>Recipes are shown after a clinician has assessed your liver.</p></section>
        ) : t.recipes.length > 0 && (
          <section className="card">
            <div className="card-head"><h2>Recipes that fit you</h2><span className="muted small">Each one meets every nutrition target for you</span></div>
            <div className="recipe-cards">
              {t.recipes.slice(0, 3).map(r => (
                <div key={r.recipe_id} className="recipe-card">
                  <strong>{r.title}</strong>
                  <ul className="plain-list small">
                    {r.reasons.slice(0, 3).map(x => (
                      <li key={x.priority_id}><span className="muted">{x.measure.replace(/ \(.*\)/, '')}</span><span>{x.value} <span className="muted">/ {x.reference.split(' (')[0]}</span></span></li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        )}


        <section className="card next-card">
          {plan ? <>
            <div><h2>{plan.status === 'awaiting_approval' ? 'Your plan is waiting for your clinician' : 'Your plan is set'}</h2>
              <p className="muted">{Math.abs(Math.round(plan.energy_change_kcal))} kcal less a day and {plan.walking_min_per_week} minutes of walking a week.</p></div>
            <button className="btn primary" onClick={() => navigate(plan.status === 'awaiting_approval' ? '/waiting' : '/choose-recipes')}>
              {plan.status === 'awaiting_approval' ? 'See where it is' : 'Choose my recipes'}</button>
          </> : <>
            <div><h2>Next: choose your plan</h2>
              <p className="muted">{priorities.length ? 'Pick a pace that feels doable; the twin works out the calories for you.' : 'Your liver looks healthy; you can still set a plan, for example to manage your weight.'}</p></div>
            <button className="btn primary" onClick={() => navigate('/choose-plan')}>Choose my plan</button>
          </>}
        </section>
      </div>
    </Shell>
  )
}
