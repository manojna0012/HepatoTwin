import { Fragment, useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { Shell } from '../components/Shell'
import { getPatient, today, type PaceOption, type PatientRecord, type Plan, type Suggestion } from '../patient'

interface Alert { id: string; message: string; severity: number; checks?: string[] }
export interface Row {
  id: number; label: string; cuisine: string | null; alerts: Alert[]; chance_a_lot_of_fat_pct: number | null
  scarring: string | null; last_verdict: string | null; plan_status: string | null
}

const ORDER = ['gentle', 'standard', 'ambitious']

/** Review of a plan that waits for approval: keep the patient's pace or pick another, with a note. */
export function Review({ row, onDone }: { row: Row; onDone: () => void }) {
  const [patient, setPatient] = useState<PatientRecord | null>(null)
  const [options, setOptions] = useState<PaceOption[]>([])
  const [pace, setPace] = useState<string>('')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getPatient(row.id).then(p => { setPatient(p); setPace(p.plan?.pace ?? 'gentle') }).catch(e => setError(e.message))
    api<Suggestion>(`/patients/${row.id}/plan/suggestion`).then(s => setOptions([...s.options].sort((a, b) => ORDER.indexOf(a.pace) - ORDER.indexOf(b.pace))))
      .catch(e => setError(e.message))
  }, [row.id])

  const approve = async () => {
    if (!patient?.plan) return
    setBusy(true); setError(null)
    try {
      if (pace === patient.plan.pace && !note.trim()) await api<Plan>(`/patients/${row.id}/plan/approve`, { method: 'POST' })
      else await api<Plan>(`/patients/${row.id}/plan`, { method: 'PUT', body: {
        start_date: patient.plan.start_date ?? today(), pace, style_step: patient.plan.style_step, note: note.trim() || undefined } })
      onDone()
    } catch (e) { setError((e as Error).message); setBusy(false) }
  }

  if (!patient?.plan) return <p className="muted small">Loading…</p>
  const asked = patient.plan
  const reason = row.alerts.find(a => a.id === 'awaiting_approval')?.message
  return (
    <div className="review">
      <p><strong>{row.label}</strong> asked for the <strong>{asked.pace}</strong> pace: {Math.abs(Math.round(asked.energy_change_kcal))} kcal less a day
        and {asked.walking_min_per_week} minutes of walking a week. {reason && <span className="muted">{reason}</span>}</p>
      <div className="pace-grid compact" role="radiogroup" aria-label="Pace to approve">
        {options.map(o => (
          <button key={o.pace} type="button" className="card pace-card" role="radio" aria-checked={pace === o.pace} onClick={() => setPace(o.pace)}>
            <span className="pace-top"><strong className="pace-name">{o.pace}</strong>{o.pace === asked.pace && <span className="mark">asked for</span>}<i className="radio" /></span>
            <span className="small">{Math.abs(o.diet_kcal)} kcal · {o.walking_min_per_week} min a week</span>
            <span className="muted small">about {o.expected_change_12m_kg} kg in a year</span>
          </button>
        ))}
      </div>
      <label className="field"><span>Note to the patient <small className="muted">(optional)</small></span>
        <input className="input" value={note} onChange={e => setNote(e.target.value)} placeholder="For example: Gentler start because of diabetes pills. Review at 3 months." />
      </label>
      {error && <p className="error">{error}</p>}
      <div className="row"><button className="btn primary" onClick={approve} disabled={busy || !pace}>{busy ? 'Saving…' : `Approve the ${pace} pace`}</button></div>
    </div>
  )
}

export function Patients() {
  const [rows, setRows] = useState<Row[] | null>(null)
  const [open, setOpen] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()
  const load = useCallback(() => { api<Row[]>('/patients').then(setRows).catch(e => setError(e.message)) }, [])
  useEffect(load, [load])

  const need = rows?.filter(r => r.alerts.some(a => a.severity <= 2)).length ?? 0
  return (
    <Shell title="Patients" alerts={need} subtitle={rows ? (need ? `${need} of ${rows.length} patients need you` : 'No one needs you today') : undefined}>
      <div className="page">
        {error && <p className="error">{error}</p>}
        {!rows ? <div className="card"><p className="muted">Loading…</p></div> : (
          <section className="card">
            <div className="scroll">
              <table className="tbl">
                <thead><tr><th>Patient</th><th>Cuisine</th><th className="r">Chance of a lot of fat</th><th>Scarring</th><th>Plan</th><th>Needs</th></tr></thead>
                <tbody>
                  {rows.map(r => (
                    <Fragment key={r.id}>
                      <tr className="link-row" onClick={e => { if (!(e.target as HTMLElement).closest('button')) navigate(`/patients/${r.id}`) }}>
                        <td><strong>{r.label}</strong> <span className="link small">Open</span></td>
                        <td>{r.cuisine ?? '–'}</td>
                        <td className="r">{r.chance_a_lot_of_fat_pct === null ? '–' : `${r.chance_a_lot_of_fat_pct}%`}</td>
                        <td>{r.scarring ? <span className="mark">{r.scarring}</span> : '–'}</td>
                        <td>{(r.plan_status ?? 'none').replace('_', ' ')}</td>
                        <td>
                          {r.alerts.length === 0 ? <span className="muted">nothing</span> : r.alerts.map(a => (
                            <div key={a.id} className="alert-line">
                              <span className={`mark ${a.severity <= 2 ? 'strong' : ''}`}>{a.message}</span>
                              {a.checks && <ul className="checks">{a.checks.map(c => <li key={c}>{c}</li>)}</ul>}
                            </div>
                          ))}
                          {r.plan_status === 'awaiting_approval' && (
                            <button className="link small" onClick={() => setOpen(open === r.id ? null : r.id)}>{open === r.id ? 'Close review' : 'Review plan'}</button>
                          )}
                        </td>
                      </tr>
                      {open === r.id && (
                        <tr><td colSpan={6}><Review row={r} onDone={() => { setOpen(null); load() }} /></td></tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small muted">Click a patient for their full page: CAP estimate and grade, what drove it, scarring, both recipe scenarios and the evidence.</p>
          </section>
        )}
      </div>
    </Shell>
  )
}
