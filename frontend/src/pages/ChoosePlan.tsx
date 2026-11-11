import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'
import { Shell } from '../components/Shell'
import { NOTICE_WORDS, today, type PaceOption, type Plan, type Suggestion } from '../patient'

const ORDER = ['gentle', 'standard', 'ambitious']
const PACE_WORDS: Record<string, string> = {
  gentle: 'A smaller change to start with',
  standard: 'The Diabetes Prevention Program plan',
  ambitious: 'The Look AHEAD plan, for faster results',
}

export function ChoosePlan() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null)
  const [pace, setPace] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!user?.patient_id) return
    api<Suggestion>(`/patients/${user.patient_id}/plan/suggestion`)
      .then(s => { setSuggestion(s); setPace(s.options.find(o => o.recommended)?.pace ?? s.options[0]?.pace ?? null) })
      .catch(e => setError(e.message))
  }, [user?.patient_id])

  const choose = async () => {
    if (!pace || !user?.patient_id) return
    setBusy(true); setError(null)
    try {
      const plan = await api<Plan>(`/patients/${user.patient_id}/plan`, { method: 'PUT', body: { start_date: today(), pace, style_step: 0 } })
      navigate(plan.status === 'awaiting_approval' ? '/waiting' : '/choose-recipes')
    } catch (e) { setError((e as Error).message); setBusy(false) }
  }

  if (error && !suggestion) return <Shell title="Your plan"><p className="error">{error}</p></Shell>
  if (!suggestion) return <Shell title="Your plan"><div className="card narrow"><p className="muted">Loading…</p></div></Shell>
  const options = [...suggestion.options].sort((a, b) => ORDER.indexOf(a.pace) - ORDER.indexOf(b.pace))
  const approval = suggestion.approval_needed_for

  if (options.length === 0) return (
    <Shell title="Your plan">
      <div className="page narrow">
        <section className="notice"><p className="eyebrow blue">Your clinician sets your plan</p><p>{suggestion.note}</p></section>
        <div className="row"><button className="btn" onClick={() => navigate('/liver')}>Back to my results</button></div>
      </div>
    </Shell>
  )

  return (
    <Shell title="Your plan" subtitle="Pick a pace that feels doable. You never have to count calories yourself.">
      <div className="page">
        {approval.length > 0 && (
          <section className="notice">
            <p className="eyebrow blue">Your clinician checks it first</p>
            <p>Because {approval.map(a => NOTICE_WORDS[a] ?? a.replace(/_/g, ' ')).join(' and ')}, your clinician will look at your plan before it starts, usually within a day.</p>
          </section>
        )}
        <div className="pace-grid" role="radiogroup" aria-label="Pace">
          {options.map((o: PaceOption) => (
            <button key={o.pace} type="button" className="card pace-card" role="radio" aria-checked={pace === o.pace} onClick={() => setPace(o.pace)}>
              <span className="pace-top"><strong className="pace-name">{o.pace}</strong>{o.recommended && <span className="mark strong">suggested</span>}<i className="radio" /></span>
              <span className="muted small">{PACE_WORDS[o.pace]}</span>
              <span className="pace-figures">
                <span><b>{Math.abs(o.diet_kcal)}</b> kcal less a day</span>
                <span><b>{o.walking_min_per_week}</b> min walking a week</span>
              </span>
              <span className="pace-outcome">
                <span className="metric">{o.expected_change_12m_kg} kg</span>
                <span className="muted small">what people like you usually lose in a year ({o.if_followed_change_12m_kg} kg if followed fully)</span>
              </span>
            </button>
          ))}
        </div>
        <p className="small muted">{suggestion.note} Paces come from the Diabetes Prevention Program and Look AHEAD trials, and the weight change from Hall's energy-balance rule.</p>
        {error && <p className="error">{error}</p>}
        <div className="row">
          <button className="btn" onClick={() => navigate('/liver')}>Back</button>
          <button className="btn primary" onClick={choose} disabled={!pace || busy}>{busy ? 'Saving…' : `Choose the ${pace} pace`}</button>
        </div>
      </div>
    </Shell>
  )
}
