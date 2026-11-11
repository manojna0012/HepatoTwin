import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'
import { Choice, NumberField } from '../components/Form'
import { Shell } from '../components/Shell'
import { getPatient, latestMeasurements, longDate, today } from '../patient'
import { MEAL_WORDS, WALK_WORDS, verdictWords, type LoopVisit, type Tip } from '../progress'
import { pct } from '../twin'
import { TipCard } from './Today'

interface Changed {
  tip: Tip | null; check_in: LoopVisit | null; next_checkin: string | null
  chances_now: { chance_a_lot_of_fat_pct: number }; chances_before: { chance_a_lot_of_fat_pct: number } | null
}
const BLOODS: [string, string, string][] = [
  ['LBXSATSI', 'ALT', 'U/L'], ['LBXSASSI', 'AST', 'U/L'], ['LBXPLTSI', 'Platelets', '×1000/µL'], ['LBXGH', 'HbA1c', '%'],
]

export function CheckIn() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const pid = user?.patient_id
  const [last, setLast] = useState<Record<string, number>>({})
  const [weight, setWeight] = useState('')
  const [waist, setWaist] = useState('')
  const [meals, setMeals] = useState<keyof typeof MEAL_WORDS | null>(null)
  const [walking, setWalking] = useState<keyof typeof WALK_WORDS | null>(null)
  const [bloods, setBloods] = useState<Record<string, string>>({})
  const [showBloods, setShowBloods] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Changed | null>(null)

  useEffect(() => { if (pid) getPatient(pid).then(p => setLast(latestMeasurements(p.visits))).catch(() => {}) }, [pid])

  const send = async () => {
    const kg = Number(weight)
    if (!pid || !(kg >= 30 && kg <= 300) || !meals || !walking) { setError('Please give your weight and answer both questions.'); return }
    setBusy(true); setError(null)
    const measurements: Record<string, number> = { BMXWT: kg }
    if (last.BMXHT) measurements.BMXBMI = Math.round((kg / (last.BMXHT / 100) ** 2) * 10) / 10   // BMI follows the new weight
    if (Number(waist) > 0) measurements.BMXWAIST = Number(waist)
    for (const [k] of BLOODS) if (Number(bloods[k]) > 0) measurements[k] = Number(bloods[k])
    try {
      setResult(await api<Changed>(`/patients/${pid}/visits`, { method: 'POST', body: { visit_date: today(), measurements, followed: { meals, walking } } }))
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }

  if (result) {
    const v = result.check_in, before = result.chances_before?.chance_a_lot_of_fat_pct, now = result.chances_now.chance_a_lot_of_fat_pct
    return (
      <Shell title="Check-in saved" subtitle={longDate(today())}>
        <div className="page narrow">
          <section className="card">
            <span className="eyebrow">What changed</span>
            <div className="changed">
              {v && v.verdict !== 'start' && <div><span className="muted small">Your progress</span><strong>{verdictWords(v)}</strong>
                {v.expected_range_kg && <span className="muted small">Expected {v.expected_range_kg[0].toFixed(1)}–{v.expected_range_kg[1].toFixed(1)} kg, you weighed {v.weight_kg} kg</span>}</div>}
              <div><span className="muted small">Chance of a lot of liver fat</span>
                <strong>{before !== undefined && before !== now ? <>{pct(before)} → {pct(now)}</> : pct(now)}</strong>
                <span className="muted small">{before === undefined ? '' : now < before ? 'Lower than last time.' : now > before ? 'Higher than last time.' : 'The same as last time.'}</span></div>
              {result.next_checkin && <div><span className="muted small">Next check-in</span><strong>{longDate(result.next_checkin)}</strong></div>}
            </div>
          </section>
          {result.tip && <TipCard tip={result.tip} onDone={() => {}} />}
          <div className="row"><button className="btn primary" onClick={() => navigate('/today')}>Back to Today</button>
            <button className="btn" onClick={() => navigate('/progress')}>See my progress</button></div>
        </div>
      </Shell>
    )
  }

  return (
    <Shell title="Check-in" subtitle="Four quick answers. The twin compares them with what it expected and adjusts your plan.">
      <div className="page narrow">
        <section className="card">
          <NumberField label="Your weight today" unit="kg" value={weight} onChange={setWeight} min={30} max={300} step={0.1}
                       hint={last.BMXWT ? `Last time: ${last.BMXWT} kg` : undefined} />
          <Choice label="How much of the meal plan did you follow?" value={meals ?? undefined} onChange={v => setMeals(v)}
                  options={(Object.keys(MEAL_WORDS) as (keyof typeof MEAL_WORDS)[]).map(value => ({ value, label: MEAL_WORDS[value] }))} />
          <Choice label="How much did you walk?" value={walking ?? undefined} onChange={v => setWalking(v)}
                  options={(Object.keys(WALK_WORDS) as (keyof typeof WALK_WORDS)[]).map(value => ({ value, label: WALK_WORDS[value] }))} />
          <NumberField label="Waist (optional)" unit="cm" value={waist} onChange={setWaist} min={40} max={250} step={0.5}
                       hint={last.BMXWAIST ? `Last time: ${last.BMXWAIST} cm` : undefined} />
          {!showBloods ? <div><button className="link" onClick={() => setShowBloods(true)}>I have new blood test results</button></div> : (
            <div className="bloods">
              {BLOODS.map(([k, label, unit]) => <NumberField key={k} label={label} unit={unit} value={bloods[k] ?? ''}
                onChange={x => setBloods(b => ({ ...b, [k]: x }))} hint={last[k] ? `Last time: ${last[k]}` : undefined} />)}
            </div>)}
          {error && <p className="error">{error}</p>}
          <div className="row"><button className="btn" onClick={() => navigate(-1)}>Cancel</button>
            <button className="btn primary" onClick={send} disabled={busy}>{busy ? 'Saving…' : 'Save check-in'}</button></div>
        </section>
      </div>
    </Shell>
  )
}
