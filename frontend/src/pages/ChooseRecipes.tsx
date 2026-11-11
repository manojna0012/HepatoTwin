import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'
import { Icon } from '../components/Icon'
import { Shell } from '../components/Shell'
import { RecipeDrawer, RecipePhoto, kcal, minutes, useRecipes } from '../recipes'
import type { TwinRun } from '../twin'

interface Criterion { label?: string; scenario_value?: number; baseline_value?: number; compared_with?: string; direction: string }
interface Scenario { name: string; recipe_ids: string[]; titles: string[]; criteria: Criterion[] }

// Scenario B (step 0) and Scenario A (step 3) are the two choices; the twin's later nudges use the steps between.
const CHOICES = [
  { step: 0, key: 'Scenario B', title: 'Close to what you eat now', text: 'Built around your favourite dishes. A smaller change you keep beats a big one you drop.' },
  { step: 3, key: 'Scenario A', title: 'The healthiest', text: 'The recipes that best fit your nutrition targets, furthest from your usual food.' },
]
// Plain names and units for the nutrients the twin compares.
const SHORT: Record<string, [string, string]> = {
  'Energy per serving (kcal)': ['Energy per meal', ' kcal'],
  'Saturated fat (% of energy)': ['Saturated fat', '%'],
  'Total sugars (% of energy)': ['Sugars', '%'],
  'Fibre (g per 1,000 kcal)': ['Fibre', ' g'],
  'Glycaemic load per serving': ['Glycaemic load', ''],
}
const VERDICT: Record<string, string> = { improves: 'better', worsens: 'worse', 'no meaningful change': 'same' }

export function ChooseRecipes() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [scenarios, setScenarios] = useState<Scenario[] | null>(null)
  const [withheld, setWithheld] = useState(false)
  const [step, setStep] = useState(0)
  const [open, setOpen] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const details = useRecipes((scenarios ?? []).flatMap(s => s.recipe_ids ?? []))

  useEffect(() => {
    if (!user?.patient_id) return
    api<TwinRun>(`/patients/${user.patient_id}/twin`)
      .then(r => { setScenarios(r.twin.scenarios as Scenario[]); setWithheld(r.twin.recipes_withheld) })
      .catch(e => setError(e.message))
  }, [user?.patient_id])

  const confirm = async () => {
    if (!user?.patient_id) return
    setBusy(true); setError(null)
    try {
      await api(`/patients/${user.patient_id}/plan/style`, { method: 'PUT', body: { style_step: step } })
      navigate('/today')
    } catch (e) { setError((e as Error).message); setBusy(false) }
  }

  if (error && !scenarios) return <Shell title="Your recipes"><p className="error">{error}</p></Shell>
  if (!scenarios) return <Shell title="Your recipes"><div className="card narrow"><p className="muted">Finding recipes for you…</p></div></Shell>
  if (withheld || !scenarios.length) return (
    <Shell title="Your recipes"><div className="page narrow"><section className="notice"><p className="eyebrow blue">No recipe change needed</p>
      <p>{withheld ? 'Recipes are shown after a clinician has assessed your liver.' : 'Your numbers do not call for a recipe change. Keep eating as you do and check in as planned.'}</p>
      {!withheld && <button className="btn primary" onClick={() => navigate('/today')}>Go to Today</button>}</section></div></Shell>
  )

  return (
    <Shell title="Your recipes" subtitle="Which recipes would you like to start with? Both meet your health targets.">
      <div className="page">
        <div className="grid2" role="radiogroup" aria-label="Recipe style">
          {CHOICES.map(c => {
            const s = scenarios.find(x => x.name.startsWith(c.key))
            const chosen = step === c.step
            const compared = (s?.criteria ?? []).filter(x => x.label && SHORT[x.label] && x.scenario_value !== undefined && x.baseline_value !== undefined)
            const vsUsual = compared.some(x => x.compared_with === 'usual diet')
            return (
              <div key={c.step} className={`card style-card${chosen ? ' chosen' : ''}`} role="radio" aria-checked={chosen} tabIndex={0}
                   onClick={() => setStep(c.step)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setStep(c.step) } }}>
                <div className="style-head">
                  <div><span className="eyebrow">{c.key}</span><h2 className="choice-name">{c.title}</h2></div>
                  <span className={`pick${chosen ? ' on' : ''}`}>{chosen ? <><Icon name="check" /> Chosen</> : 'Choose'}</span>
                </div>
                <p className="muted small">{c.text}</p>
                {s && <ol className="dish-rows">
                  {(s.recipe_ids ?? []).map((id, i) => (
                    <li key={id}>
                      <RecipePhoto recipe={details[id]} size={52} />
                      <span className="dish-main"><strong>{details[id]?.title ?? s.titles[i]}</strong>
                        <span className="muted small">{[minutes(details[id]?.total_time), kcal(details[id])].filter(Boolean).join(' · ') || ' '}</span></span>
                      <button type="button" className="link small" onClick={e => { e.stopPropagation(); setOpen(id) }}>Recipe</button>
                    </li>
                  ))}
                </ol>}
                {compared.length > 0 && <div className="compare">
                  <p className="eyebrow">{vsUsual ? 'Compared with what you eat now' : 'Compared with the guideline'}</p>
                  {compared.map(x => {
                    const [name, unit] = SHORT[x.label!]
                    return (
                      <div key={x.label} className="compare-row">
                        <span>{name}</span>
                        <span className="muted num">{Math.round(x.baseline_value!)}{unit}</span>
                        <span className="muted">→</span>
                        <strong className="num">{Math.round(x.scenario_value!)}{unit}</strong>
                        <span className={`verdict ${x.direction === 'improves' ? 'good' : ''}`}>{VERDICT[x.direction] ?? ''}</span>
                      </div>
                    )
                  })}
                </div>}
              </div>
            )
          })}
        </div>
        <p className="muted small narrow">Once you settle in, the twin will suggest small steps towards the healthiest recipes, and smaller ones back if it gets hard.</p>
        {error && <p className="error">{error}</p>}
        <div className="row">
          <button className="btn" onClick={() => navigate('/liver')}>Back</button>
          <button className="btn primary" onClick={confirm} disabled={busy}>{busy ? 'Saving…' : `Start with ${step === 0 ? 'familiar' : 'the healthiest'} recipes`}</button>
        </div>
      </div>
      {open && <RecipeDrawer id={open} onClose={() => setOpen(null)} />}
    </Shell>
  )
}
