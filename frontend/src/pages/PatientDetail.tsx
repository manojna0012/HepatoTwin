import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { ScaleBar } from '../components/Charts'
import { Shell } from '../components/Shell'
import { ChartKey, WeightChart } from '../components/WeightChart'
import type { Plan, Visit } from '../patient'
import { MEAL_WORDS, PACE_WORDS, WALK_WORDS, shortDate, verdictWords, type Timeline } from '../progress'
import { RecipeDrawer, RecipePhoto, kcal, useRecipes } from '../recipes'
import { Review, type Row } from './Patients'

interface Driver { feature: string; label: string; value: number; contribution_db_m: number }
interface Evidence { evidence_id: string; evidence_strength: string; outcome: string; source: string }
interface Criterion { priority_id: string; label?: string; scenario_value?: number; baseline_value?: number; compared_with?: string; direction: string; evidence_strength?: string | null; evidence?: Evidence[]; conditional?: { condition: string } }
interface Scenario { name: string; recipe_ids: string[]; titles: string[]; projected_direction: string; evidence_strength: string | null; shift_from_usual: string | null; criteria: Criterion[]; timeline: { horizon: string; grade?: string; direction?: string; evidence_strength?: string }[] }
interface Reason { priority_id: string; priority: string; measure: string; value: number | null; reference: string; meets_reference: boolean; evidence: string[]; causal_path?: string; links?: string[] }
interface Clinical {
  patient: { id: number; label: string; cuisine: string | null; food: Record<string, unknown>; visits: Visit[] }
  plan: (Plan & { approved_by: number | null }) | null; plans: (Plan & { created_at: string })[]
  alerts: Row['alerts']; scarring: string | null; plan_status: string | null
  run: { visit_id: number; created_at: string; inputs_last_measured: Record<string, string> }
  twin: {
    current_state: { predicted_cap: number; predicted_grade: string; p_cap248: number; p_cap280: number }
    layer2: { thresholds: { cap248: number; cap280: number } }
    uncertainty: { cap_interval_90: [number, number]; probability_bands: { cap248: [number, number]; cap280: [number, number] } }
    cap_explanation: { average_cap: number; predicted_cap: number; raising: Driver[]; lowering: Driver[]; note: string }
    clinical: {
      fibrosis: { method: string; fib4: number | null; liver_stiffness_kpa: number | null; category: string; message: string; source?: string }
      liver_label: { label: string; criteria: Record<string, boolean>; alcohol_g_per_week: number | null; alcohol_band: string; note: string }
      guardrails: { notices: { id: string; message: string }[]; block_recipes: boolean }
    }
    nutritional_priorities: { id: string; title: string; priority: string; reason: string; evidence_ids: string[]; requires_clinical_review: boolean }[]
    scenarios: Scenario[]
    recipe_explanations: { recipe_id: string; title: string; rank: number; reasons: Reason[] }[]
    summary: { text: string; written_by: string; check: { faithful: boolean } }
    recipes_withheld: boolean
    top_recipes: { recipe_id: string; title: string; cuisine: string; overall: number; rank: number }[]
  }
  timeline: Timeline & { visits: (Timeline['visits'][number] & { predicted_cap?: number })[] }
}

const CRITERIA_WORDS: Record<string, string> = { body_size: 'BMI or waist', glycaemia: 'Glucose or HbA1c', blood_pressure: 'Blood pressure', triglycerides: 'Triglycerides', low_hdl: 'Low HDL' }
const pc = (p: number) => `${Math.round(p * 100)}%`
const r1 = (x: number) => Math.round(x * 10) / 10

/** What raised or lowered the CAP estimate from the average (SHAP contributions, dB/m), as bars from a centre line. */
function Drivers({ e }: { e: Clinical['twin']['cap_explanation'] }) {
  const rows = [...e.raising, ...e.lowering].sort((a, b) => b.contribution_db_m - a.contribution_db_m)
  const max = Math.max(...rows.map(r => Math.abs(r.contribution_db_m)), 1)
  return (
    <div className="drivers">
      {rows.map(r => {
        const w = `${(Math.abs(r.contribution_db_m) / max) * 50}%`, up = r.contribution_db_m > 0
        return (
          <div key={r.feature} className="driver">
            <span className="driver-label">{r.label}<span className="muted"> {r1(r.value)}</span></span>
            <span className="driver-track"><i className={up ? 'up' : 'down'} style={up ? { left: '50%', width: w } : { right: '50%', width: w }} /></span>
            <span className={`driver-value ${up ? 'up' : ''}`}>{up ? '+' : '−'}{r1(Math.abs(r.contribution_db_m))}</span>
          </div>)
      })}
      <div className="driver driver-axis"><span /><span className="driver-track"><em>lowers</em><em>raises</em></span><span /></div>
    </div>
  )
}

function ScenarioCard({ s, details, onOpen }: { s: Scenario; details: ReturnType<typeof useRecipes>; onOpen: (id: string) => void }) {
  return (
    <section className="card">
      <div className="card-head"><div><span className="eyebrow">{s.name.split(' - ')[0]}</span><h2>{s.name.split(' - ')[1] ?? s.name}</h2></div>
        {s.evidence_strength && <span className="mark">{s.evidence_strength.toLowerCase()} evidence</span>}</div>
      <p className="small">Projected: <strong>{s.projected_direction}</strong>{s.shift_from_usual && <> · a {s.shift_from_usual} change from usual food</>}</p>
      <ol className="dish-rows">{s.recipe_ids.map((id, i) => (
        <li key={id}><RecipePhoto recipe={details[id]} size={44} />
          <span className="dish-main"><strong>{details[id]?.title ?? s.titles[i]}</strong><span className="muted small">{kcal(details[id])}</span></span>
          <button className="link small" onClick={() => onOpen(id)}>Recipe</button></li>))}</ol>
      <div className="tbl-wrap"><table className="tbl compact">
        <thead><tr><th>Measure</th><th className="r">Usual</th><th className="r">Scenario</th><th>Direction</th><th>Evidence</th></tr></thead>
        <tbody>{s.criteria.filter(c => c.label).map(c => (
          <tr key={c.priority_id}><td>{c.label}{c.compared_with === 'reference value' && <div className="muted small">vs reference</div>}</td>
            <td className="r">{c.baseline_value ?? '–'}</td><td className="r"><strong>{c.scenario_value ?? '–'}</strong></td>
            <td>{c.direction}{c.conditional && <div className="muted small">{c.conditional.condition}</div>}</td>
            <td className="small">{(c.evidence ?? []).map(e => <div key={e.evidence_id} title={`${e.outcome}. ${e.source}`}>{e.evidence_id} <span className="muted">{e.evidence_strength.toLowerCase()}</span></div>)}</td></tr>))}</tbody>
      </table></div>
      <p className="muted small">{s.timeline.filter(t => t.direction).map(t => `${t.horizon}: ${t.direction}`).join(' · ')}. Simulated projection, not a clinical forecast.</p>
    </section>
  )
}

export function PatientDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState<Clinical | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [reviewing, setReviewing] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const load = useCallback(() => { api<Clinical>(`/patients/${id}/clinical`).then(setData).catch(e => setError((e as Error).message)) }, [id])
  useEffect(load, [load])
  const t = data?.twin
  const details = useRecipes((t?.scenarios ?? []).flatMap(s => s.recipe_ids))

  if (error) return <Shell title="Patient"><p className="error">{error}</p></Shell>
  if (!data || !t) return <Shell title="Patient" crumb="Patients"><div className="card narrow"><p className="muted">Loading the patient…</p></div></Shell>

  const p = data.patient, cs = t.current_state, u = t.uncertainty, fib = t.clinical.fibrosis, lab = t.clinical.liver_label
  const plan = data.plan
  const lastVisit = p.visits[p.visits.length - 1]
  const loop = data.timeline.loop?.visits ?? []
  const row: Row = { id: p.id, label: p.label, cuisine: p.cuisine, alerts: data.alerts, chance_a_lot_of_fat_pct: Math.round(cs.p_cap280 * 100), scarring: data.scarring, last_verdict: null, plan_status: data.plan_status }

  return (
    <Shell title={p.label} crumb={`Patients › ${p.label}`}
           subtitle={[p.cuisine && `${p.cuisine} cuisine`, `latest visit ${shortDate(lastVisit.visit_date)}`, `plan ${(data.plan_status ?? 'not chosen').replace('_', ' ')}`].filter(Boolean).join(' · ')}
           action={<button className="btn" onClick={() => navigate('/patients')}>All patients</button>}>
      <div className="page">
        {data.alerts.length > 0 && <section className="card alerts-card">
          <div className="card-head"><h2>Needs you</h2>{data.plan_status === 'awaiting_approval' && <button className="btn primary" onClick={() => setReviewing(!reviewing)}>{reviewing ? 'Close review' : 'Review plan'}</button>}</div>
          <ul className="alert-list">{data.alerts.map(a => <li key={a.id}><span className={`mark ${a.severity <= 2 ? 'strong' : ''}`}>{a.severity <= 2 ? 'Act' : 'Note'}</span>{a.message}
            {a.checks && <ul className="checks">{a.checks.map(c => <li key={c}>{c}</li>)}</ul>}</li>)}</ul>
          {reviewing && <Review row={row} onDone={() => { setReviewing(false); load() }} />}
        </section>}

        <div className="grid3">
          <section className="card">
            <div className="card-head"><h2>Liver fat estimate</h2><span className="mark strong">{cs.predicted_grade}</span></div>
            <div><span className="big-num">{Math.round(cs.predicted_cap)}</span> <span className="muted">dB/m predicted CAP</span></div>
            <p className="muted small">90% interval {Math.round(u.cap_interval_90[0])}–{Math.round(u.cap_interval_90[1])} dB/m</p>
            {/* the scale runs from 200 to 360 dB/m so the narrow S1 and S2 bands can be read */}
            <ScaleBar value={Math.min(Math.max(cs.predicted_cap, 200), 360) - 200} max={160} format={v => String(Math.round(v + 200))}
                      bands={[{ upTo: 48, label: 'S0' }, { upTo: 68, label: 'S1' }, { upTo: 80, label: 'S2' }, { upTo: 160, label: 'S3' }]} />
            <dl className="kv">
              <div><dt>P(CAP ≥ 248)</dt><dd><strong>{pc(cs.p_cap248)}</strong> <span className="muted">({pc(u.probability_bands.cap248[0])}–{pc(u.probability_bands.cap248[1])}; threshold {t.layer2.thresholds.cap248.toFixed(2)})</span></dd></div>
              <div><dt>P(CAP ≥ 280)</dt><dd><strong>{pc(cs.p_cap280)}</strong> <span className="muted">({pc(u.probability_bands.cap280[0])}–{pc(u.probability_bands.cap280[1])}; threshold {t.layer2.thresholds.cap280.toFixed(2)})</span></dd></div>
            </dl>
            <p className="muted small">Grades by Karlas et al. 2017 cut-offs. The point estimate explains about 56% of CAP variance on held-out NHANES data, so the calibrated chances are what the patient sees.</p>
          </section>
          <section className="card">
            <div className="card-head"><h2>Scarring</h2>{!['not_applicable', 'unknown'].includes(fib.category) && <span className="mark">{fib.category} risk</span>}</div>
            {fib.method === 'fib4' && fib.fib4 !== null ? <>
              <div><span className="big-num">{fib.fib4.toFixed(2)}</span> <span className="muted">FIB-4</span></div>
              <ScaleBar value={fib.fib4} max={4} format={v => v.toFixed(2)} bands={[{ upTo: 1.3, label: 'Low' }, { upTo: 2.67, label: 'Indeterminate' }, { upTo: 4, label: 'High' }]} />
            </> : fib.liver_stiffness_kpa !== null ? <div><span className="big-num">{fib.liver_stiffness_kpa}</span> <span className="muted">kPa stiffness</span></div> : null}
            <p className="small">{fib.message}</p>
            {fib.source && <p className="muted small">{fib.source}</p>}
          </section>
          <section className="card">
            <div className="card-head"><h2>Label and safety</h2><span className="mark">{lab.label.replace(/_/g, ' ')}</span></div>
            <ul className="criteria">{Object.entries(lab.criteria).map(([k, v]) => <li key={k} className={v ? 'met' : ''}><i>{v ? '✓' : '–'}</i>{CRITERIA_WORDS[k] ?? k}</li>)}</ul>
            {lab.alcohol_g_per_week !== null && <p className="small">Alcohol about {Math.round(lab.alcohol_g_per_week)} g a week ({lab.alcohol_band.replace(/_/g, ' ')}).</p>}
            {t.clinical.guardrails.notices.map(n => <p key={n.id} className="notice-line"><strong>Safety notice:</strong> {n.message}</p>)}
            {t.recipes_withheld && <p className="notice-line"><strong>Recipes withheld</strong> until the liver has been assessed.</p>}
            <p className="muted small">{lab.note}</p>
          </section>
        </div>

        <section className="card">
          <div className="card-head"><h2>What moved the estimate</h2>
            <span className="muted small">average {Math.round(t.cap_explanation.average_cap)} → this patient {Math.round(t.cap_explanation.predicted_cap)} dB/m</span></div>
          <Drivers e={t.cap_explanation} />
          <p className="muted small">SHAP contributions in dB/m. {t.cap_explanation.note}</p>
        </section>

        <div className="grid23">
          <section className="card">
            <div className="card-head"><h2>Weight and plan</h2></div>
            <WeightChart timeline={data.timeline} height={220} />
            <ChartKey expected={loop.some(v => v.expected_range_kg)} />
          </section>
          <section className="card">
            <div className="card-head"><h2>Visits</h2></div>
            <div className="tbl-wrap"><table className="tbl compact">
              <thead><tr><th>Date</th><th className="r">Weight</th><th className="r">CAP est.</th><th className="r">P(≥280)</th><th>Followed</th><th>Verdict</th></tr></thead>
              <tbody>{[...data.timeline.visits].reverse().map(v => {
                const l = loop.find(x => x.visit_date === v.visit_date)
                return <tr key={v.visit_id}><td>{shortDate(v.visit_date)}</td><td className="r">{v.measured.BMXWT ?? '–'}</td>
                  <td className="r">{v.predicted_cap !== undefined ? Math.round(v.predicted_cap) : '–'}</td><td className="r">{v.chance_a_lot_of_fat_pct}%</td>
                  <td className="small">{v.followed?.meals ? `${MEAL_WORDS[v.followed.meals]} · ${WALK_WORDS[v.followed.walking ?? 'some'].toLowerCase()}` : '–'}</td>
                  <td className="small">{l ? verdictWords(l) : '–'}</td></tr>
              })}</tbody></table></div>
          </section>
        </div>

        <section className="card">
          <div className="card-head"><h2>Nutritional priorities</h2><span className="muted small">candidates for discussion, not prescriptions</span></div>
          <div className="tbl-wrap"><table className="tbl">
            <thead><tr><th>Priority</th><th>Focus</th><th>Why</th><th>Evidence</th></tr></thead>
            <tbody>{t.nutritional_priorities.map(n => <tr key={n.id}><td><span className={`mark ${n.priority === 'high' ? 'strong' : ''}`}>{n.priority}</span></td>
              <td><strong>{n.title}</strong>{n.requires_clinical_review && <div className="muted small">needs clinical review</div>}</td>
              <td className="small">{n.reason}</td><td className="small">{n.evidence_ids.join(', ')}</td></tr>)}</tbody>
          </table></div>
        </section>

        {t.scenarios.length === 2 && <div className="grid2">{t.scenarios.map(s => <ScenarioCard key={s.name} s={s} details={details} onOpen={setOpen} />)}</div>}

        {t.recipe_explanations.length > 0 && <section className="card">
          <div className="card-head"><h2>Why the top recipes were chosen</h2><span className="muted small">knowledge-graph paths and evidence</span></div>
          <div className="explain-grid">{t.recipe_explanations.map(r => (
            <div key={r.recipe_id} className="explain">
              <div className="row"><span className="mark">#{r.rank}</span><button className="link" onClick={() => setOpen(r.recipe_id)}>{r.title}</button></div>
              <ul>{r.reasons.map(x => <li key={x.priority_id}>
                <span>{x.measure}: <strong>{x.value ?? '–'}</strong> <span className="muted">({x.reference.split(' (')[0]})</span> {x.meets_reference ? '✓' : '✗'}</span>
                {x.causal_path && <span className="path">{x.causal_path.split(' -> ').join(' → ')}</span>}
                <span className="muted small">{x.evidence.slice(0, 2).join(' · ')}{x.evidence.length > 2 && ` · +${x.evidence.length - 2} more`}</span></li>)}</ul>
            </div>))}</div>
        </section>}

        <div className="grid2">
          <section className="card">
            <div className="card-head"><h2>Plan</h2></div>
            {plan ? <dl className="kv">
              <div><dt>Status</dt><dd>{plan.status.replace('_', ' ')} · set by {plan.set_by}</dd></div>
              <div><dt>Pace</dt><dd>{plan.pace ? PACE_WORDS[plan.pace] ?? plan.pace : 'custom'} · {Math.round(-plan.energy_change_kcal)} kcal less a day · {Math.round(plan.walking_min_per_week)} min walking a week</dd></div>
              <div><dt>Recipe style</dt><dd>step {plan.style_step} of 3</dd></div>
              <div><dt>Started</dt><dd>{shortDate(plan.start_date)} · next check-in {shortDate(plan.next_checkin)}</dd></div>
              {plan.note && <div><dt>Note</dt><dd>{plan.note}</dd></div>}
            </dl> : <p className="muted small">No plan yet.</p>}
            {data.plans.length > 1 && <><h3>Earlier plans</h3><ul className="changes">{data.plans.slice(0, -1).reverse().map(x => <li key={x.id}><span className="muted small">{shortDate(x.start_date)}</span><span>{x.pace ?? 'custom'} · {Math.round(-x.energy_change_kcal)} kcal · {x.status.replace('_', ' ')}</span></li>)}</ul></>}
          </section>
          <section className="card">
            <div className="card-head"><h2>What the patient reads</h2>{t.summary.check.faithful && <span className="mark">checked against the twin</span>}</div>
            <p className="small summary-text">{t.summary.text}</p>
          </section>
        </div>
      </div>
      {open && <RecipeDrawer id={open} onClose={() => setOpen(null)} />}
    </Shell>
  )
}
