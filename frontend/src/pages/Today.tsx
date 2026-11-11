import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'
import { GoalRing, LiverFigure } from '../components/Charts'
import { Icon } from '../components/Icon'
import { Shell } from '../components/Shell'
import { getPatient, latestMeasurements, longDate, today } from '../patient'
import { WALK_WORDS, daysUntil, usePatientGet, verdictWords, type Timeline, type Tip, type TodayData } from '../progress'
import { RecipeDrawer, RecipePhoto, kcal, minutes, useRecipes } from '../recipes'
import { pct, plainPriority, type TwinRun } from '../twin'

const MEAL_NAMES = ['Lunch', 'Dinner']
const WALK_SHARE: Record<string, number> = { as_planned: 90, some: 50, little: 15 }
const dayMonth = (iso: string) => new Date(iso + 'T12:00').toLocaleDateString('en-GB', { day: 'numeric', month: 'long' })
const kg = (x: number) => `${Math.round(x * 10) / 10}`

export function Today() {
  const navigate = useNavigate()
  const { user } = useAuth()
  const day = usePatientGet<TodayData>('/today')
  const line = usePatientGet<Timeline>('/timeline')
  const run = usePatientGet<TwinRun>('/twin')
  const t = run.data?.twin
  const [height, setHeight] = useState<number | null>(null)
  const [open, setOpen] = useState<{ id: string; when: string } | null>(null)
  const d = day.data, tl = line.data
  const details = useRecipes((d?.meals_today ?? []).map(m => m.recipe_id))
  useEffect(() => { if (user?.patient_id) getPatient(user.patient_id).then(p => setHeight(latestMeasurements(p.visits).BMXHT ?? null)).catch(() => {}) }, [user?.patient_id])

  if (day.error) return <Shell title="Today"><p className="error">{day.error}</p></Shell>
  if (!d || !tl) return <Shell title="Today" subtitle={longDate(today())}><div className="card narrow"><p className="muted">Getting your day ready…</p></div></Shell>

  const plan = tl.plan
  const visits = tl.visits
  const first = visits[0], latest = visits[visits.length - 1]
  const weights = visits.filter(v => v.measured.BMXWT), waists = visits.filter(v => v.measured.BMXWAIST)
  const loop = tl.loop?.visits ?? []
  const checkins = loop.filter(v => v.verdict !== 'start')
  const lastCheck = checkins[checkins.length - 1]
  const in12 = tl.projection.find(p => p.months_ahead === 12)
  const days = d.next_checkin ? daysUntil(d.next_checkin) : null
  const dayOfPlan = plan ? Math.max(1, Math.round((new Date(today() + 'T12:00').getTime() - new Date(plan.start_date + 'T12:00').getTime()) / 86400000) + 1) : null
  const active = plan?.status === 'active'
  const lastWalk = [...visits].reverse().find(v => v.followed?.walking)?.followed?.walking

  // hero words and buttons, by where the patient is
  const hero = !plan ? { pill: 'Getting started', head: 'Choose your plan to start.', main: ['Choose my plan', '/choose-plan'] as const }
    : !active ? { pill: 'Waiting for approval', head: 'Your clinician is reviewing your plan.', main: ['See where my plan is', '/waiting'] as const }
    : days !== null && days <= 0 ? { pill: `Day ${dayOfPlan}`, head: 'Time for your check-in.', main: ['Start check-in', '/check-in'] as const }
    : { pill: `Day ${dayOfPlan}`, head: dayOfPlan === 1 ? 'Your plan starts today.' : `Day ${dayOfPlan} of your plan.`, main: ['Start check-in', '/check-in'] as const }
  const when = d.next_checkin && days !== null
    ? `Next check-in ${days > 1 ? `in ${days} days, on ${dayMonth(d.next_checkin)}` : days === 1 ? 'tomorrow' : days === 0 ? 'today' : `was due on ${dayMonth(d.next_checkin)}`}.` : ''

  // goals
  const startKg = weights[0]?.measured.BMXWT, nowKg = weights[weights.length - 1]?.measured.BMXWT
  const target = startKg ? startKg * 0.05 : 0
  const lost = startKg && nowKg ? Math.max(0, startKg - nowKg) : 0
  const change = (a?: number, b?: number, unit = '') => a === undefined || b === undefined || Math.abs(b - a) < 0.05 ? 'no change' : `${b < a ? '−' : '+'}${kg(Math.abs(b - a))}${unit}`

  return (
    <Shell title="Today" subtitle={longDate(today())}>
      <div className="page">
        <section className="card hero">
          <div className="hero-text">
            <span className="pill"><i />{hero.pill}</span>
            <h2 className="hero-head">{hero.head}</h2>
            <p className="hero-sub">{[when, plan?.note ? `Note from your clinician: ${plan.note}` : 'Your care team: your clinician and your twin.'].filter(Boolean).join(' ')}</p>
            <div className="row">
              <button className="btn primary lg" onClick={() => navigate(hero.main[1])} disabled={hero.main[1] === '/check-in' && !active}>{hero.main[0]}</button>
              {plan && <button className="btn lg" onClick={() => navigate('/plan')}>View my plan</button>}
            </div>
          </div>
          <div className="hero-figure">
            <LiverFigure chance={latest?.chance_a_lot_of_fat_pct ?? 0} />
            {latest && <div className="hero-stat"><strong>{pct(latest.chance_a_lot_of_fat_pct)}</strong>
              <span>chance of a lot of fat<br />from your {visits.length > 1 ? 'latest check-in' : 'sign-up'}</span>
              <button className="link small" onClick={() => navigate('/liver')}>See my results</button></div>}
          </div>
        </section>

        {d.tip && <TipCard tip={d.tip} onDone={() => { day.reload(); line.reload() }} />}

        <div className="three">
          <section className="card">
            <div className="card-head"><h2>Body</h2>{first && <span className="muted">since {new Date(first.visit_date + 'T12:00').toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}</span>}</div>
            <dl className="body-list">
              <div><dt>Weight</dt><dd><strong>{nowKg ? kg(nowKg) : '—'}</strong> kg</dd><span>{change(startKg, nowKg, ' kg')}</span></div>
              <div><dt>Waist</dt><dd><strong>{waists.length ? kg(waists[waists.length - 1].measured.BMXWAIST) : '—'}</strong> cm</dd>
                <span>{change(waists[0]?.measured.BMXWAIST, waists[waists.length - 1]?.measured.BMXWAIST, ' cm')}</span></div>
              <div><dt>Height</dt><dd><strong>{height ? kg(height) : '—'}</strong> cm</dd><span /></div>
            </dl>
          </section>
          <section className="card">
            <div className="card-head"><h2>Goals</h2><button className="link small" onClick={() => navigate('/progress')}>Details</button></div>
            <div className="goals">
              <div><GoalRing share={target ? (lost / target) * 100 : 0} value={kg(lost)} unit={`of ${kg(target)} kg`} /><span>Lose 5% of weight</span></div>
              <div><GoalRing share={latest?.chance_a_lot_of_fat_pct ?? 0} value={latest ? pct(latest.chance_a_lot_of_fat_pct).replace('more than ', '>').replace('less than ', '<') : '—'} unit="a lot of fat" tone="ink" /><span>Liver fat chance</span></div>
              <div><GoalRing share={lastWalk ? WALK_SHARE[lastWalk] : 0} value={plan ? String(Math.round(plan.walking_min_per_week)) : '—'} unit="min / week" />
                <span>Walking{lastWalk ? <small>{WALK_WORDS[lastWalk].toLowerCase()}</small> : <small>not reported yet</small>}</span></div>
            </div>
          </section>
          <section className="card">
            <div className="card-head"><h2>Coming up</h2></div>
            <ul className="upcoming">
              <li><span className="up-kind">Check-in</span>
                <span className="up-main"><strong>{d.next_checkin ? dayMonth(d.next_checkin) : 'Once your plan starts'}</strong><span>Weight and how it went</span></span></li>
              <li><span className="up-kind">Walk</span>
                <span className="up-main"><strong>{plan ? `${Math.round(plan.walking_min_per_week / 5)} min today` : 'Set with your plan'}</strong><span>{plan ? `${Math.round(plan.walking_min_per_week)} min a week, on 5 days` : ''}</span></span></li>
              {d.next_step.id === 'add_bloods'
                ? <li><span className="up-kind">Bloods</span>
                    <span className="up-main"><strong>Liver blood tests</strong><span>ALT, AST and platelets at your next visit</span></span></li>
                : <li><span className="up-kind">Meals</span>
                    <span className="up-main"><strong>{d.meals_today.length ? `${d.meals_today.length} meals today` : 'No recipe changes'}</strong><span>{d.meals_today.length ? 'Lunch and dinner, below' : 'Keep eating as you do'}</span></span></li>}
            </ul>
          </section>
        </div>

        <section className="loop-band" aria-label="How your twin works">
          <div className="loop-head"><span>How your twin works</span><h2>Predict, measure, adjust.</h2></div>
          <div className="loop-cols">
            <div><span className="loop-num">01</span><h3>Predict</h3>
              {lastCheck?.expected_range_kg ? <><strong>{kg(lastCheck.expected_range_kg[0])}–{kg(lastCheck.expected_range_kg[1])} kg</strong><p>What the twin expected at your last check-in</p></>
                : <><strong>{startKg ? `${kg(startKg)} kg` : '—'}</strong><p>Your starting point</p></>}</div>
            <div><span className="loop-num">02</span><h3>Measure</h3>
              {lastCheck ? <><strong>{kg(lastCheck.weight_kg)} kg</strong><p>Measured on {dayMonth(lastCheck.visit_date)}: {verdictWords(lastCheck).toLowerCase()}</p></>
                : <><strong>{d.next_checkin ? dayMonth(d.next_checkin) : '—'}</strong><p>Your first check-in</p></>}</div>
            <div><span className="loop-num">03</span><h3>Adjust</h3>
              <strong>{in12 ? pct(in12.chance_a_lot_of_fat_pct) : '—'}</strong><p>Chance of a lot of fat in 12 months if you keep to the plan</p></div>
          </div>
        </section>

        <div className="bottom-row">
          <section className="card">
            <div className="card-head"><h2>My liver</h2><button className="link small" onClick={() => navigate('/liver')}>See my results</button></div>
            {t ? <dl className="glance-list">
              <div><dt>Liver fat</dt><dd><strong>{latest ? pct(latest.chance_a_lot_of_fat_pct) : '—'}</strong> chance of a lot of fat{in12 && <span className="muted"> · about {pct(in12.chance_a_lot_of_fat_pct)} in 12 months on the plan</span>}</dd></div>
              <div><dt>Scarring</dt><dd>{!t.fibrosis || ['not_applicable', 'unknown'].includes(t.fibrosis.category) ? <span className="muted">Not worked out yet</span>
                : <><strong>{t.fibrosis.category[0].toUpperCase() + t.fibrosis.category.slice(1)} risk</strong>{t.fibrosis.fib4 !== null && <span className="muted"> · FIB-4 {t.fibrosis.fib4.toFixed(2)}</span>}</>}</dd></div>
              <div><dt>Focus on</dt><dd>{t.priorities.length ? t.priorities.slice(0, 3).map(p => plainPriority(p.title).split(',')[0].split(' (')[0]).join(' · ') : <span className="muted">No diet change needed</span>}</dd></div>
            </dl> : <p className="muted small">Loading…</p>}
          </section>
          <section className="card">
            <div className="card-head"><h2>Today's meals</h2><button className="link small" onClick={() => navigate('/meals')}>This week</button></div>
            {d.meals_today.length ? <ul className="meal-list">
              {d.meals_today.map((m, i) => (
                <li key={m.recipe_id}>
                  <button type="button" className="meal-row" onClick={() => setOpen({ id: m.recipe_id, when: MEAL_NAMES[i] ?? 'Meal' })}>
                    <RecipePhoto recipe={details[m.recipe_id]} size={64} />
                    <span className="dish-main"><span className="caps">{MEAL_NAMES[i] ?? 'Meal'}</span><strong>{m.title}</strong>
                      <span className="muted small">{[minutes(m.total_time), kcal(details[m.recipe_id])].filter(Boolean).join(' · ')}</span></span>
                    <Icon name="chev" />
                  </button>
                </li>))}
            </ul> : <p className="muted small">{d.plan_status === 'awaiting_approval' ? 'Your meals appear once your clinician approves the plan.'
              : d.plan_status === 'active' ? 'Your numbers do not call for recipe changes. Keep eating as you do.' : 'Choose a plan to get your meals.'}</p>}
          </section>
        </div>
      </div>
      {open && <RecipeDrawer id={open.id} when={open.when} onClose={() => setOpen(null)} />}
    </Shell>
  )
}

/** The twin's suggestion after a check-in; recipe-step suggestions can be accepted here. */
export function TipCard({ tip, onDone }: { tip: Tip; onDone: () => void }) {
  const pid = useAuth().user?.patient_id
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)
  const apply = async () => {
    if (!pid || tip.style_step === undefined) return
    setBusy(true)
    try { await api(`/patients/${pid}/plan/style`, { method: 'PUT', body: { style_step: tip.style_step } }); setDone(true); onDone() }
    finally { setBusy(false) }
  }
  return (
    <section className="tip-card">
      <Icon name="check" />
      <div className="grow"><span className="eyebrow">From your last check-in</span><p>{done ? 'Done. Your meals from now on use the new recipes.' : tip.message}</p></div>
      {tip.style_step !== undefined && !done && <button className="btn" onClick={apply} disabled={busy}>
        {tip.id === 'step_up' ? 'Try healthier recipes' : 'Use more familiar recipes'}</button>}
    </section>
  )
}
