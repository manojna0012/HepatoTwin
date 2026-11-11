import { useNavigate } from 'react-router-dom'
import { Shell } from '../components/Shell'
import { ChartKey, WeightChart } from '../components/WeightChart'
import { MEAL_WORDS, WALK_WORDS, shortDate, usePatientGet, verdictWords, type Timeline } from '../progress'
import { pct } from '../twin'

export function Progress() {
  const navigate = useNavigate()
  const line = usePatientGet<Timeline>('/timeline')
  const tl = line.data
  if (line.error) return <Shell title="Progress"><p className="error">{line.error}</p></Shell>
  if (!tl) return <Shell title="Progress"><div className="card narrow"><p className="muted">Loading your progress…</p></div></Shell>

  const loop = tl.loop?.visits ?? []
  const weights = tl.visits.filter(v => v.measured.BMXWT).map(v => v.measured.BMXWT)
  const first = weights[0], now = weights[weights.length - 1]
  const checkins = loop.filter(v => v.verdict !== 'start')
  const followedOn = (date: string) => tl.visits.find(v => v.visit_date === date)?.followed
  const chances = [...tl.visits.map(v => ({ label: shortDate(v.visit_date), value: v.chance_a_lot_of_fat_pct, ahead: false })),
    ...tl.projection.map(p => ({ label: `In ${p.months_ahead} months`, value: p.chance_a_lot_of_fat_pct, ahead: true }))]

  return (
    <Shell title="Progress" subtitle="What you measured, what the twin expected, and where the plan is heading."
           action={tl.plan?.status === 'active' ? <button className="btn primary" onClick={() => navigate('/check-in')}>Check in</button> : undefined}>
      <div className="page">
        <div className="stat-row">
          <div className="card stat"><span className="eyebrow">Starting weight</span><strong>{first ? `${first} kg` : '—'}</strong></div>
          <div className="card stat"><span className="eyebrow">Now</span><strong>{now ? `${now} kg` : '—'}</strong></div>
          <div className="card stat"><span className="eyebrow">Change</span>
            <strong>{first && now ? `${now - first > 0 ? '+' : now - first < 0 ? '−' : ''}${Math.abs(now - first).toFixed(1)} kg` : '—'}</strong></div>
          <div className="card stat"><span className="eyebrow">Check-ins</span><strong>{checkins.length}</strong>
            <span className="muted small">{tl.plan ? `Next ${shortDate(tl.plan.next_checkin)}` : ''}</span></div>
        </div>

        <section className="card">
          <div className="card-head"><h2>Weight</h2></div>
          <WeightChart timeline={tl} height={260} />
          <ChartKey projection={tl.projection.length > 0} expected={checkins.length > 0} />
          {tl.projection.length > 0 && <p className="muted small">The projection is a simulation from your plan and how your body has responded so far, not a clinical forecast.</p>}
        </section>

        <div className="grid23">
          <section className="card">
            <div className="card-head"><h2>Chance of a lot of liver fat</h2></div>
            <div className="bars">
              {chances.map((c, i) => (
                <div key={i} className={`bar-row${c.ahead ? ' ahead' : ''}`}>
                  <span className="small">{c.label}</span>
                  <span className="bar"><span style={{ width: `${Math.max(2, c.value)}%` }} /></span>
                  <strong className="num">{pct(c.value)}</strong>
                </div>))}
            </div>
            <p className="muted small">Lighter bars: if the plan goes on as now.</p>
          </section>
          <section className="card">
            <div className="card-head"><h2>Check-ins</h2></div>
            {checkins.length ? <div className="tbl-wrap"><table className="tbl">
              <thead><tr><th>Date</th><th className="r">Weight</th><th>Meals · walking</th><th>Twin's verdict</th></tr></thead>
              <tbody>{[...checkins].reverse().map(v => {
                const f = followedOn(v.visit_date)
                return (
                  <tr key={v.visit_date}><td>{shortDate(v.visit_date)}</td>
                    <td className="r">{v.weight_kg} kg{v.expected_range_kg && <div className="muted small">expected {v.expected_range_kg[0].toFixed(1)}–{v.expected_range_kg[1].toFixed(1)}</div>}</td>
                    <td className="small">{f?.meals ? MEAL_WORDS[f.meals] : '—'} · {f?.walking ? WALK_WORDS[f.walking].toLowerCase() : '—'}</td>
                    <td>{verdictWords(v)}</td></tr>)
              })}</tbody></table></div>
              : <p className="muted small">Your first check-in is on {tl.plan ? shortDate(tl.plan.next_checkin) : 'the date in your plan'}. After it, the twin compares your weight with what it expected.</p>}
          </section>
        </div>
      </div>
    </Shell>
  )
}
