import { Suspense, lazy, useEffect, useState } from 'react'
import type { Timeline } from '../progress'
import { LABEL_WORDS, pct, type PatientTwin } from '../twin'

const Liver3D = lazy(() => import('../components/Liver3D'))   // three.js loads only on this page
// the colour key, as in components/Liver3D.tsx (kept here so the key shows before the 3D code has loaded)
const LEGEND = 'linear-gradient(to top, #163aa8 0%, #2f7df6 30%, #2fcfe6 50%, #f2d94e 68%, #ff9a2e 84%, #ef4a2f 100%)'

function verdict(some: number, aLot: number, future: boolean) {
  // the twin's own decision thresholds (0.365 for CAP >= 280, 0.432 for CAP >= 248), as in the summary
  if (aLot >= 37) return future ? 'A lot of fat would still be likely.' : 'There is a high chance you have a lot of fat in your liver.'
  if (some >= 43) return future ? 'Some fat would still be likely.' : 'You likely have some fat in your liver.'
  return future ? 'Little sign of liver fat would be expected.' : 'Your results show little sign of liver fat.'
}

function useNarrow(px = 720) {
  const [narrow, setNarrow] = useState(() => window.innerWidth < px)
  useEffect(() => { const on = () => setNarrow(window.innerWidth < px); window.addEventListener('resize', on); return () => window.removeEventListener('resize', on) }, [px])
  return narrow
}

/** The top of My liver: today's liver beside the liver at a chosen time on the plan (the twin's projection), in a
 *  dark 3D panel. How much lights up, and how warm the colour, follows the chance of a lot of fat. */
export function LiverView({ twin, timeline }: { twin: PatientTwin; timeline: Timeline | null }) {
  const now = { label: 'Today', some: twin.chance_some_fat_pct, aLot: twin.chance_a_lot_of_fat_pct }
  const ahead = (timeline?.projection ?? []).map(p => ({ label: `In ${p.months_ahead} months`, months: p.months_ahead, some: p.chance_some_fat_pct, aLot: p.chance_a_lot_of_fat_pct, kg: p.weight_kg }))
  const [at, setAt] = useState<number | null>(null)
  const pick = ahead.length ? ahead[at ?? ahead.length - 1] : null
  const narrow = useNarrow()
  const views = pick ? [now, pick] : [now]

  return (
    <section className="card liver-view">
      <div className={`liver-stage${pick ? ' two' : ''}${narrow ? ' stacked' : ''}`}>
        <div className="stage-labels">
          {views.map((v, i) => (
            <div key={i} className={i === 1 ? 'later' : ''}>
              <strong>{v.label}</strong>
              <span>{pct(v.aLot)} chance of a lot of fat</span>
            </div>))}
        </div>
        <Suspense fallback={<div className="liver3d" />}>
          <Liver3D views={views.map(v => ({ fat: v.aLot }))} stacked={narrow}
                   label={views.map(v => `${v.label}: ${pct(v.aLot)} chance of a lot of fat`).join('; ')} />
        </Suspense>
        <div className="legend" aria-hidden="true">
          <span className="legend-title">Liver fat</span>
          <div className="legend-body"><i style={{ background: LEGEND }} /><div><span>A lot</span><span>Some</span><span>Little</span><span>None</span></div></div>
        </div>
        <span className="stage-hint">Drag to turn · double-click to reset</span>
      </div>

      <div className="liver-info">
        <div className="liver-info-main">
          <p className="eyebrow blue">Liver fat today</p>
          <h2 className="lead">{verdict(now.some, now.aLot, false)}</h2>
          <div className="chance-pair">
            <div><span className="metric">{pct(now.some)}</span><span className="muted small">chance of at least some fat (FibroScan 248 dB/m or more)</span></div>
            <div><span className="metric">{pct(now.aLot)}</span><span className="muted small">chance of a lot of fat (280 dB/m or more)</span></div>
          </div>
        </div>
        <div className="liver-info-side">
          {pick ? <>
            <p className="eyebrow">Compare with</p>
            <div className="timeline-slider">
              <input type="range" min={0} max={ahead.length - 1} step={1} value={at ?? ahead.length - 1} onChange={e => setAt(Number(e.target.value))}
                     aria-label="Time on the plan" aria-valuetext={pick.label} />
              <div className="slider-labels">{ahead.map((x, i) => <button key={x.label} className={x === pick ? 'on' : ''} onClick={() => setAt(i)}>{x.months} months</button>)}</div>
            </div>
            <p><strong>{pick.label}:</strong> {verdict(pick.some, pick.aLot, true).toLowerCase()} {pct(pick.some)} chance of some fat, {pct(pick.aLot)} of a lot.</p>
            <p className="muted small">A simulated projection, not a clinical forecast: it assumes you reach about {pick.kg.toFixed(1)} kg, as your plan and your progress so far suggest.</p>
          </> : <p className="muted small">Once your plan starts, a second liver appears here showing how yours could change over 12 months.</p>}
          <p className="muted small">The twin cannot measure your liver fat exactly, so it gives the chance that a FibroScan would read above these levels
            (cut-offs from Karlas et al. 2017). The colours show how likely a lot of fat is; it is a drawing, not a scan of your liver.
            {twin.liver_label && <> This pattern fits <strong>{LABEL_WORDS[twin.liver_label] ?? twin.liver_label}</strong>, a research label, not a diagnosis.</>}</p>
        </div>
      </div>
    </section>
  )
}
