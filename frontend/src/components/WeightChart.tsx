import { useEffect, useRef, useState } from 'react'
import type { Timeline } from '../progress'

const DAY = 86400000
const t = (iso: string) => new Date(iso + 'T12:00').getTime()
const shortDate = (ms: number) => new Date(ms).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })

/** Weight over time: what was measured (dots), the range the twin expected at each check-in (grey bars)
 *  and the projection at 3, 6 and 12 months if the plan goes on as now (dashed, with its range shaded). */
export function WeightChart({ timeline, height = 220, projection = true }: { timeline: Timeline; height?: number; projection?: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const [W, setW] = useState(640)
  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const loop = timeline.loop?.visits ?? []
  const measured = (loop.length ? loop.map(v => ({ ms: t(v.visit_date), kg: v.weight_kg, range: v.expected_range_kg, expected: v.expected_kg }))
    : timeline.visits.filter(v => v.measured.BMXWT).map(v => ({ ms: t(v.visit_date), kg: v.measured.BMXWT!, range: null, expected: null })))
  if (!measured.length) return <p className="muted small">No weight yet.</p>
  const last = measured[measured.length - 1]
  const ahead = projection ? timeline.projection.map(p => ({ ms: last.ms + p.months_ahead * 30.44 * DAY, kg: p.weight_kg, range: p.range_kg, months: p.months_ahead })) : []

  const H = height, L = 40, R = 18, T = 14, B = 26
  const xs = [measured[0].ms, ...ahead.map(p => p.ms), last.ms + (ahead.length ? 0 : 30 * DAY)]
  const x0 = Math.min(...xs), x1 = Math.max(...xs)
  const ys = [...measured.flatMap(m => [m.kg, ...(m.range ?? [])]), ...ahead.flatMap(p => [p.kg, ...p.range])]
  const span = Math.max(6, Math.max(...ys) - Math.min(...ys) + 2)
  const step = [1, 2, 5, 10, 20].find(s => span / s <= 5) ?? 50
  const y0 = Math.floor((Math.min(...ys) - 1) / step) * step, y1 = Math.ceil((Math.max(...ys) + 1) / step) * step
  const X = (ms: number) => L + ((ms - x0) / (x1 - x0 || 1)) * (W - L - R)
  const Y = (kg: number) => T + ((y1 - kg) / (y1 - y0)) * (H - T - B)
  const ticks = Array.from({ length: Math.round((y1 - y0) / step) + 1 }, (_, i) => y0 + i * step)
  const lastLabel = measured.length > 1 && X(last.ms) - X(measured[0].ms) > 64 && (!ahead.length || X(ahead[0].ms) - X(last.ms) > 48)
  const band = ahead.length ? [`${X(last.ms)},${Y(last.kg)}`, ...ahead.map(p => `${X(p.ms)},${Y(p.range[1])}`),
    ...[...ahead].reverse().map(p => `${X(p.ms)},${Y(p.range[0])}`)].join(' ') : ''

  return (
    <div ref={box} className="chart-box"><svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Weight chart, latest ${last.kg} kg`}>
      {ticks.map(k => <g key={k}><line x1={L} x2={W - R} y1={Y(k)} y2={Y(k)} className="grid" /><text x={L - 8} y={Y(k) + 4} textAnchor="end" className="axis">{k}</text></g>)}
      {band && <polygon points={band} className="proj-band" />}
      {ahead.length > 0 && <polyline points={[last, ...ahead].map(p => `${X(p.ms)},${Y(p.kg)}`).join(' ')} className="proj-line" />}
      {ahead.map(p => <g key={p.months}><circle cx={X(p.ms)} cy={Y(p.kg)} r="3.5" className="proj-dot" />
        <text x={X(p.ms)} y={H - 8} textAnchor="middle" className="axis">{p.months} mo</text></g>)}
      {measured.map((m, i) => m.range && <line key={'r' + i} x1={X(m.ms)} x2={X(m.ms)} y1={Y(m.range[1])} y2={Y(m.range[0])} className="expect" />)}
      {measured.length > 1 && <polyline points={measured.map(m => `${X(m.ms)},${Y(m.kg)}`).join(' ')} className="obs-line" />}
      {measured.map((m, i) => <circle key={i} cx={X(m.ms)} cy={Y(m.kg)} r="4.5" className="obs-dot" />)}
      <text x={X(measured[0].ms)} y={H - 8} textAnchor="start" className="axis">{shortDate(measured[0].ms)}</text>
      {lastLabel && <text x={X(last.ms)} y={H - 8} textAnchor="middle" className="axis">{shortDate(last.ms)}</text>}
    </svg></div>
  )
}

export function ChartKey({ projection = true, expected = true }: { projection?: boolean; expected?: boolean }) {
  return (
    <div className="chart-key">
      <span><i className="k-obs" />Measured</span>
      {expected && <span><i className="k-exp" />Expected range at each check-in</span>}
      {projection && <span><i className="k-proj" />If the plan goes on as now</span>}
    </div>
  )
}
