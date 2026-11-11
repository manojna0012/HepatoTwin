/** A ring showing a share (0-100) with the value in the middle. */
export function Ring({ value, label, size = 148 }: { value: number; label: string; size?: number }) {
  const r = 52, c = 2 * Math.PI * r, shown = Math.max(0, Math.min(100, value))
  const text = value >= 100 ? '>99%' : value <= 0 ? '<1%' : `${value}%`
  return (
    <svg className="ring" width={size} height={size} viewBox="0 0 120 120" role="img" aria-label={`${text} ${label}`}>
      <circle cx="60" cy="60" r={r} fill="none" stroke="var(--line)" strokeWidth="10" />
      <circle cx="60" cy="60" r={r} fill="none" stroke="var(--blue)" strokeWidth="10" strokeLinecap="round"
              strokeDasharray={`${(shown / 100) * c} ${c}`} transform="rotate(-90 60 60)" />
      <text x="60" y="62" textAnchor="middle" className="ring-value">{text}</text>
      <text x="60" y="80" textAnchor="middle" className="ring-label">{label}</text>
    </svg>
  )
}

export interface Band { upTo: number; label: string }

/** A horizontal scale split into published ranges, with a marker at the patient's value. */
export function ScaleBar({ value, bands, max, unit = '', format = (v: number) => String(v) }: {
  value: number; bands: Band[]; max: number; unit?: string; format?: (v: number) => string
}) {
  const at = (v: number) => `${(Math.max(0, Math.min(v, max)) / max) * 100}%`
  let start = 0
  return (
    <div className="scale">
      <div className="scale-track">
        {bands.map((b, i) => {
          const left = start; start = Math.min(b.upTo, max)
          return <span key={b.label} className={`scale-band b${i}`} style={{ left: at(left), width: `calc(${at(start)} - ${at(left)})` }} />
        })}
        <span className="scale-marker" style={{ left: at(value) }}><span className="scale-pin">{format(value)}{unit}</span></span>
      </div>
      <div className="scale-labels">
        {(() => {
          let s = 0
          return bands.map(b => {
            const left = s; s = Math.min(b.upTo, max)
            return <span key={b.label} style={{ left: at(left), width: `calc(${at(s)} - ${at(left)})` }}>{b.label}</span>
          })
        })()}
      </div>
      <div className="scale-ticks">
        {bands.slice(0, -1).map(b => <span key={b.label} style={{ left: at(b.upTo) }}>{format(b.upTo)}</span>)}
      </div>
    </div>
  )
}

/** A goal ring: a share (0-100) of the ring filled, with a big value and a small unit in the middle. */
export function GoalRing({ share, value, unit, tone = 'blue', size = 112 }: { share: number; value: string; unit: string; tone?: 'blue' | 'ink'; size?: number }) {
  const r = 46, c = 2 * Math.PI * r, s = Math.max(0, Math.min(100, share))
  return (
    <svg className={`goal-ring ${tone}`} width={size} height={size} viewBox="0 0 112 112" role="img" aria-label={`${value} ${unit}`}>
      <circle cx="56" cy="56" r={r} className="track" />
      {s > 0 && <circle cx="56" cy="56" r={r} className="fill" strokeDasharray={`${(s / 100) * c} ${c}`} transform="rotate(-90 56 56)" />}
      <text x="56" y="58" textAnchor="middle" className="gr-value">{value}</text>
      <text x="56" y="75" textAnchor="middle" className="gr-unit">{unit}</text>
    </svg>
  )
}

// Fixed spots for fat droplets inside the liver shape; the first n are drawn for n droplets.
const DROPS: [number, number, number][] = [
  [96, 84, 7], [158, 70, 6], [212, 92, 7], [128, 122, 6], [186, 128, 5], [74, 116, 5], [246, 76, 6], [130, 56, 4],
  [104, 150, 5], [228, 120, 4], [160, 100, 5], [62, 92, 4], [262, 104, 4], [150, 150, 4],
]

/** A drawn liver whose fat droplets follow the chance of a lot of fat (an illustration, not an image of the patient). */
export function LiverFigure({ chance }: { chance: number }) {
  const n = Math.round((Math.max(0, Math.min(100, chance)) / 100) * DROPS.length)
  const shape = 'M36 118C26 72 64 36 140 30c70-6 140 4 162 40 12 20 2 38-26 52-40 20-92 40-140 52-46 12-86 4-100-56z'
  return (
    <svg className="liver-figure" viewBox="0 0 320 200" role="img" aria-label={`Drawing of a liver with ${n} of ${DROPS.length} fat droplets`}>
      <defs>
        <radialGradient id="lv-fill" cx="0.34" cy="0.28" r="0.85"><stop offset="0" stopColor="#e6b3a3" /><stop offset=".55" stopColor="#cf8c79" /><stop offset="1" stopColor="#b06654" /></radialGradient>
        <radialGradient id="lv-drop" cx="0.35" cy="0.3" r="0.8"><stop offset="0" stopColor="#fbf3dc" /><stop offset="1" stopColor="#e8d29b" /></radialGradient>
        <clipPath id="lv-clip"><path d={shape} /></clipPath>
      </defs>
      <ellipse cx="166" cy="186" rx="112" ry="7" fill="#000" opacity=".07" />
      <path d={shape} fill="url(#lv-fill)" />
      <g clipPath="url(#lv-clip)">
        <path d="M196 30c-18 42-18 92 4 140" fill="none" stroke="#f7e4dc" strokeOpacity=".5" strokeWidth="1.6" />
        {DROPS.slice(0, n).map(([x, y, r], i) => <circle key={i} cx={x} cy={y} r={r} fill="url(#lv-drop)" />)}
      </g>
      <path d="M70 70c24-18 58-26 96-27" fill="none" stroke="#fff" strokeOpacity=".4" strokeWidth="5" strokeLinecap="round" />
    </svg>
  )
}
