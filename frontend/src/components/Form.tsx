import type { ReactNode } from 'react'

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  )
}

/** A number box with its unit; empty means "not given". */
export function NumberField({ label, unit, value, onChange, min, max, step = 'any', hint, required }: {
  label: ReactNode; unit?: string; value: string; onChange: (v: string) => void; min?: number; max?: number
  step?: number | 'any'; hint?: ReactNode; required?: boolean
}) {
  return (
    <Field label={label} hint={hint}>
      <span className="unit-input">
        <input className="input" type="number" inputMode="decimal" min={min} max={max} step={step} required={required}
               value={value} onChange={e => onChange(e.target.value)} />
        {unit && <span className="unit">{unit}</span>}
      </span>
    </Field>
  )
}

/** One answer from a short list (shown as buttons). */
export function Choice<T extends string | number | null>({ label, options, value, onChange, hint }: {
  label: ReactNode; options: { value: T; label: string }[]; value: T | undefined; onChange: (v: T) => void; hint?: ReactNode
}) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="choices" role="radiogroup">
        {options.map(o => (
          <button type="button" key={String(o.value)} className="choice" role="radio" aria-checked={value === o.value}
                  aria-pressed={value === o.value} onClick={() => onChange(o.value)}>{o.label}</button>
        ))}
      </div>
      {hint && <small className="muted">{hint}</small>}
    </div>
  )
}

/** Any number of answers from a list. */
export function Choices({ label, options, value, onChange, hint }: {
  label: ReactNode; options: { value: string; label: string }[]; value: string[]; onChange: (v: string[]) => void; hint?: ReactNode
}) {
  const toggle = (v: string) => onChange(value.includes(v) ? value.filter(x => x !== v) : [...value, v])
  return (
    <div className="field">
      <span>{label} <small className="muted">(choose any)</small></span>
      <div className="choices">
        {options.map(o => (
          <button type="button" key={o.value} className="choice" aria-pressed={value.includes(o.value)} onClick={() => toggle(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
      {hint && <small className="muted">{hint}</small>}
    </div>
  )
}

export const words = (text: string) => text.split(',').map(x => x.trim()).filter(Boolean)
