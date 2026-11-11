import { Shell } from '../components/Shell'

// Pages still to build; each names the step of the build plan that adds it.
export function Placeholder({ title, step }: { title: string; step: number }) {
  return (
    <Shell title={title}>
      <div className="card narrow">
        <p className="eyebrow blue">Coming in step {step}</p>
        <p className="muted">This screen is part of the next build steps.</p>
      </div>
    </Shell>
  )
}
