import { useEffect, useState } from 'react'
import { Icon } from './components/Icon'
import { api } from './api'

export interface RecipeDetail {
  recipe_id: string; title: string; cuisine: string | null; total_time: number | null; servings: number | null; url: string | null; image_url: string | null
  per_serving: Record<string, number>; ingredients: string[]; instructions: string[]
}

const cache = new Map<string, RecipeDetail>()

/** Recipe details by id, fetched once per page load and kept. */
export function useRecipes(ids: string[]) {
  const key = ids.join(',')
  const [byId, setById] = useState<Record<string, RecipeDetail>>(() => Object.fromEntries(ids.filter(i => cache.has(i)).map(i => [i, cache.get(i)!])))
  useEffect(() => {
    const missing = ids.filter(i => !cache.has(i))
    const done = () => setById(Object.fromEntries(ids.filter(i => cache.has(i)).map(i => [i, cache.get(i)!])))
    if (!missing.length) { done(); return }
    api<RecipeDetail[]>('/recipes', { params: { ids: missing.join(',') } })
      .then(list => { list.forEach(r => cache.set(r.recipe_id, r)); done() }).catch(() => done())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return byId
}

export const minutes = (m: number | null | undefined) => (m ? `${m} min` : '')
export const kcal = (r?: RecipeDetail) => (r?.per_serving.calories ? `${Math.round(r.per_serving.calories)} kcal` : '')

/** The recipe's food.com photo, or a plain tile with a plate icon when there is none or it does not load. */
export function RecipePhoto({ recipe, size, className = '' }: { recipe?: RecipeDetail; size?: number; className?: string }) {
  const [failed, setFailed] = useState(false)
  const style = size ? { width: size, height: size } : undefined
  if (!recipe?.image_url || failed) return <span className={`photo photo-empty ${className}`} style={style} aria-hidden="true"><Icon name="dish" /></span>
  return <img className={`photo ${className}`} style={style} src={size ? recipe.image_url.replace('w_640,', 'w_240,') : recipe.image_url} alt="" loading="lazy" referrerPolicy="no-referrer" onError={() => setFailed(true)} />
}

/** A side panel with one recipe: why it fits, nutrients, ingredients, steps and the original page. */
export function RecipeDrawer({ id, when, onClose }: { id: string; when?: string; onClose: () => void }) {
  const r = useRecipes([id])[id]
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])
  return (
    <div className="drawer-wrap" role="dialog" aria-modal="true" aria-label={r?.title ?? 'Recipe'}>
      <div className="drawer-shade" onClick={onClose} />
      <aside className="drawer">
        <div className="row"><span className="eyebrow">{when}</span><span style={{ flex: 1 }} /><button className="btn" onClick={onClose}>Close</button></div>
        {!r ? <p className="muted">Loading…</p> : <>
          <RecipePhoto recipe={r} className="photo-hero" />
          <h2 className="lead">{r.title}</h2>
          <p className="muted small">{[minutes(r.total_time), kcal(r), r.cuisine && r.cuisine !== 'Other' ? r.cuisine : ''].filter(Boolean).join(' · ')}</p>
          <div className="nutrients">
            {([['calories', 'kcal', 'Energy'], ['protein', 'g', 'Protein'], ['saturated_fat', 'g', 'Saturated fat'], ['sugar', 'g', 'Sugars'], ['fiber', 'g', 'Fibre']] as const)
              .filter(([k]) => r.per_serving[k] !== undefined)
              .map(([k, unit, label]) => <div key={k}><b>{Math.round(r.per_serving[k] * 10) / 10}</b><span className="muted small">{unit} {label}</span></div>)}
          </div>
          <p className="muted small">Per serving.</p>
          <section><h3>Ingredients</h3><ul className="ingredients">{r.ingredients.map((x, i) => <li key={i}>{x}</li>)}</ul></section>
          <section><h3>Method</h3><ol className="method">{r.instructions.map((x, i) => <li key={i}>{x}</li>)}</ol></section>
          {r.url && <p className="small"><a className="link" href={r.url} target="_blank" rel="noreferrer">Open the original recipe ↗</a></p>}
        </>}
      </aside>
    </div>
  )
}
