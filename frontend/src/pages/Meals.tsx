import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Shell } from '../components/Shell'
import { today } from '../patient'
import { usePatientGet, type Meal } from '../progress'
import { RecipeDrawer, RecipePhoto, kcal, minutes, useRecipes } from '../recipes'

interface Week { withheld: boolean; message?: string; style?: string; start_date?: string | null; days: Meal[][] }
interface Shopping { servings: string; people: number; groups: Record<string, { ingredient: string; grams: number | null }[]> }

const MEAL_NAMES = ['Lunch', 'Dinner']
// FoodOn-style groups from the ingredient table, in shop order and plain words
const GROUPS: [string, string][] = [
  ['Vegetable', 'Vegetables'], ['Vegetable Fruit', 'Vegetables'], ['Fruit', 'Fruit'], ['Herb', 'Herbs'], ['Legume', 'Beans and lentils'],
  ['Meat', 'Meat and fish'], ['Poultry', 'Meat and fish'], ['Fish', 'Meat and fish'], ['Seafood', 'Meat and fish'],
  ['Dairy', 'Dairy and eggs'], ['Egg', 'Dairy and eggs'], ['Cereal', 'Grains, bread and corn'], ['Maize', 'Grains, bread and corn'],
  ['Bakery', 'Grains, bread and corn'], ['Nuts & Seed', 'Nuts and seeds'], ['Essential Oil', 'Oils and vinegar'],
  ['Plant Derivative', 'Oils and vinegar'], ['Spice', 'Spices and tins'], ['Condiment', 'Sauces'], ['Dish', 'Ready-made'],
  ['Beverage', 'Drinks and juices'], ['Additive', 'Salt, seasonings and baking'], ['Additive-Salt', 'Salt, seasonings and baking'],
]
const NAMES = [...new Set(GROUPS.map(([, n]) => n)), 'Other']
const groupName = (g: string) => GROUPS.find(([k]) => k === g)?.[1] ?? 'Other'
const amount = (g: number | null) => (g === null ? 'as needed' : g >= 1000 ? `${(g / 1000).toFixed(1)} kg` : `${g} g`)

export function Meals() {
  const navigate = useNavigate()
  const [tab, setTab] = useState<'week' | 'shopping'>('week')
  const week = usePatientGet<Week>('/week')
  const [open, setOpen] = useState<{ id: string; when: string } | null>(null)
  const w = week.data
  const details = useRecipes((w?.days ?? []).flat().map(m => m.recipe_id))

  if (week.error) return <Shell title="Meals"><div className="page narrow"><section className="notice"><p>{week.error.includes('twin') ? 'Your meals appear once your results are ready.' : week.error}</p>
    <div><button className="btn" onClick={() => navigate('/today')}>Back to Today</button></div></section></div></Shell>
  if (!w) return <Shell title="Meals"><div className="card narrow"><p className="muted">Planning your week…</p></div></Shell>
  if (w.withheld || !w.days.length) return <Shell title="Meals"><div className="page narrow"><section className="notice">
    <p>{w.message ?? 'Your numbers do not call for recipe changes, so there is no meal plan. Keep eating as you do.'}</p></section></div></Shell>

  const start = w.start_date ?? today()
  const offset = Math.max(0, Math.round((new Date(today() + 'T12:00').getTime() - new Date(start + 'T12:00').getTime()) / 86400000))
  const todayIndex = offset % w.days.length
  const dateOf = (i: number) => { const d = new Date(today() + 'T12:00'); d.setDate(d.getDate() + i - todayIndex); return d }

  return (
    <Shell title="Meals" subtitle={`Two meals a day${w.style ? `, recipes ${w.style === 'healthiest' ? 'that are the healthiest' : w.style}` : ''}. Every meal meets your nutrition targets.`}>
      <div className="page">
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'week'} onClick={() => setTab('week')}>This week</button>
          <button role="tab" aria-selected={tab === 'shopping'} onClick={() => setTab('shopping')}>Shopping list</button>
        </div>
        {tab === 'week' ? (
          <section className="card week">
            {w.days.map((day, i) => {
              const date = dateOf(i)
              return (
                <div key={i} className={`week-day${i === todayIndex ? ' is-today' : ''}`}>
                  <div className="week-date"><strong>{date.toLocaleDateString('en-GB', { weekday: 'short' })}</strong>
                    <span className="muted small">{i === todayIndex ? 'Today' : date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}</span></div>
                  {day.map((m, j) => (
                    <button key={j} type="button" className="week-meal" onClick={() => setOpen({ id: m.recipe_id, when: `${MEAL_NAMES[j]} · ${date.toLocaleDateString('en-GB', { weekday: 'long' })}` })}>
                      <RecipePhoto recipe={details[m.recipe_id]} size={56} />
                      <span className="dish-main"><span className="caps">{MEAL_NAMES[j] ?? 'Meal'}</span>
                        <strong>{m.title}</strong>
                        <span className="muted small">{[minutes(m.total_time), kcal(details[m.recipe_id])].filter(Boolean).join(' · ')}</span></span>
                    </button>
                  ))}
                </div>)
            })}
          </section>
        ) : <ShoppingList />}
      </div>
      {open && <RecipeDrawer id={open.id} when={open.when} onClose={() => setOpen(null)} />}
    </Shell>
  )
}

function ShoppingList() {
  const shop = usePatientGet<Shopping>('/week/shopping')
  const key = `ht-shopping-${shop.pid}`
  const [got, setGot] = useState<Record<string, boolean>>({})
  useEffect(() => { try { setGot(JSON.parse(localStorage.getItem(key) ?? '{}')) } catch { /* private window */ } }, [key])
  const toggle = (name: string) => setGot(g => {
    const next = { ...g, [name]: !g[name] }
    try { localStorage.setItem(key, JSON.stringify(next)) } catch { /* private window */ }
    return next
  })
  if (shop.error) return <p className="error">{shop.error}</p>
  if (!shop.data) return <div className="card"><p className="muted">Adding up the week's ingredients…</p></div>
  const merged: Record<string, Shopping['groups'][string]> = {}
  for (const [g, items] of Object.entries(shop.data.groups)) (merged[groupName(g)] ??= []).push(...items)
  const groups = Object.entries(merged).sort((a, b) => NAMES.indexOf(a[0]) - NAMES.indexOf(b[0]))
  const total = groups.reduce((n, [, items]) => n + items.length, 0)
  const ticked = groups.reduce((n, [, items]) => n + items.filter(i => got[i.ingredient]).length, 0)
  return (
    <section className="card">
      <div className="card-head"><div><h2>For the week</h2><p className="muted small">Amounts for {shop.data.servings.replace('one serving of each meal for ', '')}, one serving of each meal.</p></div>
        <span className="muted small">{ticked} of {total} ticked</span></div>
      <div className="shop-groups">
        {groups.map(([g, items]) => (
          <div key={g} className="shop-group">
            <h3>{g}</h3>
            <ul>{items.map(i => (
              <li key={i.ingredient}><label className={got[i.ingredient] ? 'got' : ''}>
                <input type="checkbox" checked={!!got[i.ingredient]} onChange={() => toggle(i.ingredient)} />
                <span className="grow">{i.ingredient}</span><span className="muted small">{amount(i.grams)}</span></label></li>))}
            </ul>
          </div>))}
      </div>
    </section>
  )
}
