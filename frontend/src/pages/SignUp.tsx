import { useEffect, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError } from '../api'
import { useAuth } from '../auth'
import { Choice, Choices, Field, NumberField, words } from '../components/Form'
import { Shell } from '../components/Shell'

// ---------- answers -> NHANES codes the twin reads ----------
const ETHNICITY = [
  { value: 3, label: 'White' }, { value: 4, label: 'Black' }, { value: 1, label: 'Mexican American' },
  { value: 2, label: 'Other Hispanic' }, { value: 6, label: 'Asian' }, { value: 7, label: 'Other or mixed' },
]
// ALQ121 codes; ALQ111 = 2 for "never in my life"
const ALCOHOL = [
  { value: 'never', label: 'Never in my life' }, { value: '0', label: 'Not in the past year' },
  { value: '9', label: 'A few times a year' }, { value: '7', label: 'About once a month' },
  { value: '6', label: '2 to 3 times a month' }, { value: '5', label: 'Once a week' },
  { value: '4', label: 'Twice a week' }, { value: '3', label: '3 to 4 times a week' },
  { value: '2', label: '5 to 6 times a week' }, { value: '1', label: 'Every day' },
]
const CUISINES = ['American', 'Mexican', 'European', 'Asian', 'Canadian']
const ALLERGEN_LABELS: Record<string, string> = {
  milk: 'Milk', egg: 'Egg', fish: 'Fish', shellfish: 'Shellfish', tree_nuts: 'Tree nuts', peanuts: 'Peanuts',
  wheat: 'Wheat', gluten: 'Gluten', soy: 'Soy', sesame: 'Sesame',
}
const BLOODS: [string, string, string][] = [
  ['LBXGH', 'HbA1c', '%'], ['LBXSGL', 'Glucose', 'mg/dL'], ['LBXSTR', 'Triglycerides', 'mg/dL'], ['LBDHDD', 'HDL cholesterol', 'mg/dL'],
  ['LBXTC', 'Total cholesterol', 'mg/dL'], ['LBXSATSI', 'ALT', 'U/L'], ['LBXSASSI', 'AST', 'U/L'], ['LBXSGTSI', 'GGT', 'U/L'],
  ['LBXPLTSI', 'Platelets', '×10⁹/L'],
]

interface Questions {
  diet: { options: string[]; labels: string[] }
  allergies: { options: string[]; note: string }
  avoid: { options: string[] }
  max_minutes: { options: (number | null)[]; labels: string[] }
}
interface Dish { recipe_id: string; title: string }

interface Answers {
  clinicCode: string
  age: string; sex?: 1 | 2; ethnicity?: number; pregnant?: 'yes' | 'no'
  height: string; weight: string; waist: string; hip: string
  diabetes?: 1 | 2; pills?: 1 | 2; insulin?: 1 | 2; bloodPressure?: 1 | 2; smoked?: 1 | 2
  alcohol?: string; drinks: string
  diet: string; allergies: string[]; otherAllergies: string; avoid: string[]
  cuisine?: string; favourites: string[]; likes: string; dislikes: string; maxMinutes?: number | null; household: number
  diary: string
  bloods: Record<string, string>; systolic: string; diastolic: string
}
const EMPTY: Answers = {
  clinicCode: '', age: '', height: '', weight: '', waist: '', hip: '', drinks: '', diet: 'anything', allergies: [],
  otherAllergies: '', avoid: [], favourites: [], likes: '', dislikes: '', maxMinutes: null, household: 1, diary: '',
  bloods: {}, systolic: '', diastolic: '',
}
const num = (v: string) => (v.trim() === '' || Number.isNaN(Number(v)) ? undefined : Number(v))

function toMeasurements(a: Answers): Record<string, number> {
  const m: Record<string, number | undefined> = {
    RIDAGEYR: num(a.age), RIAGENDR: a.sex, RIDRETH3: a.ethnicity,
    RIDEXPRG: a.sex === 2 && a.pregnant ? (a.pregnant === 'yes' ? 1 : 2) : undefined,
    BMXHT: num(a.height), BMXWT: num(a.weight), BMXWAIST: num(a.waist), BMXHIP: num(a.hip),
    DIQ010: a.diabetes, DIQ070: a.pills, DIQ050: a.insulin, BPQ020: a.bloodPressure, SMQ020: a.smoked,
  }
  const h = num(a.height), w = num(a.weight)
  if (h && w) m.BMXBMI = +(w / (h / 100) ** 2).toFixed(1)
  if (a.alcohol === 'never') { m.ALQ111 = 2 } else if (a.alcohol !== undefined) {
    m.ALQ111 = 1; m.ALQ121 = Number(a.alcohol)
    if (a.alcohol !== '0') m.ALQ130 = num(a.drinks)
  }
  for (const [code] of BLOODS) m[code] = num(a.bloods[code] ?? '')
  const sys = num(a.systolic), dia = num(a.diastolic)
  for (const i of [1, 2, 3]) { m[`BPXOSY${i}`] = sys; m[`BPXODI${i}`] = dia }
  return Object.fromEntries(Object.entries(m).filter(([, v]) => v !== undefined)) as Record<string, number>
}

const STEPS = ['Account', 'About you', 'Body', 'Health', 'What you can eat', 'Your tastes', "Yesterday's meals", 'Blood tests']

export function SignUp() {
  const { user, refresh } = useAuth()
  const navigate = useNavigate()
  const [step, setStep] = useState(0)
  const [a, setA] = useState<Answers>(EMPTY)
  const [questions, setQuestions] = useState<Questions | null>(null)
  const [dishes, setDishes] = useState<Dish[]>([])
  const [diaryPreview, setDiaryPreview] = useState<null | { kcal: number; understood: number; missed: string[] }>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const set = (patch: Partial<Answers>) => setA(prev => ({ ...prev, ...patch }))

  useEffect(() => { api<Questions>('/food/questions').then(setQuestions).catch(e => setError(e.message)) }, [])
  useEffect(() => {
    if (!a.cuisine) return
    api<Dish[]>('/dishes', { params: { cuisine: a.cuisine, diet: a.diet, allergies: a.allergies.join(','), avoid: a.avoid.join(',') } })
      .then(d => { setDishes(d); set({ favourites: a.favourites.filter(f => d.some(x => x.recipe_id === f)) }) })
      .catch(e => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a.cuisine, a.diet, a.allergies.join(','), a.avoid.join(',')])

  const missing = (): string | null => {
    if (step === 1) {
      const age = num(a.age)
      if (age === undefined || age < 12 || age > 80) return 'Please give your age (12 to 80).'
      if (!a.sex) return 'Please choose your sex.'
      if (!a.ethnicity) return 'Please choose your ethnic background.'
    }
    if (step === 2 && (!num(a.height) || !num(a.weight) || !num(a.waist))) return 'Height, weight and waist are needed for your result.'
    if (step === 4 && !a.diet) return 'Please say how you eat.'
    if (step === 5 && !a.cuisine) return 'Please choose the cuisine you cook most.'
    return null
  }
  const next = () => {
    const problem = missing()
    setError(problem)
    if (!problem) { setStep(s => s + 1); window.scrollTo(0, 0) }
  }

  const previewDiary = async () => {
    setBusy('diary')
    try {
      const out = await api<{ measurements: Record<string, number>; share_understood: number; not_understood: string[] }>(
        '/diary/preview', { body: { text: a.diary, usual_day: true } })
      setDiaryPreview({ kcal: Math.round(out.measurements.DR1TKCAL ?? 0), understood: out.share_understood, missed: out.not_understood })
    } catch (e) { setError((e as Error).message) }
    setBusy(null)
  }

  const finish = async () => {
    setBusy('save'); setError(null)
    const food = {
      diet: a.diet, allergies: a.allergies, other_allergies: words(a.otherAllergies), avoid: a.avoid,
      dislikes: words(a.dislikes), likes: words(a.likes), max_minutes: a.maxMinutes, household: a.household,
    }
    const today = new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10)
    try {
      const made = await api<{ id: number }>('/patients', { body: {
        visit_date: today, measurements: toMeasurements(a), cuisine: a.cuisine, favourites: a.favourites, food,
        clinic_code: a.clinicCode.trim() || undefined, diary_text: a.diary.trim() || undefined, usual_day: true } })
      setBusy('twin')
      await api(`/patients/${made.id}/twin`, { method: 'POST' })
      await refresh()
      navigate('/liver')
    } catch (e) {
      const message = (e as Error).message
      setError(message)
      if (e instanceof ApiError && /clinic code/i.test(message)) setStep(0)
      setBusy(null)
    }
  }

  const screens: [string, string, ReactNode][] = [
    ['Create your account', 'Your clinic gives you a code that links you to your clinician. You can add it later.', <>
      <Field label="Email"><input className="input" value={user?.email ?? ''} readOnly /></Field>
      <Field label="Clinic code" hint="Optional. For example HT-4821; without it you can still use HepatoTwin on your own.">
        <input className="input" value={a.clinicCode} onChange={e => set({ clinicCode: e.target.value })} />
      </Field>
    </>],
    ['About you', 'Some results depend on age, sex and ethnic background.', <>
      <NumberField label="Age" unit="years" min={12} max={80} value={a.age} onChange={v => set({ age: v })} required />
      <Choice label="Sex" options={[{ value: 1 as const, label: 'Man' }, { value: 2 as const, label: 'Woman' }]} value={a.sex} onChange={v => set({ sex: v })} />
      <Choice label="Ethnic background" options={ETHNICITY} value={a.ethnicity} onChange={v => set({ ethnicity: v })}
              hint="Asian backgrounds use lower BMI thresholds." />
      {a.sex === 2 && <Choice label="Are you pregnant?" options={[{ value: 'no' as const, label: 'No' }, { value: 'yes' as const, label: 'Yes' }]}
                              value={a.pregnant} onChange={v => set({ pregnant: v })} />}
    </>],
    ['Your body', 'A tape measure around your belly button is enough for the waist.', <div className="grid2">
      <NumberField label="Height" unit="cm" min={100} max={230} value={a.height} onChange={v => set({ height: v })} required />
      <NumberField label="Weight" unit="kg" min={25} max={300} value={a.weight} onChange={v => set({ weight: v })} required />
      <NumberField label="Waist" unit="cm" min={40} max={200} value={a.waist} onChange={v => set({ waist: v })} required />
      <NumberField label="Hips" unit="cm" min={50} max={200} value={a.hip} onChange={v => set({ hip: v })} hint="Optional." />
    </div>],
    ['Your health', 'This tells us whether your clinician should check your plan first.', <>
      <Choice label="Has a doctor told you that you have diabetes?" options={[{ value: 1 as const, label: 'Yes' }, { value: 2 as const, label: 'No' }]} value={a.diabetes} onChange={v => set({ diabetes: v })} />
      {a.diabetes === 1 && <>
        <Choice label="Do you take diabetes pills?" options={[{ value: 1 as const, label: 'Yes' }, { value: 2 as const, label: 'No' }]} value={a.pills} onChange={v => set({ pills: v })} />
        <Choice label="Do you take insulin?" options={[{ value: 1 as const, label: 'Yes' }, { value: 2 as const, label: 'No' }]} value={a.insulin} onChange={v => set({ insulin: v })} />
      </>}
      <Choice label="Has a doctor told you that you have high blood pressure?" options={[{ value: 1 as const, label: 'Yes' }, { value: 2 as const, label: 'No' }]} value={a.bloodPressure} onChange={v => set({ bloodPressure: v })} />
      <Choice label="Have you smoked at least 100 cigarettes in your life?" options={[{ value: 1 as const, label: 'Yes' }, { value: 2 as const, label: 'No' }]} value={a.smoked} onChange={v => set({ smoked: v })} />
      <Choice label="How often did you drink alcohol in the past year?" options={ALCOHOL} value={a.alcohol} onChange={v => set({ alcohol: v })} />
      {a.alcohol && a.alcohol !== 'never' && a.alcohol !== '0' &&
        <NumberField label="On a day you drink, how many drinks?" unit="drinks" min={1} max={30} value={a.drinks} onChange={v => set({ drinks: v })}
                     hint="One drink is a small glass of wine, a can of beer or a single measure of spirits." />}
    </>],
    ['What you can eat', 'We never show a recipe that breaks these.', questions && <>
      <Choice label="How do you eat?" options={questions.diet.options.map((o, i) => ({ value: o, label: questions.diet.labels[i] }))} value={a.diet} onChange={v => set({ diet: v })} />
      <Choices label="Allergies" options={questions.allergies.options.map(o => ({ value: o, label: ALLERGEN_LABELS[o] ?? o }))}
               value={a.allergies} onChange={v => set({ allergies: v })} hint={questions.allergies.note} />
      <Field label="Other allergies" hint="Type any others, separated by commas, for example kiwi, mustard. Recipes that list them are never shown.">
        <input className="input" value={a.otherAllergies} onChange={e => set({ otherAllergies: e.target.value })} />
      </Field>
      <Choices label="Anything you never eat?" options={questions.avoid.options.map(o => ({ value: o, label: o[0].toUpperCase() + o.slice(1) }))}
               value={a.avoid} onChange={v => set({ avoid: v })} />
    </>],
    ['Your tastes', 'So the recipes look like food you enjoy and fit your day.', questions && <>
      <Choice label="Cuisine you cook most" options={CUISINES.map(c => ({ value: c, label: c }))} value={a.cuisine} onChange={v => set({ cuisine: v })} />
      {a.cuisine && <Choices label="Dishes you cook often" hint="Pick a few; only dishes that fit your answers are shown. Recipes like these move up your list."
                             options={dishes.map(d => ({ value: d.recipe_id, label: d.title }))} value={a.favourites} onChange={v => set({ favourites: v })} />}
      <Field label="Ingredients you enjoy" hint="Separated by commas. Recipes with these move up your list.">
        <input className="input" value={a.likes} onChange={e => set({ likes: e.target.value })} placeholder="beans, chicken, peppers" />
      </Field>
      <Field label="Ingredients you dislike" hint="Recipes with these are never shown.">
        <input className="input" value={a.dislikes} onChange={e => set({ dislikes: e.target.value })} placeholder="mushroom" />
      </Field>
      <Choice label="How long can you spend cooking a meal?" options={questions.max_minutes.options.map((o, i) => ({ value: o, label: questions.max_minutes.labels[i] }))}
              value={a.maxMinutes} onChange={v => set({ maxMinutes: v })} />
      <Choice label="How many people do you cook for?" options={[1, 2, 3, 4, 5, 6].map(n => ({ value: n, label: n === 6 ? '6+' : String(n) }))}
              value={a.household} onChange={v => set({ household: v })} hint="Your shopping list is sized for them." />
    </>],
    ["Yesterday's meals", 'Optional. It tells us how far the recipes are from what you eat now, so the changes stay realistic.', <>
      <Field label="What you ate yesterday" hint="In your own words, with amounts if you know them.">
        <textarea className="input" rows={4} value={a.diary} onChange={e => { set({ diary: e.target.value }); setDiaryPreview(null) }}
                  placeholder="Breakfast: 2 eggs and toast with butter. Lunch: a turkey sandwich and chips. Dinner: spaghetti with meat sauce, a glass of milk." />
      </Field>
      <div className="row">
        <button type="button" className="btn" disabled={!a.diary.trim() || busy === 'diary'} onClick={previewDiary}>
          {busy === 'diary' ? 'Reading…' : 'Check what we understood'}</button>
        {diaryPreview && <span className="muted small">About {diaryPreview.kcal} kcal; {Math.round(100 * diaryPreview.understood)}% of items understood
          {diaryPreview.missed.length > 0 && <> (not understood: {diaryPreview.missed.join(', ')})</>}</span>}
      </div>
    </>],
    ['Blood tests', 'From your last check-up, all optional. ALT, AST and platelets let us check for liver scarring.', <>
      <div className="grid3">
        {BLOODS.map(([code, label, unit]) => (
          <NumberField key={code} label={label} unit={unit} value={a.bloods[code] ?? ''} min={0}
                       onChange={v => set({ bloods: { ...a.bloods, [code]: v } })} />
        ))}
      </div>
      <div className="grid3">
        <NumberField label="Blood pressure, upper" unit="mmHg" min={60} max={250} value={a.systolic} onChange={v => set({ systolic: v })} />
        <NumberField label="Blood pressure, lower" unit="mmHg" min={30} max={150} value={a.diastolic} onChange={v => set({ diastolic: v })} />
      </div>
    </>],
  ]
  const [title, lead, body] = screens[step]
  const last = step === STEPS.length - 1

  const nav = (
    <nav className="nav" aria-label="Sign-up steps">
      <p className="side-label">Getting started</p>
      {STEPS.map((label, i) => (
        <span key={label} className={`step ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}>
          <span className="lab">{i < step ? '✓ ' : ''}{label}</span>
        </span>
      ))}
    </nav>
  )

  return (
    <Shell title="Create your account" crumb={title} nav={nav}>
      <div className="page narrow">
        <div>
          <p className="eyebrow">Step {step + 1} of {STEPS.length}</p>
          <h2 className="lead">{title}</h2>
          <p className="muted">{lead}</p>
        </div>
        <section className="card form">{body ?? <p className="muted">Loading…</p>}</section>
        {error && <p className="error">{error}</p>}
        <div className="row">
          {step > 0 && <button className="btn" onClick={() => { setError(null); setStep(s => s - 1) }} disabled={!!busy}>Back</button>}
          {!last && <button className="btn primary" onClick={next}>Continue</button>}
          {last && <button className="btn primary" onClick={finish} disabled={!!busy}>
            {busy === 'save' ? 'Saving…' : busy === 'twin' ? 'Working out your results…' : 'See my results'}</button>}
        </div>
      </div>
    </Shell>
  )
}
