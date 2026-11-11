// What the patient's view of a twin run holds (src/api/app.py patient_view), and plain words for it.

export interface Reason { priority_id: string; priority: string; measure: string; value: number | null; reference: string; meets_reference: boolean; evidence: string[] }
export interface RecipeExplanation { recipe_id: string; title: string; rank: number; reasons: Reason[]; text?: string }
export interface PatientTwin {
  summary: string
  summary_checked: boolean
  chance_some_fat_pct: number
  chance_a_lot_of_fat_pct: number
  liver_label: string | null
  fibrosis: { method: string; fib4: number | null; liver_stiffness_kpa: number | null; category: string; message: string; note: string | null }
  notices: string[]
  recipes_withheld: boolean
  priorities: { title: string; priority: string; evidence: string[] }[]
  recipes: RecipeExplanation[]
  scenarios: unknown[]
}
export interface TwinRun { run_id: number; visit_id: number; visit_date: string; view: string; twin: PatientTwin }

// as in src/twin_summary.py
const PLAIN_PRIORITY: [string, string][] = [
  ['weight', 'Losing some weight'], ['dietary quality', 'Eating a healthier diet overall'], ['added sugar', 'Less added sugar'],
  ['saturated fat', 'Less saturated fat, the kind in butter, fatty meat and cheese'],
  ['processed', 'Food made from fresh or less processed ingredients'],
  ['glycaemic load', 'Meals that raise blood sugar less (a lower glycaemic load)'],
]
export const plainPriority = (title: string) => {
  const low = title.toLowerCase()
  return PLAIN_PRIORITY.find(([k]) => low.includes(k))?.[1] ?? title
}

export const LABEL_WORDS: Record<string, string> = {
  MASLD: 'metabolic dysfunction-associated steatotic liver disease (MASLD)',
  MetALD: 'MASLD with increased alcohol intake (MetALD)',
  ALD: 'liver fat that is likely related to alcohol',
  cryptogenic_SLD: 'liver fat without a clear metabolic cause',
  no_steatosis_indicated: 'no sign of fatty liver',
  unknown: 'an unclear liver status',
}

/** A chance as shown to patients: never 0% or 100% (as in the summary). */
export const pct = (value: number) => (value >= 100 ? 'more than 99%' : value <= 0 ? 'less than 1%' : `${value}%`)
