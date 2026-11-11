import { api } from './api'

export interface Visit { id: number; visit_date: string; measurements: Record<string, number | null>; followed: unknown }
export interface Plan {
  id: number; start_date: string; energy_change_kcal: number; walking_min_per_week: number; pace: string | null
  style_step: number; next_checkin: string; status: 'active' | 'awaiting_approval'; set_by: string; note: string | null
}
export interface PatientRecord {
  id: number; label: string; cuisine: string | null; favourites: string[]; clinic_id: number | null
  food: Record<string, unknown>; visits: Visit[]; plan: Plan | null
}
export interface PaceOption {
  pace: 'gentle' | 'standard' | 'ambitious'; diet_kcal: number; walking_min_per_week: number; total_kcal: number
  expected_change_12m_kg: number; if_followed_change_12m_kg: number; recommended: boolean
}
export interface Suggestion { options: PaceOption[]; note: string; approval_needed_for: string[]; style_steps: string[] }

export const getPatient = (id: number) => api<PatientRecord>(`/patients/${id}`)

/** The latest value of each measurement across visits (what the twin carries forward). */
export function latestMeasurements(visits: Visit[]) {
  const out: Record<string, number> = {}
  for (const v of [...visits].sort((a, b) => a.visit_date.localeCompare(b.visit_date)))
    for (const [k, x] of Object.entries(v.measurements)) if (x !== null && x !== undefined && Number.isFinite(x)) out[k] = x
  return out
}

export const NOTICE_WORDS: Record<string, string> = {
  under_18: 'you are under 18', alcohol: 'your alcohol intake', glucose_lowering_medicine: 'you take medicine that lowers blood sugar',
  pregnant: 'you are pregnant', fibrosis_referral: 'your scarring check needs a clinician first',
}

export const today = () => new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10)
export const longDate = (iso: string) =>
  new Date(iso + 'T12:00').toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })
