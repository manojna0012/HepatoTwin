import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { useAuth } from './auth'
import type { Plan } from './patient'

export type Followed = { meals?: 'most' | 'half' | 'few'; walking?: 'as_planned' | 'some' | 'little' } | null

export interface LoopVisit {
  visit_date: string; weight_kg: number; expected_kg: number | null; expected_range_kg: [number, number] | null
  verdict: string; progress: string | null; cause: string | null; adherence: number | null
}
export interface Tip { id: string; message: string; style_step?: number }
export interface Timeline {
  plan: Plan | null
  visits: { visit_id: number; visit_date: string; followed: Followed; measured: Record<string, number>; chance_some_fat_pct: number; chance_a_lot_of_fat_pct: number }[]
  loop: { visits: LoopVisit[]; energy_change_kcal: number; plan_changes: { start_date: string; energy_change_kcal: number }[] } | null
  projection: { months_ahead: number; weight_kg: number; range_kg: [number, number]; change_kg: number; chance_some_fat_pct: number; chance_a_lot_of_fat_pct: number }[]
  tip?: Tip | null
}
export interface Meal { recipe_id: string; title: string; cuisine: string | null; total_time: number | null; ingredients: number }
export interface TodayData {
  status: string | null; next_step: { id: string; message: string }; meals_today: Meal[]
  next_checkin: string | null; tip: Tip | null; plan_status: string | null
}

/** GET a patient endpoint for the signed-in patient; reload() fetches it again. */
export function usePatientGet<T>(path: string) {
  const { user } = useAuth()
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pid = user?.patient_id
  const reload = useCallback(() => {
    if (!pid) return
    setError(null)
    api<T>(`/patients/${pid}${path}`).then(setData).catch(e => setError((e as Error).message))
  }, [pid, path])
  useEffect(reload, [reload])
  return { data, error, reload, pid }
}

export const MEAL_WORDS = { most: 'Most of the meals', half: 'About half', few: 'A few or none' } as const
export const WALK_WORDS = { as_planned: 'As planned', some: 'Some of it', little: 'Little or none' } as const
export const PACE_WORDS: Record<string, string> = { gentle: 'Gentle', standard: 'Standard', ambitious: 'Faster' }

/** Plain words for the twin's verdict at a check-in. */
export function verdictWords(v: LoopVisit) {
  if (v.verdict === 'start') return 'Start'
  if (v.progress === 'behind plan') return v.cause === 'plan not followed' ? 'Behind: plan hard to follow' : 'Behind: no progress yet despite effort'
  if (v.progress === 'ahead of plan' || v.verdict === 'faster than expected') return 'Ahead of plan'
  if (v.verdict === 'slower than expected') return 'On plan, slower this month'
  return 'On track'
}

export const daysUntil = (iso: string) => Math.round((new Date(iso + 'T12:00').getTime() - new Date(new Date().toDateString() + ' 12:00').getTime()) / 86400000)
export const shortDate = (iso: string) => new Date(iso + 'T12:00').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
