// Calls to the HepatoTwin backend (src/api/app.py). In development Vite forwards /api to the backend.
const BASE = '/api'
const TOKEN_KEY = 'hepatotwin.token'

export type Role = 'patient' | 'clinician'
export interface User { id: number; email: string; role: Role; patient_id: number | null; clinic_id: number | null }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

export function savedToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY) } catch { return null }
}
export function saveToken(token: string | null) {
  try { token ? localStorage.setItem(TOKEN_KEY, token) : localStorage.removeItem(TOKEN_KEY) } catch { /* private window */ }
}

function message(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map(d => (d && typeof d === 'object' && 'msg' in d ? String(d.msg) : String(d))).join('; ')
  return 'Something went wrong.'
}

export async function api<T>(path: string, options: { method?: string; body?: unknown; params?: Record<string, string | number | undefined> } = {}): Promise<T> {
  const query = options.params
    ? '?' + new URLSearchParams(Object.entries(options.params).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)])).toString()
    : ''
  const token = savedToken()
  const response = await fetch(BASE + path + query, {
    method: options.method ?? (options.body ? 'POST' : 'GET'),
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })
  if (!response.ok) {
    let detail: unknown = response.statusText
    try { detail = (await response.json()).detail } catch { /* not JSON */ }
    throw new ApiError(response.status, message(detail))
  }
  return response.status === 204 ? (undefined as T) : response.json()
}

export const auth = {
  // delivery 'email': the code was emailed; 'screen': email is not set up, so the code comes back here
  requestCode: (email: string) => api<{ sent_to: string; delivery: 'email' | 'screen'; code?: string; expires_in_minutes: number }>(
    '/auth/code', { body: { email } }),
  signIn: (email: string, code: string) => api<{ token: string; user: User }>('/auth/session', { body: { email, code } }),
  signOut: () => api<{ signed_out: boolean }>('/auth/logout', { method: 'POST' }),
  me: () => api<User>('/me'),
}
