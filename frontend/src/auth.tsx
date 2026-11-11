import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { auth, saveToken, savedToken, type User } from './api'

interface AuthState {
  user: User | null
  loading: boolean
  signIn: (email: string, code: string) => Promise<User>
  signOut: () => Promise<void>
  refresh: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    if (!savedToken()) { setUser(null); return }
    try { setUser(await auth.me()) } catch { saveToken(null); setUser(null) }
  }, [])

  useEffect(() => { refresh().finally(() => setLoading(false)) }, [refresh])

  const signIn = async (email: string, code: string) => {
    const { token, user } = await auth.signIn(email, code)
    saveToken(token)
    setUser(user)
    return user
  }
  const signOut = async () => {
    try { await auth.signOut() } catch { /* already signed out */ }
    saveToken(null)
    setUser(null)
  }
  return <AuthContext.Provider value={{ user, loading, signIn, signOut, refresh }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const state = useContext(AuthContext)
  if (!state) throw new Error('useAuth must be used inside AuthProvider')
  return state
}
