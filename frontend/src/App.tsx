import type { ReactNode } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import type { Role } from './api'
import { AuthProvider, useAuth } from './auth'
import { CheckIn } from './pages/CheckIn'
import { ChoosePlan } from './pages/ChoosePlan'
import { ChooseRecipes } from './pages/ChooseRecipes'
import { Meals } from './pages/Meals'
import { MyPlan } from './pages/MyPlan'
import { PatientDetail } from './pages/PatientDetail'
import { Patients } from './pages/Patients'
import { Progress } from './pages/Progress'
import { Results } from './pages/Results'
import { SignUp } from './pages/SignUp'
import { SignIn, homeFor } from './pages/SignIn'
import { Today } from './pages/Today'
import { Waiting } from './pages/Waiting'

function Guard({ role, children }: { role: Role; children: ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="centered muted">Loading…</div>
  if (!user) return <Navigate to="/signin" replace />
  if (user.role !== role) return <Navigate to={homeFor(user)} replace />
  return <>{children}</>
}

function Home() {
  const { user, loading } = useAuth()
  if (loading) return <div className="centered muted">Loading…</div>
  return <Navigate to={user ? homeFor(user) : '/signin'} replace />
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<Guard role="patient"><SignUp /></Guard>} />
          <Route path="/results" element={<Navigate to="/liver" replace />} />
          <Route path="/choose-plan" element={<Guard role="patient"><ChoosePlan /></Guard>} />
          <Route path="/waiting" element={<Guard role="patient"><Waiting /></Guard>} />
          <Route path="/choose-recipes" element={<Guard role="patient"><ChooseRecipes /></Guard>} />
          <Route path="/today" element={<Guard role="patient"><Today /></Guard>} />
          <Route path="/liver" element={<Guard role="patient"><Results /></Guard>} />
          <Route path="/plan" element={<Guard role="patient"><MyPlan /></Guard>} />
          <Route path="/meals" element={<Guard role="patient"><Meals /></Guard>} />
          <Route path="/progress" element={<Guard role="patient"><Progress /></Guard>} />
          <Route path="/check-in" element={<Guard role="patient"><CheckIn /></Guard>} />
          <Route path="/patients" element={<Guard role="clinician"><Patients /></Guard>} />
          <Route path="/patients/:id" element={<Guard role="clinician"><PatientDetail /></Guard>} />
          <Route path="*" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
