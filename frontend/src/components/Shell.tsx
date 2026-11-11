import type { ReactNode } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { Icon } from './Icon'

export interface NavItem { to: string; label: string; icon: string; count?: number }

const PATIENT_NAV: NavItem[] = [
  { to: '/today', label: 'Today', icon: 'today' },
  { to: '/liver', label: 'My liver', icon: 'liver' },
  { to: '/plan', label: 'My plan', icon: 'plan' },
  { to: '/meals', label: 'Meals', icon: 'meals' },
  { to: '/progress', label: 'Progress', icon: 'progress' },
]
const CLINICIAN_NAV: NavItem[] = [{ to: '/patients', label: 'Patients', icon: 'patients' }]

interface Props {
  title: string
  crumb?: string
  subtitle?: string
  action?: ReactNode
  alerts?: number
  nav?: ReactNode          // replaces the menu, e.g. the sign-up steps
  children: ReactNode
}

export function Shell({ title, crumb, subtitle, action, alerts = 0, nav, children }: Props) {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const items = user?.role === 'clinician' ? CLINICIAN_NAV : PATIENT_NAV
  const initials = user?.role === 'clinician' ? 'DR' : 'P'
  const name = user?.email.split('@')[0] ?? ''
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="wordmark">HepatoTwin</div>
        {nav ?? (
          <nav className="nav" aria-label="Sections">
            {items.map(item => (
              <NavLink key={item.to} to={item.to} className={({ isActive }) => (isActive ? 'active' : undefined)}>
                <Icon name={item.icon} /><span className="lab">{item.label}</span>
                {item.to === '/patients' && alerts > 0 && <span className="count">{alerts}</span>}
              </NavLink>
            ))}
          </nav>
        )}
        <div className="grow" />
        <button className="signout" onClick={async () => { await signOut(); navigate('/signin') }}>
          <Icon name="out" /><span className="lab">Sign out</span>
        </button>
      </aside>
      <main className="main">
        <header className="topbar">
          <div className="crumb">
            <Icon name="home" /><span>{user?.role === 'clinician' ? 'Clinic' : 'Home'}</span>
            <span className="sep">›</span><span>{crumb ?? title}</span>
          </div>
          <div className="tb-right">
            {action}
            <button className="icon-btn" aria-label={`${alerts} notifications`}>
              <Icon name="bell" />{alerts > 0 && <span className="badge">{alerts}</span>}
            </button>
            <span className="user"><span className="avatar">{initials}</span><strong>{name}</strong></span>
          </div>
        </header>
        <div className="pagehead">
          <h1>{title}</h1>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {children}
      </main>
    </div>
  )
}
