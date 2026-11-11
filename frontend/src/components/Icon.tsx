const PATHS: Record<string, string> = {
  today: 'M4 11l8-7 8 7v8a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1z',
  liver: 'M3 11c0-4 4-6 9-6s9 1 9 4-3 4-6 5-5 4-8 4-4-3-4-7z',
  plan: 'M6 4h12a1 1 0 0 1 1 1v15a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zM9 4v2h6V4M8.5 11h7M8.5 15h5',
  meals: 'M12 6a7 7 0 1 1 0 14 7 7 0 0 1 0-14zM12 10v3l2.5 2',
  progress: 'M3 17l5-5 4 3 8-8M15 7h5v5',
  patients: 'M9 5.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7zM3 20c0-3.5 2.7-5.5 6-5.5s6 2 6 5.5M17 5.5a2.5 2.5 0 1 1 0 5M16 14c3 0 5 1.8 5 5',
  patient: 'M7 4h10a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3zM12 8v8M8 12h8',
  home: 'M4 11l8-7 8 7v8a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1z',
  bell: 'M6 16V11a6 6 0 0 1 12 0v5l1.5 2h-15zM10 20a2 2 0 0 0 4 0',
  out: 'M14 4h4a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-4M10 16l-4-4 4-4M6 12h10',
  plus: 'M12 5v14M5 12h14',
  check: 'M5 12l5 5 9-10',
  dish: 'M7 3v8M5 3v5a2 2 0 0 0 4 0V3M7 11v10M16 3c-1.7 0-3 2.2-3 5s1.3 4 3 4v9',
  chev: 'M9 6l6 6-6 6',
}

export function Icon({ name }: { name: keyof typeof PATHS | string }) {
  return <svg className="icon" viewBox="0 0 24 24" aria-hidden="true"><path d={PATHS[name] ?? ''} /></svg>
}
