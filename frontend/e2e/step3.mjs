// Step 3 check: a patient on diabetes pills chooses a pace, waits, the clinician approves (through the API),
// and the patient chooses Scenario A or B. The patient is made through the API so the demo accounts stay untouched.
import { openBrowser } from './browser.mjs'
const BASE = 'http://localhost:5173'
const API = 'http://127.0.0.1:8000'
const out = process.argv[2] || '.'

async function call(path, { token, method, body } = {}) {
  const r = await fetch(API + path, { method: method ?? (body ? 'POST' : 'GET'),
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: body ? JSON.stringify(body) : undefined })
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`)
  return r.json()
}
async function tokenFor(email) {
  const { code } = await call('/auth/code', { body: { email } })
  return (await call('/auth/session', { body: { email, code } })).token
}

const email = `plan.${Date.now()}@example.org`
const patientToken = await tokenFor(email)
const made = await call('/patients', { token: patientToken, body: {
  visit_date: new Date().toISOString().slice(0, 10), cuisine: 'American', clinic_code: 'HT-4821', food: { diet: 'anything' },
  measurements: { RIDAGEYR: 49, RIAGENDR: 1, RIDRETH3: 3, BMXHT: 172, BMXWT: 96, BMXBMI: 32.4, BMXWAIST: 112, DIQ010: 1, DIQ070: 1,
    LBXGH: 7.4, LBXSTR: 210, LBDHDD: 38, LBXSATSI: 52, LBXSASSI: 34, LBXPLTSI: 230 } } })
await call(`/patients/${made.id}/twin`, { token: patientToken, method: 'POST' })

const { page, close } = await openBrowser(9335, { width: 1280, height: 900 })
try {
  await page.goto(BASE + '/signin')
  await page.fill('input[type=email]', email)
  await page.click('text=Send me a code')
  await page.fill('input[inputmode=numeric]', (await page.textContent('.code')).trim())
  await page.click('button:has-text("Sign in")')
  await page.waitForURL('**/today')
  await page.goto(BASE + '/liver')
  await page.waitForSelector('.liver-view', { timeout: 120000 })
  await page.screenshot({ path: `${out}/s3_results.png`, fullPage: true })
  await page.click('button:has-text("Choose my plan")')
  await page.waitForSelector('.pace-card')
  await page.screenshot({ path: `${out}/s3_pace.png`, fullPage: true })
  await page.click('button:has-text("Choose the")')
  await page.waitForURL('**/waiting')
  await page.screenshot({ path: `${out}/s3_waiting.png`, fullPage: true })

  // Dr. Rao approves through his page, choosing the gentle pace with a note (second browser tab)
  const doctorPage = await page.context().newPage()
  await doctorPage.setViewportSize({ width: 1280, height: 900 })
  await page.click('text=Sign out').catch(() => {})
  await doctorPage.goto(BASE + '/signin')
  await doctorPage.fill('input[type=email]', 'dr.rao@example.org')
  await doctorPage.click('text=Send me a code')
  await doctorPage.fill('input[inputmode=numeric]', (await doctorPage.textContent('.code')).trim())
  await doctorPage.click('button:has-text("Sign in")')
  await doctorPage.waitForURL('**/patients')
  await doctorPage.waitForSelector('table.tbl')
  await doctorPage.screenshot({ path: `${out}/s3_doctor_list.png`, fullPage: true })
  await doctorPage.locator('tr', { hasText: made.label }).locator('button:has-text("Review plan")').click()
  await doctorPage.waitForSelector('.review .pace-card')
  await doctorPage.locator('.review .pace-card', { hasText: 'gentle' }).click()
  await doctorPage.fill('.review input', 'Gentler start because of diabetes pills. Review at 3 months.')
  await doctorPage.screenshot({ path: `${out}/s3_doctor_review.png`, fullPage: true })
  await doctorPage.click('button:has-text("Approve the gentle pace")')
  await doctorPage.waitForSelector('.review', { state: 'detached' })
  await doctorPage.click('text=Sign out')
  await doctorPage.close()
  await page.goto(BASE + '/signin')
  await page.fill('input[type=email]', email)
  await page.click('text=Send me a code')
  await page.fill('input[inputmode=numeric]', (await page.textContent('.code')).trim())
  await page.click('button:has-text("Sign in")')
  await page.waitForURL('**/today')
  await page.goto(BASE + '/waiting')
  await page.waitForSelector('text=Your plan is ready')
  await page.screenshot({ path: `${out}/s3_ready.png`, fullPage: true })
  await page.click('button:has-text("Choose my recipes")')
  await page.waitForSelector('.style-card', { timeout: 120000 })
  await page.screenshot({ path: `${out}/s3_recipes.png`, fullPage: true })
  await page.click('button:has-text("Start with familiar recipes")')
  await page.waitForURL('**/today')
  const plan = (await call(`/patients/${made.id}`, { token: patientToken })).plan
  console.log('plan:', plan.pace, plan.status, 'style step', plan.style_step, '| landed on', page.url())
} finally { await close() }
