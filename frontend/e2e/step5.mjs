// Step 5 check: My liver shows the 3D liver; the slider moves it from today to 12 months on the plan.
// The patient is a plan.<timestamp>@example.org test account (remove with src.api.remove_test_accounts --yes).
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
const day = n => new Date(Date.now() + n * 86400000).toISOString().slice(0, 10)

const email = `plan.${Date.now()}@example.org`
const { code } = await call('/auth/code', { body: { email } })
const token = (await call('/auth/session', { body: { email, code } })).token
const made = await call('/patients', { token, body: {
  visit_date: day(-30), cuisine: 'Mexican', clinic_code: 'HT-4821', food: { diet: 'anything' },
  measurements: { RIDAGEYR: 46, RIAGENDR: 1, RIDRETH3: 3, BMXHT: 175, BMXWT: 102, BMXBMI: 33.3, BMXWAIST: 114,
    LBXGH: 6.1, LBXSTR: 230, LBDHDD: 36, LBXSATSI: 58, LBXSASSI: 36, LBXPLTSI: 250 } } })
await call(`/patients/${made.id}/twin`, { token, method: 'POST' })
await call(`/patients/${made.id}/plan`, { token, method: 'PUT', body: { start_date: day(-30), pace: 'standard', style_step: 0 } })

const { page, close } = await openBrowser(9337, { width: 1280, height: 900 })
try {
  await page.goto(BASE + '/signin')
  await page.fill('input[type=email]', email)
  await page.click('text=Send me a code')
  await page.fill('input[inputmode=numeric]', (await page.textContent('.code')).trim())
  await page.click('button:has-text("Sign in")')
  await page.waitForURL('**/today')
  await page.click('nav >> text=My liver')
  await page.waitForSelector('.liver3d canvas, .liver3d.flat', { timeout: 20000 })
  console.log('3D shown:', await page.locator('.liver3d canvas').count() > 0)
  await page.waitForTimeout(2500)
  await page.screenshot({ path: `${out}/s5_today.png` })
  await page.click('.slider-labels button:has-text("3 months")')
  await page.waitForTimeout(2500)
  await page.screenshot({ path: `${out}/s5_3m.png` })
  console.log('labels:', (await page.textContent('.stage-labels')).trim())
  const box = await page.locator('.liver3d').boundingBox()
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2 + 20, { steps: 10 }); await page.mouse.up()
  await page.waitForTimeout(800)
  await page.screenshot({ path: `${out}/s5_turned.png` })
  await page.setViewportSize({ width: 400, height: 860 })
  await page.waitForTimeout(1200)
  await page.screenshot({ path: `${out}/s5_phone.png` })
} finally { await close() }
