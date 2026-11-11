// Step 4 check: a patient who started a plan 30 days ago picks a recipe style, looks at Today, the week, the
// shopping list, the recipe drawer, My plan and Progress, then checks in through the form and sees what changed.
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
  measurements: { RIDAGEYR: 46, RIAGENDR: 1, RIDRETH3: 3, BMXHT: 175, BMXWT: 95, BMXBMI: 31, BMXWAIST: 108,
    LBXGH: 5.9, LBXSTR: 190, LBDHDD: 40, LBXSATSI: 45, LBXSASSI: 30, LBXPLTSI: 250 } } })
await call(`/patients/${made.id}/twin`, { token, method: 'POST' })
await call(`/patients/${made.id}/plan`, { token, method: 'PUT', body: { start_date: day(-30), pace: 'standard', style_step: 0, next_checkin: day(0) } })

const { page, close } = await openBrowser(9336, { width: 1280, height: 900 })
const shot = async name => page.screenshot({ path: `${out}/s4_${name}.png`, fullPage: true })
const timed = async (label, fn) => { const t = Date.now(); await fn(); console.log(`${label}: ${((Date.now() - t) / 1000).toFixed(1)} s`) }
try {
  await page.goto(BASE + '/signin')
  await page.fill('input[type=email]', email)
  await page.click('text=Send me a code')
  await page.fill('input[inputmode=numeric]', (await page.textContent('.code')).trim())
  await page.click('button:has-text("Sign in")')
  await page.waitForURL('**/today')

  await page.goto(BASE + '/choose-recipes')
  await timed('recipe choice', () => page.waitForSelector('.style-card .dish-rows'))
  await page.waitForTimeout(800)
  await shot('recipes')
  await page.locator('.style-card').nth(1).click()
  await page.locator('.style-card').nth(1).locator('button:has-text("Recipe")').first().click()
  await page.waitForSelector('.drawer h2')
  await shot('drawer')
  await page.click('.drawer button:has-text("Close")')
  await page.click('button:has-text("Start with the healthiest")')
  await page.waitForURL('**/today')
  await timed('today', () => page.waitForSelector('.meal-row'))
  await page.waitForSelector('.hero .liver-figure')
  await page.waitForFunction(() => [...document.images].every(i => i.complete), null, { timeout: 20000 }).catch(() => console.log('photos still loading'))
  await page.waitForTimeout(500)
  await shot('today')

  await page.click('nav >> text=Meals')
  await timed('week', () => page.waitForSelector('.week-meal'))
  await page.waitForFunction(() => [...document.querySelectorAll('.week-meal')].every(m => m.textContent.includes('kcal')), null, { timeout: 20000 })
  await page.waitForFunction(() => [...document.images].every(i => i.complete), null, { timeout: 20000 }).catch(() => console.log('photos still loading'))
  await shot('week')
  await page.click('button[role=tab]:has-text("Shopping list")')
  await timed('shopping', () => page.waitForSelector('.shop-group'))
  await page.locator('.shop-group input').first().check()
  await shot('shopping')

  await page.click('nav >> text=My plan')
  await timed('my plan', () => page.waitForSelector('.ladder'))
  await shot('plan')
  await page.click('nav >> text=Progress')
  await timed('progress', () => page.waitForSelector('svg.chart'))
  await shot('progress_before')

  await page.click('button:has-text("Check in")')
  await page.waitForURL('**/check-in')
  await page.locator('.field', { hasText: 'Your weight today' }).locator('input').fill('93.1')
  await page.click('button[role=radio]:has-text("Most of the meals")')
  await page.click('button[role=radio]:has-text("As planned")')
  await shot('checkin_form')
  await page.click('button:has-text("Save check-in")')
  await timed('check-in saved', () => page.waitForSelector('text=What changed'))
  await shot('checkin_done')
  await page.click('button:has-text("See my progress")')
  await page.waitForSelector('.tbl')
  await page.waitForTimeout(500)
  await shot('progress_after')

  await page.setViewportSize({ width: 400, height: 860 })
  await page.click('nav a[href="/today"]')
  await page.waitForSelector('.meal-row')
  await shot('today_phone')
  const plan = (await call(`/patients/${made.id}`, { token })).plan
  console.log('style step', plan.style_step, '| next check-in', plan.next_checkin)
} finally { await close() }
