// Step 6 check: Dr. Rao opens a demo patient's full page from the patient list (read-only: nothing is changed).
import { openBrowser } from './browser.mjs'
const BASE = 'http://localhost:5173'
const out = process.argv[2] || '.'
const label = process.argv[3] || 'P0001'

const { page, close } = await openBrowser(9338, { width: 1280, height: 900 })
try {
  await page.goto(BASE + '/signin')
  await page.fill('input[type=email]', 'dr.rao@example.org')
  await page.click('text=Send me a code')
  await page.fill('input[inputmode=numeric]', (await page.textContent('.code')).trim())
  await page.click('button:has-text("Sign in")')
  await page.waitForURL('**/patients')
  await page.waitForSelector('table.tbl')
  await page.locator('tr.link-row', { hasText: label }).locator('td').first().click()
  await page.waitForURL('**/patients/*')
  const t = Date.now()
  await page.waitForSelector('.drivers', { timeout: 60000 })
  console.log('patient page:', ((Date.now() - t) / 1000).toFixed(1), 's')
  await page.waitForFunction(() => [...document.images].every(i => i.complete), null, { timeout: 20000 }).catch(() => {})
  await page.screenshot({ path: `${out}/s6_patient.png`, fullPage: true })
  await page.setViewportSize({ width: 400, height: 860 })
  await page.waitForTimeout(800)
  await page.screenshot({ path: `${out}/s6_phone.png` })
} finally { await close() }
