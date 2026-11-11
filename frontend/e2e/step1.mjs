// Step 1 check: sign in as a new patient through the real UI, then sign out.
import { openBrowser } from './browser.mjs'
const BASE = 'http://localhost:5173'
const out = process.argv[2] || '.'
const { page, close } = await openBrowser()
try {
  await page.goto(BASE)
  await page.waitForURL('**/signin')
  await page.screenshot({ path: `${out}/s1_signin.png` })
  await page.fill('input[type=email]', `patient.${Date.now()}@example.org`)
  await page.click('text=Send me a code')
  const code = (await page.textContent('.code')).trim()
  await page.screenshot({ path: `${out}/s1_code.png` })
  await page.fill('input[inputmode=numeric]', code)
  await page.click('button:has-text("Sign in")')
  await page.waitForURL('**/signup')
  await page.screenshot({ path: `${out}/s1_after_patient.png` })
  console.log('patient signed in, landed on', page.url())
  await page.click('text=Sign out')
  await page.waitForURL('**/signin')
  console.log('signed out, back on', page.url())
} finally { await close() }
