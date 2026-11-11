// Step 2 check: a new patient signs up through all 8 screens and sees their results;
// then the seeded demo patient signs in and lands on theirs.
import { openBrowser } from './browser.mjs'
const BASE = 'http://localhost:5173'
const out = process.argv[2] || '.'
const { page, close } = await openBrowser(9334, { width: 1280, height: 900 })

async function signIn(email) {
  await page.goto(BASE + '/signin')
  await page.fill('input[type=email]', email)
  await page.click('text=Send me a code')
  const code = (await page.textContent('.code')).trim()
  await page.fill('input[inputmode=numeric]', code)
  await page.click('button:has-text("Sign in")')
}
const field = label => page.locator('label.field', { hasText: label }).locator('input, textarea').first()
const pick = async (question, answer) =>
  page.locator('.field', { hasText: question }).locator('button.choice', { hasText: answer }).first().click()
const next = () => page.click('button:has-text("Continue")')

try {
  await signIn(`newpatient.${Date.now()}@example.org`)
  await page.waitForURL('**/signup')
  await field('Clinic code').fill('HT-4821'); await next()
  await field('Age').fill('52'); await pick('Sex', 'Woman'); await pick('Ethnic background', 'White'); await pick('pregnant', 'No')
  await page.screenshot({ path: `${out}/s2_about.png`, fullPage: true }); await next()
  await field('Height').fill('165'); await field('Weight').fill('88'); await field('Waist').fill('104'); await next()
  await pick('told you that you have diabetes', 'No'); await pick('high blood pressure', 'Yes'); await pick('100 cigarettes', 'No')
  await pick('alcohol', 'Once a week'); await field('how many drinks').fill('2')
  await page.screenshot({ path: `${out}/s2_health.png`, fullPage: true }); await next()
  await pick('How do you eat', 'Vegetarian (eggs'); await pick('Allergies', 'Peanuts'); await field('Other allergies').fill('kiwi')
  await page.screenshot({ path: `${out}/s2_food.png`, fullPage: true }); await next()
  await pick('Cuisine you cook most', 'European')
  await page.locator('.field', { hasText: 'Dishes you cook often' }).locator('button.choice').first().waitFor()
  await page.locator('.field', { hasText: 'Dishes you cook often' }).locator('button.choice').nth(0).click()
  await page.locator('.field', { hasText: 'Dishes you cook often' }).locator('button.choice').nth(1).click()
  await field('Ingredients you enjoy').fill('lentils, spinach'); await pick('How long can you spend', '45 minutes'); await pick('How many people', '2')
  await page.screenshot({ path: `${out}/s2_tastes.png`, fullPage: true }); await next()
  await field('What you ate yesterday').fill('Breakfast: porridge with milk and a banana. Lunch: cheese sandwich and an apple. Dinner: vegetable lasagne and a glass of orange juice.')
  await page.click('text=Check what we understood')
  await page.waitForSelector('text=% of items understood')
  await page.screenshot({ path: `${out}/s2_diary.png`, fullPage: true }); await next()
  await field('HbA1c').fill('5.9'); await field('Triglycerides').fill('190'); await field('HDL').fill('42')
  await field('ALT').fill('38'); await field('AST').fill('30'); await field('Platelets').fill('240')
  await page.click('button:has-text("See my results")')
  await page.waitForURL('**/results', { timeout: 120000 })
  await page.waitForSelector('.result-number .big', { timeout: 120000 })
  await page.screenshot({ path: `${out}/s2_results_new.png`, fullPage: true })
  console.log('new patient:', (await page.textContent('.result-number')).trim())
  await page.click('text=Sign out')

  await signIn('demo.patient@example.org')
  await page.waitForURL('**/results')
  await page.waitForSelector('.result-number .big', { timeout: 120000 })
  await page.screenshot({ path: `${out}/s2_results_demo.png`, fullPage: true })
  console.log('demo patient:', (await page.textContent('.result-number')).trim())
} finally { await close() }
