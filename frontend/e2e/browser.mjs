// Start headless Edge with a debugging port and connect Playwright to it (Playwright's own launch of
// this Edge exits at once here). Returns { browser, page, close }.
import { spawn } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium } from 'playwright-core'

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'

export async function openBrowser(port = 9333, viewport = { width: 1280, height: 800 }) {
  const profile = mkdtempSync(join(tmpdir(), 'ht-edge-'))
  const edge = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
    '--no-first-run', '--disable-extensions', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', 'about:blank'], { stdio: 'ignore' })
  let browser
  for (let i = 0; i < 40 && !browser; i++) {
    try { browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`) } catch { await new Promise(r => setTimeout(r, 500)) }
  }
  if (!browser) throw new Error('Edge did not start')
  const context = browser.contexts()[0]
  const page = await context.newPage()
  await page.setViewportSize(viewport)
  return { browser, page, close: async () => { await browser.close().catch(() => {}); edge.kill() } }
}
