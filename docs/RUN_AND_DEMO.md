# Running HepatoTwin and giving the demo

This guide starts the dashboard on one computer and walks through a ten-minute demo: a patient's journey,
then the clinician's view of the same patients. Everything runs locally; nothing needs to be deployed.

## 1. What you need

- Python 3.10 with the project's virtual environment (`.venv`)
- Node.js 18 or newer (for the dashboard)
- A `.env` file in the project folder (never committed). It holds:
  - `RECIPEDB_BASE_URL`: the RecipeDB address, used for recipe photos
  - `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` (optional): set these to email sign-in
    codes to real addresses. Without them, codes appear on screen.

## 2. First-time setup (once per computer)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
cd ..
.venv\Scripts\python.exe -m src.api.seed_demo
```

The last command creates the demo clinic, Dr. Rao and five demo patients (about a minute). The patients are
real held-out NHANES participants who were never used to train the model.

## 3. Start it (two terminals)

**Terminal 1, the backend** (from the project folder):

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.app:app --port 8000
```

**Terminal 2, the dashboard:**

```powershell
cd frontend
npx vite --port 5173
```

Open **http://localhost:5173** (use `localhost`, not `127.0.0.1`).

- After the backend starts, it spends about a minute loading the recipes, the model and the ingredient links
  in the background. Pages work during that minute but are slower, so start it a few minutes before the demo.
- **Restart the backend** (Ctrl+C, then the same command) after changing `.env` or any Python file. The
  `--reload` option does not reliably pick up changes on this setup.
- The dashboard reloads by itself when its files change.

## 4. Accounts

| Email | Who | What it shows |
|---|---|---|
| `dr.rao@example.org` | Clinician, Demo clinic (code HT-4821) | Patient list, alerts, plan approval, full patient pages |
| `demo.patient@example.org` | 49-year-old man on diabetes pills, 84% chance of a lot of fat | His plan **needs Dr. Rao's approval** (safety notice) |
| `patient2@example.org` | Mexican cuisine, cooks for three | Plan starts straight away |
| `patient3@example.org` | European cuisine, vegetarian, tree-nut allergy | Food rules: only vegetarian, nut-free recipes |
| `patient4@example.org` | American cuisine, 45-minute cooking limit | Time limit on recipes |
| `patient5@example.org` | Asian cuisine, pescatarian, little sign of liver fat | Low-risk result, no recipe change needed |

To sign in, type the email and press **Send me a code**. For these `@example.org` addresses the code appears on
screen (no email is sent). Real addresses get an email when SMTP is set up.

## 5. Before every demo or rehearsal

```powershell
.venv\Scripts\python.exe -m src.api.seed_demo --reset
.venv\Scripts\python.exe -m src.api.remove_test_accounts --yes
```

The first command puts the five demo patients back to fresh sign-ups (no plan, no check-ins); Dr. Rao and any
other accounts are kept. The second removes accounts made by the automated browser tests.

## 6. The demo script (about ten minutes)

**Start: the idea (30 seconds).** "HepatoTwin is a digital twin of liver fat. From a routine check-up it
estimates the chance of fatty liver, picks recipes in the patient's own cuisine that meet every nutrition target
for them, and then predicts, measures and adjusts at each monthly check-in."

**Patient: results (2 minutes).** Sign in as `demo.patient@example.org`.
1. **Today** is the home page. Point out the headline, the liver drawing with the 84% chance, and the
   "Predict, measure, adjust" band.
2. Open **My liver**. Today's liver is shown beside the liver in 12 months. Drag to turn it.
   - Say: "The colour follows the chance of a lot of fat. It's a drawing, not a scan. The model gives chances
     rather than an exact reading, because its exact prediction is only moderately accurate (R² 0.56)."
3. Scroll down: At a glance, the scarring check (FIB-4), the numbers against published cut-offs, and the safety
   notice about his diabetes medicine.

**Patient: choosing a plan (2 minutes).**
4. Press **Choose my plan**. Three paces come from trial prescriptions (DPP, Look AHEAD), each with the expected
   weight change in a year. Pick one.
5. Because of his diabetes medicine, the plan waits for the clinician: show the **Waiting** page.

**Clinician: approval and the patient page (3 minutes).** Sign out, then sign in as `dr.rao@example.org`.
6. The **patient list** is sorted by what needs attention; his plan is at the top, awaiting approval.
7. Click his row to open the **patient page**:
   - predicted CAP with its 90% range, the S0–S3 grade and the calibrated chances;
   - **What moved the estimate** (SHAP): which measurements raised or lowered it from the average;
   - scarring, the MASLD criteria and the safety notice;
   - nutritional priorities with their evidence;
   - **Scenario A vs B** side by side, and why the top recipes were chosen, as knowledge-graph paths
     ("energy intake → body weight → liver fat") with the studies behind each.
8. Press **Review plan**, choose the gentle pace, add a note ("Gentler start because of diabetes pills"), and
   approve.

**Patient: recipes and the week (2 minutes).** Sign back in as `demo.patient@example.org`.
9. The plan is approved (Waiting shows **Your plan is ready**). Choose **close to what you eat now** or **the
   healthiest**: each card lists the dishes with photos and compares them with what he eats now.
10. Show **Meals**: the week with photos, a recipe (ingredients, method, nutrients) and the **shopping list**.

**Patient: a check-in (1 minute).**
11. Go to **Today**, press **Check in early**, enter a weight 1–2 kg lower, choose "Most of the meals" and "As
    planned", and save. Show **what changed**: the twin's verdict against the weight it expected, the new
    chance, and the next check-in. **Progress** then shows the weight chart with the expected range.

**Close (30 seconds).** "Every number traces back to the model, the evidence table or a published cut-off; the
summary the patient reads is checked against the twin's output; and projections are labelled as simulations,
not forecasts."

## 7. If something goes wrong

| Problem | What to do |
|---|---|
| A page says it cannot reach the server | The backend is not running: start Terminal 1 again. |
| First page loads are slow | The backend is still warming up (first minute after starting). Wait, then reload. |
| No email arrives | Check the SMTP settings in `.env`, then restart the backend. Demo accounts never get email. |
| The 3D liver shows as a flat drawing | The browser has 3D graphics turned off; the flat drawing is the intended fallback. |
| Recipe photos are grey tiles | That recipe has no photo on food.com, or RecipeDB was unreachable; the page still works. |
| Port 8000 or 5173 is already in use | An old copy is still running: close it, or restart the computer. |
| An Indian-cuisine patient gets few recipes | Known limit: the recipe sample has only 13 Indian main meals. |

## 8. Tests

```powershell
.venv\Scripts\python.exe -m pytest src/api -q
cd frontend
npx tsc --noEmit -p tsconfig.app.json
node e2e/step4.mjs        # patient journey in a headless browser (needs both servers running)
node e2e/step6.mjs        # clinician patient page
cd ..
.venv\Scripts\python.exe -m src.api.remove_test_accounts --yes
```

The browser tests create `@example.org` test accounts; the last command removes them.
