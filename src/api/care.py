"""What the patient should do next, the tip after a check-in, and the clinician's alerts.

All are rules on the twin's outputs (src/reassessment.py, src/liver_labels.py, src/fibrosis.py) and the
plan (src/care_plan.py), written so each message can be traced to one condition.

Next step on the patient's Today screen, the first that applies:
1. the plan awaits a clinician's approval;
2. no plan yet: choose a pace;
3. a check-in is due (today is on or after the plan's next check-in);
4. the scarring check is missing (an adult without ALT, AST and platelets): add these blood tests;
5. otherwise: cook today's meals.

Tip after a check-in, from the latest progress against the plan and its cause (src/reassessment.py):
- behind, plan not followed: move one step towards familiar food (a smaller change kept beats a
  larger one dropped), or shorter walks at the lowest step;
- behind, slow response: keep going; after two such check-ins in a row the progress is shared with the
  clinician (an alert);
- on or ahead of plan with good adherence at two check-ins in a row: offer one step towards the
  healthiest recipes;
- ahead: keep going, and make sure meals are not being skipped.

Clinician alerts, most urgent first: plan awaiting approval, scarring risk indeterminate or high,
behind twice despite good adherence, check-in overdue (CHECKIN_GRACE_DAYS past the date), new patient
without a plan, scarring check missing.
"""
from datetime import date, timedelta

from src.care_plan import Prescriptions, reported_adherence, walking_kcal
from src.reassessment import GOOD_ADHERENCE, WeightRule, reassess
from src.recipe_scoring.style_ladder import STEPS

CHECKIN_WEEKS = 4
CHECKIN_GRACE_DAYS = 7
SEVERITY = {'awaiting_approval': 1, 'scarring_risk': 1, 'behind_twice': 2, 'checkin_overdue': 3,
            'no_plan': 3, 'scarring_check_missing': 4}


def next_checkin(from_date):
    return (date.fromisoformat(str(from_date)) + timedelta(weeks=CHECKIN_WEEKS)).isoformat()


def plan_parts(plan, weight_kg, rx=None):
    """(diet kcal, walking kcal) of a plan for a patient of this weight."""
    rx = rx or Prescriptions.from_table()
    walk = walking_kcal(weight_kg, plan.get('walking_min_per_week') or 0, rx.met) if weight_kg else 0.0
    return plan['energy_change_kcal'], walk


def plan_history(plans, plan):
    """The plans in force over time, oldest first: for each start date the last one set, ending with the
    current plan. Plans that were replaced before they started (a patient's request the clinician changed)
    drop out."""
    by_start = {}
    for p in sorted(plans or [plan], key=lambda p: p['id']):
        if p['id'] <= plan['id'] and p['start_date'] <= plan['start_date'] and \
                (p['id'] == plan['id'] or p.get('status') == 'active'):
            by_start[p['start_date']] = p
    return [by_start[d] for d in sorted(by_start)]


def loop(visits, plan, rule=None, plans=None):
    """The re-assessment loop on the patient's visits, with the walking in the plan's energy change and
    the adherence reported at each check-in. With the patient's earlier plans (`plans`, as stored), each
    month is judged against the plan in force at the time. None without a plan or a weight."""
    if plan is None or plan.get('status') == 'awaiting_approval':
        return None
    weights = [(v['visit_date'], v['measurements'].get('BMXWT')) for v in visits]
    first = next((w for _, w in weights if w is not None), None)
    if first is None:
        return None
    history = plan_history(plans, plan)
    parts = [(p['start_date'], *plan_parts(p, first)) for p in history]

    def in_force(day):
        return next((x for x in reversed(parts) if x[0] <= str(day)), parts[0])

    adherence = [reported_adherence(v.get('followed'), *in_force(v['visit_date'])[1:]) for v in visits]
    energy = [(d, diet + walk) for d, diet, walk in parts]
    out = reassess(weights, history[0]['start_date'], energy if len(energy) > 1 else energy[0][1],
                   rule or WeightRule.from_tables(), adherence)
    diet, walk = parts[-1][1:]
    out['energy_change_kcal'] = round(diet + walk)
    out['plan_changes'] = [{'start_date': d, 'energy_change_kcal': round(e)} for d, e in energy]
    return out


def _last_checkins(result, n=2):
    return [v for v in (result or {}).get('visits', []) if v['verdict'] != 'start'][-n:]


def tip(result, plan):
    """The suggestion shown after the latest check-in, or None before the first."""
    last = _last_checkins(result)
    if not last:
        return None
    v, step = last[-1], (plan or {}).get('style_step') or 0
    if v['progress'] == 'behind plan':
        if v['cause'] == 'plan not followed':
            if step > 0:
                return {'id': 'step_down', 'style_step': step - 1,
                        'message': 'A smaller change you keep beats a big change you drop. Try recipes closer to '
                                   'what you usually eat this month.'}
            return {'id': 'easier_walks', 'message': 'Try two short walks this week; small steps count.'}
        if v['cause'] == 'slow response':
            if len(last) == 2 and last[0]['cause'] == 'slow response':
                return {'id': 'shared_with_clinician',
                        'message': "You're sticking to the plan and your body is responding slowly. We've shared "
                                   'your progress with your clinician.'}
            return {'id': 'keep_going', 'message': "You're following the plan well. Bodies respond at different "
                                                   'speeds; keep going.'}
        return {'id': 'keep_going', 'message': 'Progress is a little slow. Following more of the meals will help.'}
    good = len(last) == 2 and all((x['adherence'] or 0) >= GOOD_ADHERENCE and x['progress'] != 'behind plan'
                                  for x in last)
    if good and step < STEPS:
        return {'id': 'step_up', 'style_step': step + 1,
                'message': 'You are doing well. Want to try a few of the healthiest recipes this month?'}
    if v['progress'] == 'ahead of plan':
        return {'id': 'keep_going', 'message': 'Ahead of plan. Keep going, and make sure you are not skipping meals.'}
    return {'id': 'keep_going', 'message': 'On plan. Keep going.'}


def scarring(run):
    fibrosis = ((run or {}).get('clinical') or {}).get('fibrosis') or {}
    return fibrosis.get('category')


def notices(run):
    return [n['id'] for n in (((run or {}).get('clinical') or {}).get('guardrails') or {}).get('notices', [])]


def alerts(patient, plan, run, result, today=None):
    """Alerts for the clinician about one patient, most urgent first."""
    today = date.fromisoformat(str(today)) if today else date.today()
    out = []
    if plan is None:
        out.append(('no_plan', 'New patient: no plan yet.'))
    elif plan.get('status') == 'awaiting_approval':
        reasons = ', '.join(n.replace('_', ' ') for n in notices(run)) or 'safety notice'
        out.append(('awaiting_approval', f'Plan waiting for your approval ({reasons}).'))
    category = scarring(run)
    if category in ('indeterminate', 'high'):
        out.append(('scarring_risk', f'Scarring risk {category}: further assessment advised.'))
    last = _last_checkins(result)
    if len(last) == 2 and all(v['cause'] == 'slow response' for v in last):
        out.append(('behind_twice', 'Behind plan at two check-ins despite good reported adherence.'))
    if plan and plan.get('next_checkin') and \
            today > date.fromisoformat(plan['next_checkin']) + timedelta(days=CHECKIN_GRACE_DAYS):
        out.append(('checkin_overdue', f'Check-in overdue since {plan["next_checkin"]}.'))
    if category == 'unknown':
        out.append(('scarring_check_missing', 'No scarring check yet: order ALT, AST and platelets.'))
    return [{'id': i, 'message': m, 'severity': SEVERITY[i]} for i, m in sorted(out, key=lambda x: SEVERITY[x[0]])]


def next_step(plan, run, today=None):
    """The one action on the patient's Today screen."""
    today = date.fromisoformat(str(today)) if today else date.today()
    if plan is not None and plan.get('status') == 'awaiting_approval':
        return {'id': 'awaiting_approval',
                'message': 'Your clinician is reviewing your plan before you start. We will let you know.'}
    if plan is None:
        return {'id': 'choose_plan', 'message': 'Choose the pace of your plan to get your first week of meals.'}
    if plan.get('next_checkin') and today >= date.fromisoformat(plan['next_checkin']):
        return {'id': 'check_in', 'message': 'Time for your check-in: weigh yourself and tell us how the plan went.'}
    if scarring(run) == 'unknown':
        return {'id': 'add_bloods', 'message': 'Add your liver blood tests (ALT, AST, platelets) at your next '
                                               'clinic visit, so we can check for scarring.'}
    return {'id': 'cook', 'message': "Cook today's meals."}
