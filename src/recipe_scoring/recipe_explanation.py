"""Knowledge-graph-traced explanation of a recommended recipe (Proposal section 7.3).

Each recipe criterion is followed back through the causal knowledge graph (src/knowledge_graph.py) to
liver fat, with the evidence rows and strength behind every link, and the explanation ends with the
cuisine and practicality context that kept the recipe appropriate for the patient. Every statement is
built from the ranking engine's own output, so nothing is added that the twin did not compute.
"""
from src.knowledge_graph import chain
from src.recipe_scoring.practical_feasibility import techniques_and_equipment

CUISINE_CATEGORY = {'exact_match': 'labelled as your cuisine', 'related_match': 'from a related cuisine',
                    'ingredients_only': 'no cuisine label; judged on its ingredients',
                    'no_match': 'labelled as a different cuisine'}


def _link_text(link):
    evidence = ', '.join(link['evidence_ids']) if link['evidence_ids'] else link['reference']
    strength = f'{link["evidence_strength"]}, ' if link['evidence_strength'] else ''
    return f'{link["from"]} {"raises" if link["direction"] == "+" else "lowers"} {link["to"]} ({strength}{evidence})'


def explain_recipe(item, recipe=None, cuisine_detail=None):
    """Structured and plain-text explanation of one ranked recipe."""
    reasons = []
    for criterion in item.get('criteria', []):
        trace = chain(criterion['priority_id'])
        met = criterion['score'] == 100
        reasons.append({
            'priority_id': criterion['priority_id'], 'priority': criterion['priority'],
            'measure': criterion['label'], 'value': criterion['value'], 'reference': criterion['reference'],
            'meets_reference': met, 'evidence': criterion['evidence'],
            'causal_path': trace['path'] if trace else None,
            'links': [_link_text(link) for link in trace['links']] if trace else [],
        })
    context = {'cuisine_score': item.get('scores', {}).get('cuisine'),
               'feasibility_score': item.get('scores', {}).get('feasibility')}
    if cuisine_detail:
        context['cuisine'] = CUISINE_CATEGORY.get(cuisine_detail.get('category'), cuisine_detail.get('category'))
        context['ingredient_typicality'] = cuisine_detail.get('typicality')
    if recipe:
        techniques, equipment = techniques_and_equipment(recipe.get('instructions'))
        context.update(total_time_min=recipe.get('total_time'), ingredients=len(recipe.get('ingredients') or []),
                       steps=len(recipe.get('instructions') or []), techniques=techniques, equipment=equipment)
    return {'recipe_id': item.get('recipe_id'), 'title': item.get('title'), 'rank': item.get('rank'),
            'reasons': reasons, 'context': context, 'text': render(item, reasons, context)}


def render(item, reasons, context):
    lines = [f'{item.get("title")} (rank {item.get("rank")})']
    for r in reasons:
        status = 'meets' if r['meets_reference'] else 'misses'
        lines.append(f'- {r["measure"]}: {r["value"]} ({status} the reference {r["reference"]}; '
                     f'{r["priority"]} priority).')
        if r['causal_path']:
            lines.append(f'  Why it matters: {r["causal_path"]}. ' + '; '.join(r['links']) + '.')
    practical = []
    if context.get('total_time_min') is not None:
        practical.append(f'{context["total_time_min"]} minutes')
    if context.get('ingredients'):
        practical.append(f'{context["ingredients"]} ingredients')
    if context.get('steps'):
        practical.append(f'{context["steps"]} steps')
    extra = context.get('techniques', []) + context.get('equipment', [])
    if extra:
        practical.append('needs ' + ', '.join(extra))
    if context.get('cuisine'):
        lines.append(f'- Cuisine: {context["cuisine"]} (score {context["cuisine_score"]}).')
    if practical:
        lines.append('- Practicality: ' + ', '.join(practical) + '.')
    return '\n'.join(lines)
