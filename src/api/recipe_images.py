"""Recipe photos. RecipeDB gives each recipe an Image_URL (a food.com photo) that the recipe collection did not
keep, so the link is looked up the first time a recipe is shown and saved in data/app/recipe_images.json.

The photo itself is never copied: the browser loads it from food.com. Some links have no photo behind them
(food.com answers 404); the dashboard then shows a plain placeholder.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

CACHE_FILE = Path('data/app/recipe_images.json')
_lock = threading.Lock()
_cache = None


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def smaller(url):
    """The same photo at 640 px wide instead of 1200 px (food.com resizes on request)."""
    return url.replace('q_92,', 'q_80,').replace('w_1200,', 'w_640,') if url else url


def fetch(recipe_id):
    """The photo link from RecipeDB: a URL, '' when the recipe has none, or None when RecipeDB could not be
    reached (not saved, so it is tried again next time)."""
    load_dotenv()
    base = os.getenv('RECIPEDB_BASE_URL')
    if not base:
        return None
    try:
        r = requests.get(f'{base}/search-recipe/{str(recipe_id).zfill(8)}', timeout=5)
        r.raise_for_status()
        return smaller(r.json().get('Image_URL') or '')
    except (requests.RequestException, ValueError):
        return None


def image_urls(ids):
    """{recipe_id: photo URL or ''} for these recipes, looking up the ones not seen before."""
    cache = _load()
    missing = [i for i in dict.fromkeys(ids) if i not in cache]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            found = dict(zip(missing, pool.map(fetch, missing)))
        found = {k: v for k, v in found.items() if v is not None}
        if found:
            with _lock:
                cache.update(found)
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                tmp = CACHE_FILE.with_suffix('.tmp')
                tmp.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding='utf-8')
                tmp.replace(CACHE_FILE)
    return {i: cache.get(i) or '' for i in ids}
