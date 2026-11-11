import os

import pytest

from src.api import mailer, recipe_images

os.environ.setdefault("HEPATOTWIN_NO_WARMUP", "1")   # tests load what they need themselves


@pytest.fixture(autouse=True)
def no_real_email(monkeypatch):
    """Tests never send email, even when SMTP is set up in .env: the code comes back in the response."""
    monkeypatch.setattr(mailer, 'settings', lambda: None)


@pytest.fixture(autouse=True)
def no_recipe_photo_lookups(monkeypatch):
    """Tests never call RecipeDB for photo links."""
    monkeypatch.setattr(recipe_images, 'fetch', lambda recipe_id: '')
    monkeypatch.setattr(recipe_images, 'CACHE_FILE', recipe_images.Path('data/app/test_recipe_images.json'))
    monkeypatch.setattr(recipe_images, '_cache', {})
