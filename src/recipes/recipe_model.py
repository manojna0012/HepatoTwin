from dataclasses import dataclass
from typing import Any


@dataclass
class Recipe:
    recipe_id: str
    title: str
    cuisine: str
    category: str
    servings: Any
    prep_time: Any
    cook_time: Any
    total_time: Any
    nutrition: dict
    ingredients: list
    instructions: list
    diet: Any
    diets: list
    ratings: Any
    ratings_count: Any
    source: str
    url: str


def create_recipe(parsed):
    metadata = parsed["metadata"]
    preparation = parsed["preparation"]

    return Recipe(
        recipe_id=metadata["recipe_id"],
        title=metadata["title"],
        cuisine=metadata["cuisine"],
        category=metadata["category"],
        servings=metadata["servings"],
        prep_time=preparation["prep_time"],
        cook_time=preparation["cook_time"],
        total_time=preparation["total_time"],
        nutrition=parsed["nutrition"],
        ingredients=parsed["ingredients"],
        instructions=parsed["instructions"],
        diet=metadata["diet"],
        diets=metadata["diets"],
        ratings=metadata["ratings"],
        ratings_count=metadata["ratings_count"],
        source=metadata["source"],
        url=metadata["url"]
    )