def create_nutrient_profile(recipe):
    nutrition = recipe.nutrition

    return {
        "recipe_id": recipe.recipe_id,
        "title": recipe.title,
        "calories": nutrition.get("calories"),
        "fat": nutrition.get("fat"),
        "saturated_fat": nutrition.get("saturated_fat"),
        "cholesterol": nutrition.get("cholesterol"),
        "sodium": nutrition.get("sodium"),
        "carbohydrates": nutrition.get("carbohydrates"),
        "fiber": nutrition.get("fiber"),
        "sugar": nutrition.get("sugar"),
        "protein": nutrition.get("protein"),
    }