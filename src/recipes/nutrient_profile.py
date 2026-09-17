def create_nutrient_profile(recipe):

    nutrition = recipe.nutrition

    return {
        # Recipe identification
        "recipe_id": recipe.recipe_id,
        "title": recipe.title,

        # Energy
        "calories": nutrition.get("calories"),

        # Macronutrients
        "protein": nutrition.get("protein"),
        "carbohydrates": nutrition.get("carbohydrates"),
        "fat": nutrition.get("fat"),

        # Fat-related nutrients
        "saturated_fat": nutrition.get("saturated_fat"),
        "cholesterol": nutrition.get("cholesterol"),

        # Other nutritional features
        "fiber": nutrition.get("fiber"),
        "sugar": nutrition.get("sugar"),
        "sodium": nutrition.get("sodium"),
    }