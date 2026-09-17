def create_nutritional_features(profile):

    return {
        "recipe_id": profile["recipe_id"],
        "title": profile["title"],

        # Energy
        "calories": profile["calories"],

        # Macronutrients
        "protein": profile["protein"],
        "carbohydrates": profile["carbohydrates"],
        "fat": profile["fat"],

        # Fat-related variables
        "saturated_fat": profile["saturated_fat"],
        "cholesterol": profile["cholesterol"],

        # Other nutritional variables
        "fiber": profile["fiber"],
        "sugar": profile["sugar"],
        "sodium": profile["sodium"],
    }