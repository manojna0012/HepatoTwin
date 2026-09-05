import ast


def parse_metadata(recipe):
    """
    Extract basic recipe metadata.
    """

    return {
        "recipe_id": recipe.get("Recipe_ID"),
        "title": recipe.get("Recipe_Title"),
        "url": recipe.get("URL"),
        "source": recipe.get("Source"),
        "image_id": recipe.get("Image_ID"),
        "image_url": recipe.get("Image_URL"),
        "category": recipe.get("Category"),
        "cuisine": recipe.get("Cuisine"),
        "servings": recipe.get("Servings"),
        "ratings": recipe.get("Ratings"),
        "ratings_count": recipe.get("Ratings_Count"),
        "diet": recipe.get("Diet"),
        "diets": recipe.get("Diets")
    }


def parse_preparation(recipe):
    """
    Extract preparation and cooking time information.
    """

    return {
        "recipe_id": recipe.get("Recipe_ID"),
        "prep_time": recipe.get("Prep_Time"),
        "cook_time": recipe.get("Cook_Time"),
        "total_time": recipe.get("Total_Time")
    }


def parse_nutrition(recipe):
    """
    Extract nutritional information.
    """

    nutrition = recipe.get("Nutrition") or {}

    return {
        "recipe_id": recipe.get("Recipe_ID"),
        "calories": nutrition.get("Calories"),
        "fat": nutrition.get("Fat"),
        "saturated_fat": nutrition.get("Saturated_Fat"),
        "cholesterol": nutrition.get("Cholesterol"),
        "sodium": nutrition.get("Sodium"),
        "carbohydrates": nutrition.get("Carbohydrates"),
        "fiber": nutrition.get("Fiber"),
        "sugar": nutrition.get("Sugar"),
        "protein": nutrition.get("Protein")
    }


def parse_ingredients(recipe):
    """
    Extract and structure ingredient information.
    """

    ingredients = []

    for ingredient in recipe.get("Ingredients") or []:
        ingredients.append({
            "recipe_id": recipe.get("Recipe_ID"),
            "ingredient_id": ingredient.get("Ing_ID"),
            "name": ingredient.get("NAME"),
            "name_lc": ingredient.get("NAME_lc"),
            "quantity": ingredient.get("QUANTITY"),
            "unit": ingredient.get("UNIT"),
            "state": ingredient.get("STATE"),
            "temperature": ingredient.get("TEMP"),
            "ingredient_phrase": ingredient.get("Ingredient_Phrases"),
            "flavordb_category": ingredient.get("FlavorDB_Category"),
            "predicted_category": ingredient.get("Predicted_Category")
        })

    return ingredients


def parse_instructions(recipe):
    """
    Convert the Instructions field into a list of steps.
    """

    raw_instructions = recipe.get("Instructions")

    if not raw_instructions:
        return []

    if isinstance(raw_instructions, list):
        return raw_instructions

    try:
        instructions = ast.literal_eval(raw_instructions)

        if isinstance(instructions, list):
            return instructions

    except (ValueError, SyntaxError):
        pass

    return []


def parse_recipe(recipe):
    """
    Parse a complete RecipeDB recipe.
    """

    return {
        "metadata": parse_metadata(recipe),
        "preparation": parse_preparation(recipe),
        "nutrition": parse_nutrition(recipe),
        "ingredients": parse_ingredients(recipe),
        "instructions": parse_instructions(recipe)
    }