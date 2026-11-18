"""Small seed food table: approximate values per 100 g (USDA-style averages; cooked unless noted). Real
products vary - users can always log custom macros. Extend by appending rows; no code changes needed."""
from typing import Dict, List

FOODS: Dict[str, dict] = {k: dict(name=n, kcal=c, protein=p, carbs=cb, fat=f) for k, n, c, p, cb, f in [
    ("chicken_breast", "Chicken breast, cooked", 165, 31.0, 0.0, 3.6), ("egg", "Egg, whole", 143, 12.6, 0.7, 9.5),
    ("white_rice", "White rice, cooked", 130, 2.7, 28.0, 0.3), ("oats", "Oats, dry", 389, 16.9, 66.3, 6.9),
    ("banana", "Banana", 89, 1.1, 22.8, 0.3), ("apple", "Apple", 52, 0.3, 13.8, 0.2),
    ("milk_whole", "Milk, whole", 61, 3.2, 4.8, 3.3), ("yogurt_plain", "Yogurt, plain (whole milk)", 61, 3.5, 4.7, 3.3),
    ("greek_yogurt", "Greek yogurt, nonfat", 59, 10.2, 3.6, 0.4), ("salmon", "Salmon, cooked", 206, 22.1, 0.0, 12.4),
    ("broccoli", "Broccoli, cooked", 35, 2.4, 7.2, 0.4), ("olive_oil", "Olive oil", 884, 0.0, 0.0, 100.0),
    ("bread_wholewheat", "Bread, whole wheat", 252, 12.5, 42.7, 3.5), ("potato", "Potato, boiled", 87, 1.9, 20.1, 0.1),
    ("lentils", "Lentils, cooked", 116, 9.0, 20.1, 0.4), ("chickpeas", "Chickpeas, cooked", 164, 8.9, 27.4, 2.6),
    ("almonds", "Almonds", 579, 21.2, 21.6, 49.9), ("peanut_butter", "Peanut butter", 588, 25.1, 20.0, 50.4),
    ("tofu", "Tofu, firm", 144, 17.3, 2.8, 8.7), ("beef_lean", "Beef, lean, cooked", 250, 26.0, 0.0, 15.0),
    ("tuna_canned", "Tuna, canned in water", 116, 25.5, 0.0, 0.8), ("sweet_potato", "Sweet potato, baked", 90, 2.0, 20.7, 0.2),
    ("spinach", "Spinach, cooked", 23, 3.0, 3.8, 0.3), ("pasta", "Pasta, cooked", 158, 5.8, 30.9, 0.9)]}


def macros_for(key: str, grams: float) -> Dict[str, float]:
    f = FOODS[key]
    k = grams / 100.0
    return {"kcal": round(f["kcal"] * k, 1), "protein_g": round(f["protein"] * k, 1),
            "carbs_g": round(f["carbs"] * k, 1), "fat_g": round(f["fat"] * k, 1)}


def search(q: str, limit: int = 20) -> List[dict]:
    q = q.lower().strip()
    return [{"key": k, **v} for k, v in FOODS.items() if not q or q in v["name"].lower() or q in k][:limit]
