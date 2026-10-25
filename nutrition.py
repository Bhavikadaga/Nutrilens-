"""Nutrition lookup + BMI + calorie target.
OWNER: Member 3 (this is a working stub so the app runs today).

Data file: data/nutrition.csv  (falls back to data/nutrition_sample.csv)
Required columns (names are matched loosely, so 'Dish Name', 'Calories (kcal)',
'Carbohydrates (g)', 'Protein (g)', 'Fats (g)' also work):
    food, calories, protein, carbs, fats      [optional: serving_desc]
Values are for ONE serving. get_nutrition(food, servings) multiplies by servings.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "data", "nutrition.csv")
SAMPLE_PATH = os.path.join(BASE, "data", "nutrition_sample.csv")
NUTRIENTS = ["calories", "protein", "carbs", "fats"]


def normalize(name) -> str:
    """'Masala Dosa' / 'masala-dosa' / 'masala_dosa' -> 'masala_dosa'."""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def display_name(label) -> str:
    return str(label).replace("_", " ").title()


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    patterns = {"food": ("dish", "food", "name"), "calories": ("calor", "energy"),
                "protein": ("protein",), "carbs": ("carb",), "fats": ("fat",)}
    rename = {}
    for target, keys in patterns.items():
        if target in df.columns:
            continue
        for col in df.columns:
            if col not in rename and any(k in col.lower() for k in keys):
                rename[col] = target
                break
    df = df.rename(columns=rename)
    missing = [c for c in ["food"] + NUTRIENTS if c not in df.columns]
    if missing:
        raise ValueError(f"Nutrition file is missing columns: {missing}")
    return df


def _table() -> pd.DataFrame:
    path = CSV_PATH if os.path.exists(CSV_PATH) else SAMPLE_PATH
    df = _standardize(pd.read_csv(path))
    df["key"] = df["food"].map(normalize)
    df = df.drop_duplicates("key").set_index("key")
    for n in NUTRIENTS:
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0)
    return df


def available_foods():
    return sorted(_table().index.tolist())


def missing_foods(class_names):
    """Model classes that have no row in the nutrition table (integration check)."""
    idx = set(_table().index)
    return [c for c in class_names if normalize(c) not in idx]


def get_nutrition(food_label: str, servings: float = 1.0) -> dict:
    row = _table().loc[normalize(food_label)]
    return {n: round(float(row[n]) * servings, 1) for n in NUTRIENTS}


def serving_description(food_label: str) -> str:
    row = _table().loc[normalize(food_label)]
    desc = row.get("serving_desc") if "serving_desc" in row.index else None
    return "1 serving" if desc is None or pd.isna(desc) else str(desc)


# ------------------------------------------------------------------ BMI / calories
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


ACTIVITY = {
    "Sedentary (little exercise)": 1.2,
    "Light (1-3 days/week)": 1.375,
    "Moderate (3-5 days/week)": 1.55,
    "Active (6-7 days/week)": 1.725,
}
GOALS = {"Lose weight": -500, "Maintain weight": 0, "Gain weight": 300}


def recommend_calories(age, gender, weight, height, activity, goal) -> int:
    """Mifflin-St Jeor BMR x activity factor + goal adjustment (min 1200 kcal)."""
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161)
    return int(max(1200, round(bmr * ACTIVITY[activity] + GOALS[goal])))
