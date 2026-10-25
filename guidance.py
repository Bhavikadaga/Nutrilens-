"""General (non-medical) nutrition guidance shown in the Profile tab."""

DISCLAIMER = ("These are general estimates for a college project, not medical advice. "
              "Consult a doctor or dietitian for personal health decisions.")

SUGGESTED_GOAL = {"Underweight": "Gain weight", "Normal": "Maintain weight",
                  "Overweight": "Lose weight", "Obese": "Lose weight"}

_TIPS = {
    "Underweight": [
        "Add energy-dense, nutritious foods such as nuts, dairy, paneer and legumes.",
        "Eat 4-5 smaller meals or snacks through the day.",
        "Include protein at every meal to support healthy weight gain.",
    ],
    "Normal": [
        "Keep a balanced plate: vegetables, protein, whole grains and some healthy fat.",
        "Stay consistent with meal timing and drink enough water.",
        "Limit deep-fried foods and sugary sweets to occasional treats.",
    ],
    "Overweight": [
        "Aim for a modest calorie deficit rather than crash dieting.",
        "Fill half your plate with vegetables and choose whole grains over refined.",
        "Watch portion sizes of fried snacks, sweets and sugary drinks.",
    ],
    "Obese": [
        "Start with small, sustainable changes: smaller portions and fewer fried foods.",
        "Prioritise protein and fibre to stay full for longer.",
        "Consider speaking to a doctor or dietitian for a personalised plan.",
    ],
}


def macro_targets(kcal: int) -> dict:
    """Grams/day from a 20% protein / 50% carbs / 30% fat calorie split."""
    return {"protein": round(kcal * 0.20 / 4), "carbs": round(kcal * 0.50 / 4),
            "fats": round(kcal * 0.30 / 9)}


def tips(category: str) -> list:
    return _TIPS.get(category, [])
