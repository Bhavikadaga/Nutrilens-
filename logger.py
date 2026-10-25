"""Food log + saved profile (plain CSV / JSON files, survive app restarts)."""
import json
import os
from datetime import date, datetime, timedelta
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE, "data", "food_log.csv")
PROFILE_PATH = os.path.join(BASE, "data", "profile.json")
COLS = ["date", "time", "meal", "food", "servings", "calories", "protein", "carbs", "fats"]
NUM = ["servings", "calories", "protein", "carbs", "fats"]


def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(LOG_PATH).reindex(columns=COLS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.dropna(subset=["date"]).reset_index(drop=True)


def add_entry(meal: str, food: str, servings: float, nutrition: dict):
    now = datetime.now()
    row = {"date": now.date().isoformat(), "time": now.strftime("%H:%M"), "meal": meal,
           "food": food, "servings": servings, **nutrition}
    df = pd.concat([load_log(), pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def delete_entry(index: int):
    load_log().drop(index=index).to_csv(LOG_PATH, index=False)


def entries_for(day: date) -> pd.DataFrame:
    df = load_log()
    return df[df["date"] == day]


def totals(df: pd.DataFrame) -> dict:
    return {n: round(float(df[n].sum()), 1) for n in ("calories", "protein", "carbs", "fats")}


def calories_by_meal(df: pd.DataFrame, meals) -> pd.DataFrame:
    by = df.groupby("meal")["calories"].sum() if len(df) else pd.Series(dtype=float)
    return pd.DataFrame({"meal": list(meals), "calories": [float(by.get(m, 0)) for m in meals]})


def last_7_days() -> pd.DataFrame:
    df = load_log()
    days = [date.today() - timedelta(days=i) for i in range(6, -1, -1)]
    daily = df.groupby("date")["calories"].sum() if len(df) else pd.Series(dtype=float)
    return pd.DataFrame({"date": [d.strftime("%d %b") for d in days],
                         "calories": [float(daily.get(d, 0)) for d in days]})


def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH) as f:
            return json.load(f)
    return None


def save_profile(profile: dict):
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f)
