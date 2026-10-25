"""NutriLens - Food Recognition & Nutrition Tracking website (Member 4).
Run:  streamlit run app.py
"""
import hashlib
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

import evaluation as ev
import guidance as gd
import logger as lg
import model_wrapper as mw
import nutrition as nu

st.set_page_config(page_title="NutriLens", page_icon="🍽️", layout="wide")
ss = st.session_state
if "profile" not in ss:
    ss.profile = lg.load_profile()
ss.setdefault("pred", None)
ss.setdefault("eval_runs", {})

CONF_THRESHOLD = 0.50
MEALS = ["Breakfast", "Lunch", "Snack", "Dinner"]


@st.cache_resource
def load_model_once():
    return mw.load()


real_model = load_model_once()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("System status")
    if real_model:
        st.success("Trained model loaded")
    else:
        st.warning(f"Mock mode: {mw.load_error}")
    classes = mw.class_names()
    with st.expander("Integration check"):
        missing = nu.missing_foods(classes)
        st.write(f"**{len(classes)}** food classes")
        if missing:
            st.error(f"{len(missing)} classes have no nutrition data:")
            st.write(", ".join(missing))
        else:
            st.success("Every class has nutrition data")
    with st.expander("About & limitations"):
        st.markdown(
            "- Nutrition values are **approximate** and vary by recipe.\n"
            "- Portion size is chosen by the user; the model does not estimate it.\n"
            "- Similar-looking dishes can be confused, and unknown foods may be misclassified.\n"
            "- Lighting and background can affect accuracy.\n\n" + gd.DISCLAIMER)

st.title("🍽️ NutriLens")
st.caption("Food recognition, nutrition tracking and personalised calorie goals")

tab_profile, tab_predict, tab_dash, tab_eval = st.tabs(
    ["👤 Profile", "📷 Predict & Log", "📊 Dashboard", "🧪 Model Evaluation"])

# ------------------------------------------------------------------ PROFILE
with tab_profile:
    p = ss.profile or {}
    acts, goals = list(nu.ACTIVITY), list(nu.GOALS)
    with st.form("profile_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 10, 100, int(p.get("age", 20)))
        gender = c2.selectbox("Gender", ["Female", "Male"],
                              index=["Female", "Male"].index(p.get("gender", "Female")))
        activity = c3.selectbox("Activity level", acts,
                                index=acts.index(p.get("activity", acts[0])))
        c4, c5, c6 = st.columns(3)
        height = c4.number_input("Height (cm)", 100.0, 250.0, float(p.get("height", 165.0)))
        weight = c5.number_input("Weight (kg)", 25.0, 300.0, float(p.get("weight", 60.0)))
        goal = c6.selectbox("Goal", goals, index=goals.index(p.get("goal", goals[1])))
        if st.form_submit_button("Save profile", type="primary"):
            bmi = nu.calculate_bmi(weight, height)
            ss.profile = {"age": age, "gender": gender, "activity": activity,
                          "height": height, "weight": weight, "goal": goal, "bmi": bmi,
                          "category": nu.bmi_category(bmi),
                          "target": nu.recommend_calories(age, gender, weight, height, activity, goal)}
            lg.save_profile(ss.profile)

    pr = ss.profile
    if pr:
        m1, m2, m3 = st.columns(3)
        m1.metric("BMI", pr["bmi"])
        m2.metric("BMI category", pr["category"])
        m3.metric("Daily calorie target", f'{pr["target"]} kcal')
        mt = gd.macro_targets(pr["target"])
        st.markdown(f'**Suggested daily macros:** protein {mt["protein"]} g · '
                    f'carbs {mt["carbs"]} g · fats {mt["fats"]} g')
        suggested = gd.SUGGESTED_GOAL[pr["category"]]
        if suggested != pr["goal"]:
            st.info(f'Based on your BMI category, a "{suggested}" goal is usually suggested.')
        st.markdown("**General guidance**")
        for t in gd.tips(pr["category"]):
            st.markdown(f"- {t}")
        st.caption(gd.DISCLAIMER)
    else:
        st.info("Enter your details and save to get your BMI and daily calorie target.")

# ------------------------------------------------------------------ PREDICT & LOG
with tab_predict:
    src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    file = (st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png", "webp"])
            if src == "Upload" else st.camera_input("Take a photo"))

    if file is None:
        ss.pred = None
        st.info("Upload a photo of a meal to get started.")
    else:
        img = Image.open(file)
        key = hashlib.md5(file.getvalue()).hexdigest()
        left, right = st.columns([1, 1.3])
        left.image(img, caption="Your image")
        with right:
            if st.button("🔍 Predict food", type="primary"):
                ss.pred = {"key": key, "preds": mw.predict_food(img)}
            pred = ss.pred if ss.pred and ss.pred["key"] == key else None

            if pred is None:
                st.write("Press **Predict food** to analyse the image.")
            else:
                top_label, top_conf = pred["preds"][0]
                st.subheader(f"{nu.display_name(top_label)} ({top_conf:.0%})")
                for label, conf in pred["preds"]:
                    st.progress(min(conf, 1.0), text=f"{nu.display_name(label)} - {conf:.0%}")
                if top_conf < CONF_THRESHOLD:
                    st.warning("Low confidence. Please check or correct the food below.")

                options = sorted({nu.normalize(c) for c in classes})
                default = options.index(nu.normalize(top_label)) if nu.normalize(top_label) in options else 0
                food = st.selectbox("Confirm or correct the food", options, index=default,
                                    format_func=nu.display_name)
                if nu.missing_foods([food]):
                    st.error("No nutrition data for this food yet (tell Member 3).")
                else:
                    c1, c2 = st.columns(2)
                    servings = c1.number_input("Servings", 0.25, 10.0, 1.0, step=0.25)
                    meal = c2.selectbox("Meal", MEALS)
                    st.caption(f"1 serving = {nu.serving_description(food)}")
                    n = nu.get_nutrition(food, servings)
                    a, b, c, d = st.columns(4)
                    a.metric("Calories", f'{n["calories"]:.0f} kcal')
                    b.metric("Protein", f'{n["protein"]} g')
                    c.metric("Carbs", f'{n["carbs"]} g')
                    d.metric("Fat", f'{n["fats"]} g')
                    if st.button("➕ Add to Daily Log"):
                        lg.add_entry(meal, food, servings, n)
                        st.success(f"Added {servings:g} x {nu.display_name(food)} to {meal}.")

# ------------------------------------------------------------------ DASHBOARD
with tab_dash:
    day = st.date_input("Date", value=date.today(), max_value=date.today())
    entries = lg.entries_for(day)
    tot = lg.totals(entries)
    target = ss.profile["target"] if ss.profile else None
    mt = gd.macro_targets(target) if target else None

    k1, k2, k3 = st.columns(3)
    k1.metric("Calories eaten", f'{tot["calories"]:.0f} kcal')
    if target:
        k2.metric("Daily target", f"{target} kcal")
        k3.metric("Remaining", f'{target - tot["calories"]:.0f} kcal')
        st.progress(min(tot["calories"] / target, 1.0),
                    text=f'{tot["calories"] / target:.0%} of daily target')
        if tot["calories"] > target:
            st.warning("You are over your calorie target for this day.")
        st.markdown("**Macro progress**")
        mc = st.columns(3)
        for col, (name, key) in zip(mc, [("Protein", "protein"), ("Carbs", "carbs"), ("Fats", "fats")]):
            col.progress(min(tot[key] / mt[key], 1.0) if mt[key] else 0.0,
                         text=f"{name}: {tot[key]:.0f} / {mt[key]} g")
    else:
        st.info("Save your profile to see progress toward your calorie target.")

    if entries.empty:
        st.info("No meals logged for this day.")
    else:
        c1, c2 = st.columns(2)
        macros = pd.DataFrame({"Macro": ["Protein", "Carbs", "Fats"],
                               "Grams": [tot["protein"], tot["carbs"], tot["fats"]]})
        c1.markdown("**Macro split (grams)**")
        c1.altair_chart(alt.Chart(macros).mark_arc(innerRadius=55).encode(
            theta="Grams:Q", color="Macro:N", tooltip=["Macro", "Grams"]
        ).properties(width="container", height=250))
        c2.markdown("**Calories by meal**")
        c2.altair_chart(alt.Chart(lg.calories_by_meal(entries, MEALS)).mark_bar().encode(
            x=alt.X("meal:N", sort=MEALS, title=None), y=alt.Y("calories:Q", title="kcal"),
            tooltip=["meal", "calories"]
        ).properties(width="container", height=250))

        st.markdown("**Meals logged**")
        view = entries.copy()
        view["food"] = view["food"].map(nu.display_name)
        st.dataframe(view[["time", "meal", "food", "servings", "calories", "protein", "carbs", "fats"]],
                     hide_index=True)
        with st.expander("Remove an entry"):
            allr = lg.load_log()
            idx = st.selectbox("Entry", list(entries.index), format_func=lambda i:
                               f'{allr.loc[i, "time"]} - {allr.loc[i, "meal"]} - {nu.display_name(allr.loc[i, "food"])}')
            if st.button("Delete entry"):
                lg.delete_entry(idx)
                st.rerun()

    st.markdown("**Last 7 days (calories)**")
    week = lg.last_7_days()
    chart = alt.Chart(week).mark_bar().encode(
        x=alt.X("date:N", sort=None, title=None), y=alt.Y("calories:Q", title="kcal"),
        tooltip=["date", "calories"])
    if target:
        chart = chart + alt.Chart(pd.DataFrame({"t": [target]})).mark_rule(
            color="red", strokeDash=[5, 4]).encode(y="t:Q")
    st.altair_chart(chart.properties(width="container", height=250))

# ------------------------------------------------------------------ EVALUATION
with tab_eval:
    st.subheader("Model evaluation & error analysis")
    st.caption("Needs TensorFlow, a trained model, and a test folder laid out as "
               "`<test folder>/<class_name>/image.jpg`. Run once per model to compare them.")
    c1, c2 = st.columns(2)
    test_dir = c1.text_input("Test folder", "data/test")
    model_path = c2.text_input("Model file", mw.MODEL_PATH)
    c3, c4, c5 = st.columns(3)
    run_name = c3.text_input("Run name", "MobileNetV2")
    mode = c4.selectbox("Preprocessing", ["mobilenetv2", "rescale", "none"],
                        index=["mobilenetv2", "rescale", "none"].index(mw.PREPROCESS))
    max_pc = c5.number_input("Max images per class (0 = all)", 0, 5000, 0)

    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Evaluating...")
        try:
            names = mw.class_names()
            res = ev.run_evaluation(test_dir, model_path, names, mode, int(max_pc),
                                    progress=lambda f: bar.progress(f, text="Evaluating..."))
            ss.eval_runs[run_name] = {"metrics": ev.compute_metrics(res, names),
                                      "classes": names, "missing": res["missing_classes"]}
            bar.empty()
        except Exception as e:
            bar.empty()
            st.error(f"Evaluation failed: {e}")

    runs = ss.eval_runs
    if runs:
        st.markdown("**Run comparison**")
        st.dataframe(pd.DataFrame([{"run": k, "images": v["metrics"]["n_images"],
                                    "top-1 accuracy": round(v["metrics"]["accuracy"], 4),
                                    "top-3 accuracy": round(v["metrics"]["top3"], 4),
                                    "macro precision": round(v["metrics"]["macro_precision"], 4),
                                    "macro recall": round(v["metrics"]["macro_recall"], 4),
                                    "macro F1": round(v["metrics"]["macro_f1"], 4)}
                                   for k, v in runs.items()]), hide_index=True)
        chosen = st.selectbox("Inspect run", list(runs))
        R, names = runs[chosen]["metrics"], runs[chosen]["classes"]
        if runs[chosen]["missing"]:
            st.warning(f'No test folder for: {", ".join(runs[chosen]["missing"])}')

        st.markdown("**Per-class results (weakest first)**")
        st.dataframe(R["per_class"].round(3), hide_index=True)

        st.markdown("**Most confused class pairs**")
        if R["confusions"].empty:
            st.success("No misclassifications.")
        else:
            st.dataframe(R["confusions"].head(15), hide_index=True)

        if len(names) <= 30:
            cm = R["confusion"]
            cdf = pd.DataFrame([(names[i], names[j], int(cm[i, j]))
                                for i in range(len(names)) for j in range(len(names))],
                               columns=["true", "predicted", "count"])
            st.markdown("**Confusion matrix**")
            st.altair_chart(alt.Chart(cdf).mark_rect().encode(
                x=alt.X("predicted:N", sort=names), y=alt.Y("true:N", sort=names),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["true", "predicted", "count"]).properties(width="container", height=450))

        st.markdown("**Misclassified examples (most confident mistakes first)**")
        mis = R["misclassified"].head(12)
        if mis.empty:
            st.success("No misclassified images.")
        else:
            cols = st.columns(4)
            for i, row in mis.iterrows():
                with cols[i % 4]:
                    st.image(row["path"], caption=f'true: {nu.display_name(row["true"])}\n'
                             f'pred: {nu.display_name(row["predicted"])} ({row["confidence"]:.0%})')
