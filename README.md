# NutriLens - Food Recognition & Nutrition Tracking (Streamlit website)

## Run
    python -m venv venv
    venv\Scripts\activate            (Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt
    streamlit run app.py

## What each teammate must deliver
| Who | File(s) | Contract |
|---|---|---|
| Member 2 | `models/food_model.keras`, `models/class_names.json` | Class list in the SAME order as the model outputs. Tell Bhavika the input size and preprocessing (MobileNetV2 -> `mobilenetv2`; /255 -> `rescale`). Set `IMG_SIZE` / `PREPROCESS` in `model_wrapper.py`. |
| Member 3 | `data/nutrition.csv` | Columns: `food,calories,protein,carbs,fats` (+ optional `serving_desc`), values per ONE serving. Food names must match class names (case/space/underscore insensitive). Also owns `nutrition.py`. |
| Member 1 | `data/test/<class_name>/*.jpg` | Held-out test split used by the Model Evaluation tab. |

The sidebar "Integration check" lists any model class that has no nutrition row.

## Files
- `app.py` - website (Profile, Predict & Log, Dashboard, Model Evaluation)
- `model_wrapper.py` - loads model, `predict_food(image)` (mock until the model exists)
- `nutrition.py` - nutrition lookup, BMI, calorie target (Member 3's stub)
- `guidance.py` - macro targets + general tips
- `logger.py` - food log (`data/food_log.csv`) and saved profile
- `evaluation.py` - metrics, confusion matrix, error analysis (also runnable from terminal)

Sample nutrition values in `data/nutrition_sample.csv` are placeholders only.
