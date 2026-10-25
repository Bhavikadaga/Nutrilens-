"""Wraps Member 2's trained model behind predict_food(image).
Until models/food_model.keras + models/class_names.json exist, MOCK predictions are used.

!! PREPROCESS must match how the model was trained (ask Member 2):
   "mobilenetv2" -> pixels scaled to [-1, 1]  (tf.keras.applications.mobilenet_v2.preprocess_input)
   "rescale"     -> pixels scaled to [0, 1]   (Rescaling(1/255), typical for a custom CNN)
   "none"        -> raw 0-255 (model already contains its own preprocessing layers)
"""
import hashlib
import json
import os
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "food_model.keras")
CLASSES_PATH = os.path.join(BASE, "models", "class_names.json")
IMG_SIZE = (224, 224)
PREPROCESS = "mobilenetv2"

_model = None
_classes = None
load_error = None


def preprocess(image: Image.Image, mode: str = None, size=None) -> np.ndarray:
    mode = mode or PREPROCESS
    img = image.convert("RGB").resize(size or IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    if mode == "mobilenetv2":
        arr = arr / 127.5 - 1.0
    elif mode == "rescale":
        arr = arr / 255.0
    return np.expand_dims(arr, 0)


def load() -> bool:
    """True if the real model is loaded, False if running in mock mode."""
    global _model, _classes, load_error
    if _model is not None:
        return True
    if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
        load_error = "model files not found in models/ (using mock predictions)"
        return False
    try:
        from tensorflow import keras
        _model = keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH) as f:
            _classes = json.load(f)
        load_error = None
        return True
    except Exception as e:  # TensorFlow missing, corrupt file, etc.
        load_error = f"could not load model: {e}"
        return False


def class_names():
    import nutrition
    return _classes if _classes else nutrition.available_foods()


def predict_food(image: Image.Image, top_k: int = 3):
    """-> [(label, confidence), ...] sorted by confidence (highest first)."""
    if load():
        probs = _model.predict(preprocess(image), verbose=0)[0]
        idx = np.argsort(probs)[::-1][:top_k]
        return [(_classes[i], float(probs[i])) for i in idx]
    names = class_names()                                   # ---- MOCK ----
    seed = int(hashlib.md5(image.tobytes()[:4000]).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(names), size=min(top_k, len(names)), replace=False)
    conf = np.sort(rng.dirichlet([6, 2, 1][:len(picks)]))[::-1]
    return [(names[p], float(c)) for p, c in zip(picks, conf)]
