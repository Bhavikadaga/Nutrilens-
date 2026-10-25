"""Model evaluation + error analysis (Member 4).
Test folder layout (same class names as models/class_names.json):
    data/test/<class_name>/img1.jpg ...
Command line:  python evaluation.py --test_dir data/test --model models/food_model.keras
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import model_wrapper as mw

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def collect_images(test_dir, class_names, max_per_class=0):
    items, missing = [], []
    for idx, cls in enumerate(class_names):
        folder = os.path.join(test_dir, cls)
        if not os.path.isdir(folder):
            missing.append(cls)
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
        if max_per_class:
            files = files[:max_per_class]
        items += [(os.path.join(folder, f), idx) for f in files]
    return items, missing


def run_evaluation(test_dir, model_path, class_names, preprocess_mode="mobilenetv2",
                   max_per_class=0, batch_size=32, progress=None):
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    shape = model.input_shape
    size = (shape[2], shape[1]) if shape and len(shape) == 4 and shape[1] else mw.IMG_SIZE
    items, missing = collect_images(test_dir, class_names, max_per_class)
    if not items:
        raise ValueError(f"No test images found in '{test_dir}'. Expected {test_dir}/<class_name>/*.jpg")
    out = {"y_true": [], "y_pred": [], "top3": [], "conf": [], "paths": [],
           "missing_classes": missing, "image_size": size}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        x = np.concatenate([mw.preprocess(Image.open(p), preprocess_mode, size) for p, _ in batch])
        probs = model.predict(x, verbose=0)
        if probs.shape[1] != len(class_names):
            raise ValueError(f"Model outputs {probs.shape[1]} classes but class list has {len(class_names)}.")
        for (path, true), pr in zip(batch, probs):
            order = np.argsort(pr)[::-1]
            out["y_true"].append(true)
            out["y_pred"].append(int(order[0]))
            out["top3"].append(bool(true in order[:3]))
            out["conf"].append(float(pr[order[0]]))
            out["paths"].append(path)
        if progress:
            progress(min(1.0, (start + len(batch)) / len(items)))
    return out


def compute_metrics(res, class_names):
    from sklearn.metrics import classification_report, confusion_matrix
    y_true, y_pred = np.array(res["y_true"]), np.array(res["y_pred"])
    labels = sorted(set(y_true.tolist()))
    names = [class_names[i] for i in labels]
    rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                output_dict=True, zero_division=0)
    per_class = pd.DataFrame([{"class": n, "precision": rep[n]["precision"],
                               "recall": rep[n]["recall"], "f1": rep[n]["f1-score"],
                               "support": int(rep[n]["support"])} for n in names])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pairs = [(class_names[i], class_names[j], int(cm[i, j]))
             for i in range(len(class_names)) for j in range(len(class_names))
             if i != j and cm[i, j] > 0]
    confusions = pd.DataFrame(pairs, columns=["true", "predicted", "count"]) \
        .sort_values("count", ascending=False).reset_index(drop=True)
    wrong = y_true != y_pred
    mis = pd.DataFrame({"path": np.array(res["paths"])[wrong],
                        "true": [class_names[i] for i in y_true[wrong]],
                        "predicted": [class_names[i] for i in y_pred[wrong]],
                        "confidence": np.array(res["conf"])[wrong]}) \
        .sort_values("confidence", ascending=False).reset_index(drop=True)
    return {"accuracy": float((y_true == y_pred).mean()), "top3": float(np.mean(res["top3"])),
            "macro_precision": rep["macro avg"]["precision"], "macro_recall": rep["macro avg"]["recall"],
            "macro_f1": rep["macro avg"]["f1-score"], "n_images": int(len(y_true)),
            "per_class": per_class.sort_values("f1").reset_index(drop=True),
            "confusion": cm, "confusions": confusions, "misclassified": mis}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="data/test")
    ap.add_argument("--model", default=mw.MODEL_PATH)
    ap.add_argument("--classes", default=mw.CLASSES_PATH)
    ap.add_argument("--preprocess", default=mw.PREPROCESS)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    classes = json.load(open(a.classes))
    res = run_evaluation(a.test_dir, a.model, classes, a.preprocess)
    m = compute_metrics(res, classes)
    os.makedirs(a.out, exist_ok=True)
    m["per_class"].to_csv(os.path.join(a.out, "per_class.csv"), index=False)
    m["confusions"].to_csv(os.path.join(a.out, "top_confusions.csv"), index=False)
    m["misclassified"].to_csv(os.path.join(a.out, "misclassified.csv"), index=False)
    print(f"Images: {m['n_images']}  Top-1: {m['accuracy']:.3f}  Top-3: {m['top3']:.3f}  Macro-F1: {m['macro_f1']:.3f}")
    print(f"Reports saved to {a.out}/")
