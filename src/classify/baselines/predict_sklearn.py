"""
Load any scikit-learn text pipeline (TF-IDF + LogReg or TF-IDF + NB) and predict a single input string.

CLI usage:
  python src/classify/baselines/predict_sklearn.py models/cls_tfidf_nb "free text here"
  python src/classify/baselines/predict_sklearn.py models/cls_tfidf_logreg "free text here"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from joblib import load


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return e / e.sum()


def predict_sklearn(model_dir: str | Path, text: str) -> Dict[str, Any]:
    """
    Server-friendly function:
      - loads model.joblib and label_encoder.json from model_dir
      - returns {"label": <str>, "probs": {class: prob}}
    """
    model_dir = Path(model_dir)

    pipe = load(model_dir / "model.joblib")
    with open(model_dir / "label_encoder.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    # Prefer predict_proba if available; otherwise approximate from decision_function
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba([text])[0]
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function([text])[0]
        probs = softmax(np.array(scores))
    else:
        # Last resort: hard prediction
        yhat = pipe.predict([text])[0]
        probs = np.zeros(len(classes), dtype=float)
        probs[int(yhat)] = 1.0

    top_idx = int(np.argmax(probs))
    label = classes[top_idx]

    return {
        "label": label,
        "probs": {cls: float(probs[i]) for i, cls in enumerate(classes)},
    }


def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: predict_sklearn.py <model_dir> "text to classify"')
        sys.exit(1)

    model_dir = sys.argv[1]
    text = sys.argv[2]
    out = predict_sklearn(model_dir, text)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()