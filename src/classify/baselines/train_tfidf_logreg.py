# src/classify/baselines/train_tfidf_logreg.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def _load_csv(path: str | Path) -> Tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    if "text" not in df.columns or "specialty" not in df.columns:
        raise ValueError("CSV must contain columns: text, specialty")
    X = df["text"].astype(str).tolist()
    y = df["specialty"].astype(str).tolist()
    return X, y


def train_logreg(*, train_csv: str | Path, outdir: str | Path, val_csv: str | Path | None = None) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _load_csv(train_csv)

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )

    pipe.fit(X_train, y_train)

    classes = sorted(set(y_train))
    (outdir / "label_encoder.json").write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    dump(pipe, outdir / "model.joblib")

    metrics: Dict[str, Any] = {"model": "logreg", "train_rows": len(y_train)}

    if val_csv is not None:
        X_val, y_val = _load_csv(val_csv)
        y_pred = pipe.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        (outdir / "val_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        metrics["val_macro_f1"] = float(report["macro avg"]["f1-score"])
        metrics["val_accuracy"] = float(report["accuracy"])

    return metrics


def main() -> None:
    import sys
    if len(sys.argv) < 3:
        print("Usage: train_tfidf_logreg.py <train_csv> <outdir> [val_csv]")
        raise SystemExit(1)

    train_csv = sys.argv[1]
    outdir = sys.argv[2]
    val_csv = sys.argv[3] if len(sys.argv) >= 4 else None

    m = train_logreg(train_csv=train_csv, outdir=outdir, val_csv=val_csv)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()