# src/server/training.py

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classify.train_distilbert import Config, train as train_distilbert
from src.classify.baselines.train_tfidf_nb import train_nb
from src.classify.baselines.train_tfidf_logreg import train_logreg


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Stable, demo-friendly. Avoid stratify crashes on rare labels.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=13, shuffle=True)
    return train_df, val_df


def activate_model(run_model_dir: Path, active_model_dir: Path) -> None:
    # Replace active model dir with new artifacts
    if active_model_dir.exists():
        shutil.rmtree(active_model_dir)
    shutil.copytree(run_model_dir, active_model_dir)


def run_training_job(
    *,
    dataset_csv: Path,
    run_id: str,
    model: str,
    storage_training_runs_dir: Path,
    models_registry_runs_dir: Path,
    models_active_root: Path,  # storage/models_registry/active
) -> Dict[str, Any]:
    """
    Train the selected model family (nb/logreg/distilbert) on the uploaded dataset.
    Writes artifacts under models_registry_runs_dir/<run_id>/<model> and activates to active/<model>.
    """
    model = model.lower().strip()
    if model not in {"nb", "logreg", "distilbert"}:
        raise ValueError("model must be one of: nb, logreg, distilbert")

    df = pd.read_csv(dataset_csv)
    if "text" not in df.columns or "specialty" not in df.columns:
        raise ValueError("CSV must contain columns: text, specialty")

    run_data_dir = storage_training_runs_dir / run_id
    run_data_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = split_train_val(df)

    train_csv = run_data_dir / "train.csv"
    val_csv = run_data_dir / "val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    run_model_dir = models_registry_runs_dir / run_id / model
    run_model_dir.mkdir(parents=True, exist_ok=True)

    # Train chosen model family
    if model == "nb":
        metrics = train_nb(train_csv=train_csv, val_csv=val_csv, outdir=run_model_dir)
        cfg_obj = {"train_csv": str(train_csv), "val_csv": str(val_csv), "outdir": str(run_model_dir)}
    elif model == "logreg":
        metrics = train_logreg(train_csv=train_csv, val_csv=val_csv, outdir=run_model_dir)
        cfg_obj = {"train_csv": str(train_csv), "val_csv": str(val_csv), "outdir": str(run_model_dir)}
    else:
        cfg = Config(outdir=str(run_model_dir), train_csv=str(train_csv), val_csv=str(val_csv), test_csv=str(val_csv))
        metrics = train_distilbert(cfg)
        cfg_obj = asdict(cfg)

    # Activate model for that family
    active_model_dir = models_active_root / model
    activate_model(run_model_dir, active_model_dir)

    return {
        "run_id": run_id,
        "model": model,
        "dataset_csv": str(dataset_csv),
        "outdir": str(run_model_dir),
        "active_dir": str(active_model_dir),
        "metrics": metrics,
        "config": cfg_obj,
    }