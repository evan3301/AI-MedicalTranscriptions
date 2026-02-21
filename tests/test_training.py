import json
from pathlib import Path

import pytest

from src.server.training import run_training_job


def test_training_nb_creates_artifacts_and_activates(tmp_path, sample_dataset_csv):
    training_runs = tmp_path / "training_runs"
    models_runs = tmp_path / "models_runs"
    active_root = tmp_path / "active"

    result = run_training_job(
        dataset_csv=sample_dataset_csv,
        run_id="run_nb",
        model="nb",
        storage_training_runs_dir=training_runs,
        models_registry_runs_dir=models_runs,
        models_active_root=active_root,
    )

    run_dir = Path(result["outdir"])
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "label_encoder.json").exists()
    assert (run_dir / "val_report.json").exists()

    active_dir = Path(result["active_dir"])
    assert active_dir.exists()
    assert any(active_dir.iterdir())


def test_training_logreg_creates_artifacts_and_activates(tmp_path, sample_dataset_csv):
    training_runs = tmp_path / "training_runs"
    models_runs = tmp_path / "models_runs"
    active_root = tmp_path / "active"

    result = run_training_job(
        dataset_csv=sample_dataset_csv,
        run_id="run_lr",
        model="logreg",
        storage_training_runs_dir=training_runs,
        models_registry_runs_dir=models_runs,
        models_active_root=active_root,
    )

    run_dir = Path(result["outdir"])
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "label_encoder.json").exists()
    assert (run_dir / "val_report.json").exists()

    active_dir = Path(result["active_dir"])
    assert active_dir.exists()
    assert any(active_dir.iterdir())


def test_training_distilbert_is_routable_with_mock(monkeypatch, tmp_path, sample_dataset_csv):
    """
    Avoid heavy downloads/training. We just ensure distilbert route writes + activates.
    """
    training_runs = tmp_path / "training_runs"
    models_runs = tmp_path / "models_runs"
    active_root = tmp_path / "active"

    # Patch the imported train_distilbert function inside src.server.training module.
    import src.server.training as training_mod

    def fake_train(cfg):
        outdir = Path(cfg.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # Create dummy "model artifacts"
        (outdir / "config.json").write_text("{}", encoding="utf-8")
        (outdir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (outdir / "pytorch_model.bin").write_text("dummy", encoding="utf-8")
        return {"best_val_macro_f1": 0.5, "test_macro_f1": 0.49}

    monkeypatch.setattr(training_mod, "train_distilbert", fake_train)

    result = run_training_job(
        dataset_csv=sample_dataset_csv,
        run_id="run_db",
        model="distilbert",
        storage_training_runs_dir=training_runs,
        models_registry_runs_dir=models_runs,
        models_active_root=active_root,
    )

    run_dir = Path(result["outdir"])
    assert (run_dir / "pytorch_model.bin").exists()

    active_dir = Path(result["active_dir"])
    assert active_dir.exists()
    assert any(active_dir.iterdir())