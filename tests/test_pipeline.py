from pathlib import Path

import src.server.pipeline as pipeline


def test_pipeline_mobile_contract_distilbert(monkeypatch, tmp_path):
    # Create active distilbert directory with a dummy file so resolver uses it
    active = tmp_path / "storage/models_registry/active/distilbert"
    active.mkdir(parents=True, exist_ok=True)
    (active / "dummy.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(pipeline, "Path", lambda p: Path(str(p)).__class__(p) if False else Path)  # no-op safeguard
    # Patch active path resolution by temporarily chdir-style logic:
    # easiest: monkeypatch _active_dir_for to point to temp active root.
    monkeypatch.setattr(pipeline, "_active_dir_for", lambda m: tmp_path / f"storage/models_registry/active/{m}")

    # Mock classifiers + summarizer
    monkeypatch.setattr(pipeline, "distilbert_predict", lambda model_dir, text: {"label": "Dermatology", "probs": {"Dermatology": 0.9, "Family Medicine": 0.1}})
    monkeypatch.setattr(pipeline, "predict_sklearn", lambda model_dir, text: {"label": "X", "probs": {"X": 0.7, "Y": 0.3}})
    monkeypatch.setattr(pipeline, "build_summary", lambda text: "SOAP TEXT")
    monkeypatch.setattr(pipeline, "run_qa", lambda text: [])

    out = pipeline.run_full_pipeline(text="hello", model="distilbert")
    assert out["model_used"] == "distilbert"
    assert out["prediction"] == "Dermatology"
    assert abs(out["confidence"] - 0.9) < 1e-9
    assert out["soap_summary"] == "SOAP TEXT"
    assert "classification" in out
    assert out["model_dir_used"].endswith("/distilbert") or out["model_dir_used"].endswith("\\distilbert")


def test_pipeline_uses_repo_fallback_when_active_empty(monkeypatch, tmp_path):
    # active exists but empty -> fallback
    active = tmp_path / "storage/models_registry/active/nb"
    active.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pipeline, "_active_dir_for", lambda m: tmp_path / f"storage/models_registry/active/{m}")

    monkeypatch.setattr(pipeline, "predict_sklearn", lambda model_dir, text: {"label": "Family Medicine", "probs": {"Family Medicine": 0.6, "Dermatology": 0.4}})
    monkeypatch.setattr(pipeline, "build_summary", lambda text: "SOAP TEXT")
    monkeypatch.setattr(pipeline, "run_qa", lambda text: [])

    out = pipeline.run_full_pipeline(text="hello", model="nb")
    assert out["model_used"] == "nb"
    assert out["prediction"] == "Family Medicine"
    assert abs(out["confidence"] - 0.6) < 1e-9
    # fallback should be repo model path
    assert out["model_dir_used"] == "models/cls_tfidf_nb"