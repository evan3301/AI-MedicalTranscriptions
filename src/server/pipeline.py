from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.classify.predict import predict as distilbert_predict
from src.classify.baselines.predict_sklearn import predict_sklearn
from src.qa.rules import run_qa
from src.server.postprocess.response_builder import build_display_response

SUPPORTED_MODELS = {"distilbert", "logreg", "nb"}


def _active_dir_for(model: str) -> Path:
    return Path("storage/models_registry/active") / model


def _resolve_model_dir(model: str) -> str:
    """
    Prefer server-trained active model if present (and non-empty), else fall back to repo models/.
    """
    active = _active_dir_for(model)
    try:
        if active.exists() and any(active.iterdir()):
            return str(active)
    except Exception:
        pass

    if model == "distilbert":
        return "models/cls_distilbert"
    if model == "logreg":
        return "models/cls_tfidf_logreg"
    return "models/cls_tfidf_nb"


def run_full_pipeline(*, text: str, model: str = "distilbert") -> Dict[str, Any]:
    model = (model or "distilbert").lower().strip()
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model}'. Choose one of: {sorted(SUPPORTED_MODELS)}")

    model_dir = _resolve_model_dir(model)

    # Classification
    if model == "distilbert":
        classification = distilbert_predict(model_dir, text)
    else:
        classification = predict_sklearn(model_dir, text)

    # QA
    qa_issues = run_qa(text)

    # Display-ready response
    return build_display_response(
        text=text,
        model_used=model,
        classification=classification,
        qa_issues=qa_issues,
        model_dir_used=model_dir,
    )