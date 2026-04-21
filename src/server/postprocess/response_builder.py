from __future__ import annotations

from typing import Dict, Any, List

from .confidence import compute_display_confidence
from src.summarize.soap_builder import build_soap_fields
from src.summarize.patient_overview import build_patient_overview


def build_display_response(
    *,
    text: str,
    model_used: str,
    classification: Dict[str, Any],
    qa_issues: List[Dict[str, Any]] | None = None,
    model_dir_used: str | None = None,
) -> Dict[str, Any]:
    prediction = classification.get("label")
    probs = classification.get("probs", {}) or {}

    confidence = compute_display_confidence(probs, model_used)
    soap_bundle = build_soap_fields(text)

    soap = soap_bundle["soap"]
    parsed_note = soap_bundle["parsed_note"]

    patient_overview = build_patient_overview(parsed_note, soap)

    return {
        # app-facing fields
        "model_used": model_used,
        "prediction": prediction,
        "confidence": confidence["display_confidence"],
        "confidence_band": confidence["confidence_band"],
        "top_alternatives": confidence["top_alternatives"],
        "patient_overview": patient_overview,
        "soap_summary": soap_bundle["soap_text"],
        "soap": soap,

        # preserve backend detail
        "classification": classification,
        "qa": {"issues": qa_issues or []},
        "summary": {
            "soap_text": soap_bundle["soap_text"],
            "soap": soap,
        },

        # technical/debug detail
        "technical_details": {
            "raw_confidence": confidence["raw_confidence"],
            "margin": confidence["margin"],
            "model_dir_used": model_dir_used,
        },
    }