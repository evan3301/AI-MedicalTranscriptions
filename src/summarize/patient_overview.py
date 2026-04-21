from __future__ import annotations

from typing import Dict, Any


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def build_patient_overview(parsed_note: Dict[str, Any], soap_fields: Dict[str, str]) -> str:
    """
    Build a short contextual overview of the patient/note without duplicating:
    - diagnosis card
    - medications card
    - plan card

    Focus on:
    - presenting complaint / subjective context
    - duration / timeline if present
    - whether the note is brief / limited / outpatient-style
    """
    subjective = _clean(soap_fields.get("subjective"))
    objective = _clean(soap_fields.get("objective"))

    sections = parsed_note.get("sections", {}) or {}
    sentences = parsed_note.get("sentences", []) or []

    parts: list[str] = []

    # Main presenting context
    if subjective and subjective.lower() != "not clearly stated":
        parts.append(f"Patient presents with {subjective.lower()}.")
    elif sections.get("chief_complaint"):
        cc = _clean(sections.get("chief_complaint"))
        if cc:
            parts.append(f"Patient is being seen for {cc.lower()}.")
    else:
        parts.append("Patient is being evaluated in a brief clinical encounter.")

    # General note richness / context
    if objective and objective.lower() != "not clearly stated":
        parts.append("Some objective clinical detail is documented in the note.")
    else:
        parts.append("Only limited objective detail is documented.")

    # If the note is extremely short, mention that explicitly
    if len(sentences) <= 3:
        parts.append("Overall, the documentation appears concise and focused on the immediate visit.")
    else:
        parts.append("Overall, the note provides a short summary of the visit context.")

    return " ".join(parts)