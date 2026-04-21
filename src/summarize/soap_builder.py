from __future__ import annotations

from typing import Dict, Any, List, Optional

from .note_parser import parse_note


def _clean_line(text: Optional[str]) -> str:
    if not text:
        return "Not clearly stated"
    cleaned = " ".join(text.strip().split())
    cleaned = cleaned.strip(" -:\n\t")
    return cleaned if cleaned else "Not clearly stated"


def _pick_best(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "Not clearly stated"
    return _clean_line(candidates[0]["text"])


def _format_medications(medications: List[Dict[str, str]]) -> str:
    if not medications:
        return "Not clearly stated"

    chunks = []
    for med in medications:
        name = med.get("name", "").strip()
        dose = med.get("dose", "").strip()
        if name and dose:
            chunks.append(f"{name} {dose}")
        elif name:
            chunks.append(name)

    if not chunks:
        return "Not clearly stated"

    return ", ".join(dict.fromkeys(chunks))


def build_soap_fields(text: str) -> Dict[str, Any]:
    parsed = parse_note(text)
    candidates = parsed["candidates"]

    assessment = _pick_best(candidates.get("assessment", []))
    plan = _pick_best(candidates.get("plan", []))
    medications = _format_medications(parsed.get("medications", []))

    subjective = _pick_best(candidates.get("subjective", []))
    objective = _pick_best(candidates.get("objective", []))

    soap = {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "medications": medications,
        "plan": plan,
    }

    soap_text = (
        "=== SOAP Summary ===\n"
        f"Subjective: {soap['subjective']}\n"
        f"Objective: {soap['objective']}\n"
        f"Assessment: {soap['assessment']}\n"
        f"Medications: {soap['medications']}\n"
        f"Plan: {soap['plan']}"
    )

    return {
        "soap": soap,
        "soap_text": soap_text,
        "parsed_note": parsed,
    }