from __future__ import annotations

import re
from typing import Dict, List, Any

# Canonical section names -> header aliases
_SECTION_ALIASES = {
    "chief_complaint": ["chief complaint", "cc"],
    "hpi": ["hpi", "history of present illness", "history"],
    "subjective": ["subjective"],
    "objective": ["objective", "exam", "physical exam", "findings", "vitals"],
    "assessment": ["assessment", "impression", "diagnosis", "dx"],
    "plan": ["plan", "recommendation", "recommendations", "treatment", "follow up", "follow-up"],
    "medications": ["medications", "meds", "medication"],
}

_ALL_ALIASES = [alias for aliases in _SECTION_ALIASES.values() for alias in aliases]
_ALL_HEADERS_PATTERN = "|".join(sorted((re.escape(x) for x in _ALL_ALIASES), key=len, reverse=True))

_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?])\s+|\n+")

_MED_DOSE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\-]{2,}(?:\s+[A-Za-z][A-Za-z\-]{2,}){0,2})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|mL|units?|tabs?|caps?)\b",
    re.IGNORECASE,
)

_VITALS_RE = re.compile(
    r"\b(?:T(?:emp)?\s*[:=]?\s*\d+(?:\.\d+)?|HR\s*[:=]?\s*\d+|RR\s*[:=]?\s*\d+|BP\s*[:=]?\s*\d+/\d+|SpO2\s*[:=]?\s*\d+%?)\b",
    re.IGNORECASE,
)

_ASSESSMENT_CUES = [
    "assessment",
    "impression",
    "diagnosis",
    "dx",
    "likely",
    "consistent with",
    "concerning for",
    "suspicious for",
    "viral",
    "bacterial",
    "pneumonia",
    "uri",
    "infection",
    "bronchitis",
    "hypertension",
    "diabetes",
    "dermatitis",
]

_PLAN_CUES = [
    "plan",
    "start",
    "continue",
    "prescribe",
    "recommend",
    "advised",
    "advice",
    "rest",
    "fluids",
    "follow up",
    "follow-up",
    "return if",
    "monitor",
    "recheck",
    "referral",
]

_SUBJECTIVE_CUES = [
    "chief complaint",
    "complains of",
    "reports",
    "states",
    "history of present illness",
    "hpi",
    "pain",
    "cough",
    "fever",
    "sob",
    "shortness of breath",
    "nausea",
    "vomiting",
]

_OBJECTIVE_CUES = [
    "exam",
    "vitals",
    "bp",
    "hr",
    "rr",
    "spo2",
    "temp",
    "temperature",
    "wheez",
    "rales",
    "clear to auscultation",
]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip(" -:\t\n") for p in parts if p and p.strip(" -:\t\n")]


def _extract_sections(text: str) -> Dict[str, str]:
    """
    Extract explicit inline/section content, e.g.
    'Assessment: viral URI. Plan: rest and fluids. Medications: aspirin 81 mg.'
    """
    sections: Dict[str, List[str]] = {k: [] for k in _SECTION_ALIASES}

    for canonical, aliases in _SECTION_ALIASES.items():
        for alias in aliases:
            pattern = re.compile(
                rf"(?is)\b{re.escape(alias)}\b\s*[:\-]\s*(.+?)(?=(?:\b(?:{_ALL_HEADERS_PATTERN})\b\s*[:\-])|$)"
            )
            for m in pattern.finditer(text):
                value = m.group(1).strip(" \n\t.-")
                if value:
                    sections[canonical].append(value)

    collapsed: Dict[str, str] = {}
    for key, values in sections.items():
        if values:
            collapsed[key] = " ".join(dict.fromkeys(values))
    return collapsed


def _score_sentence(sentence: str, cues: List[str]) -> int:
    s = sentence.lower()
    score = 0
    for cue in cues:
        if cue in s:
            score += 1
    return score


def _collect_candidates(sentences: List[str], sections: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    assessment_candidates: List[Dict[str, Any]] = []
    plan_candidates: List[Dict[str, Any]] = []
    subjective_candidates: List[Dict[str, Any]] = []
    objective_candidates: List[Dict[str, Any]] = []

    if "assessment" in sections:
        assessment_candidates.append({"text": sections["assessment"], "score": 100, "source": "section"})
    if "plan" in sections:
        plan_candidates.append({"text": sections["plan"], "score": 100, "source": "section"})
    if "subjective" in sections:
        subjective_candidates.append({"text": sections["subjective"], "score": 100, "source": "section"})
    if "objective" in sections:
        objective_candidates.append({"text": sections["objective"], "score": 100, "source": "section"})
    if "hpi" in sections and "subjective" not in sections:
        subjective_candidates.append({"text": sections["hpi"], "score": 90, "source": "section"})
    if "chief_complaint" in sections and "subjective" not in sections:
        subjective_candidates.append({"text": sections["chief_complaint"], "score": 80, "source": "section"})

    for sent in sentences:
        assessment_score = _score_sentence(sent, _ASSESSMENT_CUES)
        plan_score = _score_sentence(sent, _PLAN_CUES)
        subjective_score = _score_sentence(sent, _SUBJECTIVE_CUES)
        objective_score = _score_sentence(sent, _OBJECTIVE_CUES)
        if _VITALS_RE.search(sent):
            objective_score += 2

        if assessment_score > 0:
            assessment_candidates.append({"text": sent, "score": assessment_score, "source": "sentence"})
        if plan_score > 0:
            plan_candidates.append({"text": sent, "score": plan_score, "source": "sentence"})
        if subjective_score > 0:
            subjective_candidates.append({"text": sent, "score": subjective_score, "source": "sentence"})
        if objective_score > 0:
            objective_candidates.append({"text": sent, "score": objective_score, "source": "sentence"})

    # Sort descending by score, preserve text uniqueness
    def _uniq_sorted(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        seen = set()
        out = []
        for item in items:
            key = item["text"].strip().lower()
            if key not in seen:
                out.append(item)
                seen.add(key)
        return out

    return {
        "assessment": _uniq_sorted(assessment_candidates),
        "plan": _uniq_sorted(plan_candidates),
        "subjective": _uniq_sorted(subjective_candidates),
        "objective": _uniq_sorted(objective_candidates),
    }


def _extract_medications(text: str, sections: Dict[str, str]) -> List[Dict[str, str]]:
    meds_source = sections.get("medications", text)
    matches = _MED_DOSE_RE.findall(meds_source)

    meds: List[Dict[str, str]] = []
    seen = set()

    for name, dose, unit in matches:
        cleaned_name = " ".join(name.strip().split())
        cleaned = {
            "name": cleaned_name,
            "dose": f"{dose} {unit}".replace("mL", "ml"),
        }
        key = (cleaned["name"].lower(), cleaned["dose"].lower())
        if key not in seen:
            meds.append(cleaned)
            seen.add(key)

    return meds


def parse_note(text: str) -> Dict[str, Any]:
    text = _normalize_text(text)
    sections = _extract_sections(text)
    sentences = _split_sentences(text)
    candidates = _collect_candidates(sentences, sections)
    medications = _extract_medications(text, sections)

    vitals = [s for s in sentences if _VITALS_RE.search(s)]

    return {
        "normalized_text": text,
        "sections": sections,
        "sentences": sentences,
        "candidates": candidates,
        "medications": medications,
        "vitals": vitals,
    }