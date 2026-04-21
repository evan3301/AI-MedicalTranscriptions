from __future__ import annotations

from math import log1p
from typing import Dict, Any, List, Tuple


_MODEL_BIAS = {
    "nb": 0.95,         # tends to be overconfident
    "logreg": 1.05,     # often under-confident in multiclass settings
    "distilbert": 1.10, # often flatter softmax despite stronger features
}


def _sorted_probs(probs: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _normalize_top1(p1: float, n_classes: int) -> float:
    """
    Normalize top probability relative to many-class baseline.
    We do not want 10% in a 40-class task to look 'terrible' if uniform is only 2.5%.
    """
    if n_classes <= 1:
        return _clamp(p1)

    uniform = 1.0 / n_classes
    ratio = max(p1 / uniform, 1.0)
    # log scaling keeps it interpretable without exploding high values
    # ratio=1 -> 0, ratio ~6 -> ~1
    score = log1p(ratio - 1.0) / log1p(5.0)
    return _clamp(score)


def _normalize_margin(p1: float, p2: float, n_classes: int) -> float:
    margin = max(p1 - p2, 0.0)
    # in many-class tasks, even a small raw margin can matter
    scaled = margin * max(n_classes / 3.0, 1.0)
    return _clamp(scaled)


def compute_display_confidence(
    probs: Dict[str, float],
    model_used: str,
) -> Dict[str, Any]:
    if not probs:
        return {
            "display_confidence": 0.0,
            "raw_confidence": 0.0,
            "confidence_band": "Low",
            "top_alternatives": [],
            "margin": 0.0,
        }

    ordered = _sorted_probs(probs)
    n_classes = len(ordered)

    top_label, p1 = ordered[0]
    p2 = ordered[1][1] if len(ordered) > 1 else 0.0

    top1_signal = _normalize_top1(p1, n_classes)
    margin_signal = _normalize_margin(p1, p2, n_classes)

    # Weighted combination: top1 matters most, but margin prevents "flat" distributions
    base = (0.65 * top1_signal) + (0.35 * margin_signal)
    bias = _MODEL_BIAS.get(model_used, 1.0)

    display_confidence = _clamp(base * bias)
    raw_confidence = float(p1)
    margin = float(max(p1 - p2, 0.0))

    if display_confidence < 0.35:
        band = "Low"
    elif display_confidence < 0.70:
        band = "Moderate"
    else:
        band = "High"

    top_alternatives = [
        {"label": label, "probability": float(prob)}
        for label, prob in ordered[:3]
    ]

    return {
        "display_confidence": display_confidence,
        "raw_confidence": raw_confidence,
        "confidence_band": band,
        "top_alternatives": top_alternatives,
        "margin": margin,
    }