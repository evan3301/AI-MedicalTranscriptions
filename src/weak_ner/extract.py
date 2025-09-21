import re
from typing import List, Dict, Tuple

DEMO_MEDS = ["aspirin", "albuterol", "atorvastatin", "metformin"]
DEMO_DX   = ["asthma", "hypertension", "diabetes", "myocardial infarction"]

def exact_matches(text: str, vocab: List[str], label: str, conf: float = 0.95) -> List[Dict]:
    hits = []
    for term in vocab:
        for m in re.finditer(rf"\b{re.escape(term)}\b", text, flags=re.I):
            hits.append({"text": m.group(), "label": label,
                         "span": (m.start(), m.end()), "confidence": conf})
    return hits

def extract_entities(text: str) -> List[Dict]:
    meds = exact_matches(text, DEMO_MEDS, "MED")
    dx   = exact_matches(text, DEMO_DX,   "DIAG")
    doses = [{"text": m.group(), "label": "DOSE",
              "span": (m.start(), m.end()), "confidence": 0.7}
             for m in re.finditer(r"\b\d+\s*(mg|mcg|g)\b", text, flags=re.I)]
    return meds + dx + doses

if __name__ == "__main__":
    s = "Patient on aspirin 81 mg daily for hypertension; uses albuterol PRN for asthma."
    print(extract_entities(s))
