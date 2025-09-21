from typing import Dict, List
from src.weak_ner.extract import extract_entities

def build_summary(text: str) -> str:
    ents = extract_entities(text)
    dx = [e["text"] for e in ents if e["label"] == "DIAG"]
    meds = [e["text"] for e in ents if e["label"] == "MED"]
    doses = [e["text"] for e in ents if e["label"] == "DOSE"]
    lines = []
    lines.append("=== SOAP Summary ===")
    lines.append(f"Assessment (Dx): {', '.join(dx) if dx else '—'}")
    if meds:
        med_str = ", ".join(meds)
        if doses: med_str += f" | doses: {', '.join(doses)}"
        lines.append(f"Medications: {med_str}")
    else:
        lines.append("Medications: —")
    lines.append("Plan: (extract from verbs like start/continue/return — TODO)")
    return "\n".join(lines)

if __name__ == "__main__":
    s = "Hypertension with aspirin 81 mg daily; asthma managed with albuterol."
    print(build_summary(s))
