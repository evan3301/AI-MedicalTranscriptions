import re
from typing import List, Dict

def flag_impossible_vitals(text: str) -> List[Dict]:
    flags = []
    if re.search(r"\bTemp(?:erature)?\b.*\b(1[2-9]\d|[5-6]\d)\s*F\b", text):
        flags.append({"type":"vital_range", "msg":"Temperature out of plausible range"})
    if re.search(r"\bHR\b.*\b([01]?\d|2\d{2,})\b", text):
        flags.append({"type":"vital_range", "msg":"Heart rate out of plausible range"})
    return flags

def flag_missing_sections(text: str) -> List[Dict]:
    expected = ["Medications", "Assessment", "Plan"]
    flags = []
    for sec in expected:
        if sec.lower() not in text.lower():
            flags.append({"type":"section_missing", "msg": f"Missing section: {sec}"})
    return flags

def run_qa(text: str) -> List[Dict]:
    return flag_impossible_vitals(text) + flag_missing_sections(text)

if __name__ == "__main__":
    sample = "Temp 120 F. Assessment: ..."
    print(run_qa(sample))
