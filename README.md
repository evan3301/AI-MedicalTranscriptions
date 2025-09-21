# Transcriptive — PyTorch Starter

A minimal, PyTorch-first scaffold for **Project 2: Transcriptive** (specialty classification, entity extraction, QA/error checks, summary).

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (Classification)
```bash
# Put/inspect example CSVs in data/
python src/classify/train_distilbert.py  # trains DistilBERT on data/train.csv & data/val.csv
python src/classify/predict.py models/cls_distilbert "Patient presents with chest pain radiating to left arm."
```

## Data Format
Each CSV has columns: `id,text,specialty`
```
id,text,specialty
1,"Chief complaint: fever and cough for 3 days...", "Pediatrics"
2,"Chest pain on exertion, ECG shows ST changes...", "Cardiology"
```

You can replace the toy examples in `data/` with your own split (stratify by specialty).

## Components
- `src/classify/` — PyTorch (Transformers) document classifier
- `src/tokencls/` — optional PyTorch token classification (NER) head
- `src/weak_ner/` — dictionary/regex-based extraction (silver labels optional)
- `src/qa/` — rule-based error checks (spelling/structure/range)
- `src/summarize/` — SOAP/brief summary builder (template-based)

## Reports
- Validation and test metrics JSON will be saved under `reports/metrics/`.
- Put illustrative before→after examples under `reports/examples/`.

## Notes
- This is an **educational prototype**, not a medical device.
- Pin library versions for reproducibility.
