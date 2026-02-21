# Transcriptive – AI-Powered Medical Transcription Intelligence (Server)

Transcriptive is a lightweight, reproducible backend system that transforms raw clinical notes into structured, clinically useful artifacts through machine learning and rule-based post-processing.

This repository contains the **FastAPI backend server** powering the Android demo application.

---

## 🚀 What This System Does

Transcriptive provides a full end-to-end AI workflow:

- 🏥 Specialty Classification (Naive Bayes, Logistic Regression, DistilBERT)
- 📊 Automated Training Pipeline
- 🗂 Model Versioning & Registry
- 📈 Validation Metrics Tracking
- 🧠 Prediction with Confidence Scores
- 📝 SOAP-Style Summary Generation

The system is designed to be:
- Reproducible
- Modular
- Version-controlled
- Presentation-ready

---

## 🧠 Model Tiers

Users can select one of three model tiers:

| Tier       | Model Used                  | Speed     | Expected Accuracy |
|------------|----------------------------|-----------|-------------------|
| Instant    | TF-IDF + Naive Bayes       | Fastest   | Baseline          |
| Standard   | TF-IDF + Logistic Regression | Moderate  | Improved          |
| Thinking   | DistilBERT (Transformer)   | Slowest   | Highest Potential |

All tiers use the same training pipeline and registry system.

---

## 🏗 System Architecture

The backend workflow operates as follows:

1. User uploads a labeled CSV dataset  
2. Server performs a stratified train/validation split  
3. Selected model tier is trained  
4. Model artifacts are stored in a versioned registry  
5. Model is activated  
6. Inference is performed  
7. Prediction + confidence + SOAP summary are returned  

All training runs are tracked with reproducible configuration metadata.

---

## 📂 Project Structure

```
src/
  server/
    app.py          # FastAPI app + API endpoints
    pipeline.py     # Training & inference orchestration
    db.py           # Run tracking (SQLite)
storage/
  uploads/              # Uploaded CSVs (runtime)
  training_data/        # Split datasets (runtime)
  models_registry/
    runs/               # Versioned model outputs
    active/             # Currently active model
```

> Runtime artifacts (uploads, splits, model runs) are excluded from version control.

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

(Windows: `.venv\Scripts\activate`)

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Server

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```

---

## ✅ Verify Server Is Running

Open in your browser:

```
http://localhost:8000/health
```

Expected response:

```json
{"ok": true}
```

---

## 📄 Required Data Format

Each CSV must include:

- `text`
- `specialty`

Example:

```csv
text,specialty
"Fever and cough for 3 days","Pediatrics"
"Chest pain on exertion","Cardiology"
```

---

## 🔌 Core API Endpoints

### Upload + Train

```
POST /train/data?model=nb
POST /train/data?model=logreg
POST /train/data?model=distilbert
```

- Accepts labeled CSV
- Automatically triggers training
- Returns `run_id`

---

### Check Run Status

```
GET /train/runs/{run_id}
```

Returns:
- Training status
- Validation metrics
- Model metadata

---

### Active Model

```
GET /models/active
```

Returns currently deployed model tier.

---

### Predict + Diagnose

```
POST /notes/predict
```

Returns:

```json
{
  "prediction": "Cardiology",
  "confidence": 0.42,
  "soap_summary": "..."
}
```

---

## 📊 Validation Metrics

Each training run records:

- Validation Accuracy
- Macro F1 Score
- Training Sample Count

Metrics are stored alongside model artifacts in the registry.

---

## 📦 Model Registry

Each training run produces:

- Model artifact
- Configuration metadata
- Validation metrics
- Timestamped run ID

The most recent completed run for a tier becomes the active model.

---

## 📱 Android Integration

The Android client connects to:

```
http://10.0.2.2:8000
```

(Emulator → localhost bridge)

User flow:
1. Select model tier
2. Upload CSV
3. Trigger training
4. View metrics
5. Submit note for diagnosis
6. Receive prediction + confidence + SOAP summary

---

## ⚠️ Notes on DistilBERT

- Large model weights are not committed to GitHub.
- Runtime-trained models are stored under `storage/models_registry/`.
- Transformer training requires more compute time and memory.

---

## 🛑 Disclaimer

This project is an educational prototype.

- Not a medical device
- No PHI should be used
- Outputs require clinical validation
- Confidence scores reflect model uncertainty, not clinical certainty

---

## 📌 Summary

Transcriptive demonstrates:

- End-to-end ML training pipeline
- Version-controlled model deployment
- Mobile client + backend coordination
- Practical AI application in healthcare context

It satisfies the requirements of a cohesive, communicating AI-powered system across server and mobile layers.
