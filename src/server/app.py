# src/server/app.py

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from .storage import init_storage, write_json, read_json
from .db import (
    init_db,
    create_note,
    update_note_status,
    get_note,
    create_dataset,
    get_dataset_path,
    create_train_run,
    update_train_run,
    get_train_run,
)
from .pipeline import run_full_pipeline
from .training import run_training_job

app = FastAPI(title="Transcriptive Server (Local)")

STORAGE = init_storage("storage")
DB = init_db("storage/server.db")

SUPPORTED_MODELS = {"nb", "logreg", "distilbert"}


class NoteCreateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str = Field(default="distilbert", description="nb | logreg | distilbert")


class NoteCreateResponse(BaseModel):
    note_id: str
    status: str


class NoteStatusResponse(BaseModel):
    note_id: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


# -----------------------
# Notes / inference endpoints
# -----------------------
@app.post("/notes", response_model=NoteCreateResponse)
async def create_note_endpoint(req: NoteCreateRequest):
    model = (req.model or "distilbert").lower().strip()
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model}'. Use one of: {sorted(SUPPORTED_MODELS)}")

    note_id = str(uuid.uuid4())
    input_path = STORAGE.uploads / f"{note_id}.txt"
    input_path.write_text(req.text, encoding="utf-8")

    create_note(DB, note_id, str(input_path))
    asyncio.create_task(_process_note(note_id=note_id, model=model))

    return NoteCreateResponse(note_id=note_id, status="queued")


@app.get("/notes/{note_id}", response_model=NoteStatusResponse)
def get_note_status(note_id: str):
    row = get_note(DB, note_id)
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")

    return NoteStatusResponse(
        note_id=row["note_id"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=row["error"],
    )


@app.get("/notes/{note_id}/results")
def get_note_results(note_id: str):
    row = get_note(DB, note_id)
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")

    if row["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Not ready. Current status: {row['status']}")

    if not row["result_path"]:
        raise HTTPException(status_code=500, detail="Missing result path")

    return read_json(Path(row["result_path"]))


async def _process_note(note_id: str, model: str):
    row = get_note(DB, note_id)
    if not row:
        return

    try:
        update_note_status(DB, note_id, "running")

        text = Path(row["input_path"]).read_text(encoding="utf-8")
        result = run_full_pipeline(text=text, model=model)

        result_path = STORAGE.results / f"{note_id}.json"
        write_json(result_path, result)

        update_note_status(DB, note_id, "complete", result_path=str(result_path))
    except Exception as e:
        update_note_status(DB, note_id, "failed", error=str(e) + "\n" + traceback.format_exc())


# -----------------------
# Training endpoints (cohesive)
# -----------------------
@app.post("/train/data")
async def upload_training_data(model: str, file: UploadFile = File(...)):
    """
    Cohesive entrypoint for the app:

      POST /train/data?model=nb|logreg|distilbert
        - uploads labeled CSV (text,specialty)
        - registers dataset
        - automatically creates a training run
        - starts training async
        - returns dataset_id + run_id for polling
    """
    model = (model or "").lower().strip()
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model}'. Use one of: {sorted(SUPPORTED_MODELS)}")

    dataset_id = str(uuid.uuid4())
    dst = STORAGE.training_data / f"{dataset_id}.csv"
    content = await file.read()
    dst.write_bytes(content)

    # Validate + count rows
    try:
        df = pd.read_csv(dst)
    except Exception:
        dst.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if "text" not in df.columns or "specialty" not in df.columns:
        dst.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="CSV must include columns: text, specialty")

    create_dataset(DB, dataset_id, str(dst), int(len(df)))

    run_id = str(uuid.uuid4())
    create_train_run(DB, run_id, dataset_id, model)

    asyncio.create_task(_run_training_background(run_id=run_id, dataset_id=dataset_id, model=model))

    return {"dataset_id": dataset_id, "rows": int(len(df)), "run_id": run_id, "status": "queued", "model": model}


@app.get("/train/runs/{run_id}")
def get_training_run(run_id: str):
    row = get_train_run(DB, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="run not found")

    metrics = json.loads(row["metrics_json"]) if row.get("metrics_json") else None

    return {
        "run_id": row["run_id"],
        "model": row["model"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "dataset_id": row["dataset_id"],
        "outdir": row["outdir"],
        "metrics": metrics,
        "error": row["error"],
    }


@app.get("/models/active")
def get_active_model():
    # Return active dirs for each model family
    out = {}
    for m in sorted(SUPPORTED_MODELS):
        d = STORAGE.models_active_root / m
        out[m] = str(d) if d.exists() and any(d.iterdir()) else None
    return {"active_models": out}


async def _run_training_background(run_id: str, dataset_id: str, model: str):
    try:
        update_train_run(DB, run_id, "running")

        dataset_path = get_dataset_path(DB, dataset_id)
        if not dataset_path:
            raise RuntimeError("dataset path missing")

        result = await asyncio.to_thread(
            run_training_job,
            dataset_csv=Path(dataset_path),
            run_id=run_id,
            model=model,
            storage_training_runs_dir=STORAGE.training_data / "runs",
            models_registry_runs_dir=STORAGE.models_registry / "runs",
            models_active_root=STORAGE.models_active_root,
        )

        update_train_run(DB, run_id, "complete", outdir=result.get("outdir"), metrics_json=json.dumps(result))
    except Exception as e:
        update_train_run(DB, run_id, "failed", error=str(e) + "\n" + traceback.format_exc())