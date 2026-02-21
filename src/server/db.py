# src/server/db.py

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

ISO = "%Y-%m-%dT%H:%M:%S"


@dataclass(frozen=True)
class DB:
    path: Path


def _now() -> str:
    return datetime.utcnow().strftime(ISO)


def init_db(db_path: str | Path = "storage/server.db") -> DB:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Notes: inference jobs
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS notes (
        note_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        input_path TEXT NOT NULL,
        result_path TEXT,
        error TEXT
    )
    """
    )

    # Uploaded training datasets
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        path TEXT NOT NULL,
        row_count INTEGER NOT NULL
    )
    """
    )

    # Training runs
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS train_runs (
        run_id TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        outdir TEXT,
        metrics_json TEXT,
        error TEXT
    )
    """
    )

    conn.commit()
    conn.close()
    return DB(path=db_path)


def _connect(db: DB) -> sqlite3.Connection:
    return sqlite3.connect(db.path)


# -----------------------
# Notes helpers
# -----------------------
def create_note(db: DB, note_id: str, input_path: str) -> None:
    conn = _connect(db)
    cur = conn.cursor()
    ts = _now()
    cur.execute(
        "INSERT INTO notes(note_id,status,created_at,updated_at,input_path) VALUES(?,?,?,?,?)",
        (note_id, "queued", ts, ts, input_path),
    )
    conn.commit()
    conn.close()


def update_note_status(
    db: DB,
    note_id: str,
    status: str,
    *,
    result_path: str | None = None,
    error: str | None = None,
) -> None:
    conn = _connect(db)
    cur = conn.cursor()
    ts = _now()
    cur.execute(
        "UPDATE notes SET status=?, updated_at=?, result_path=COALESCE(?, result_path), error=? WHERE note_id=?",
        (status, ts, result_path, error, note_id),
    )
    conn.commit()
    conn.close()


def get_note(db: DB, note_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM notes WHERE note_id=?", (note_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# -----------------------
# Dataset helpers
# -----------------------
def create_dataset(db: DB, dataset_id: str, path: str, row_count: int) -> None:
    conn = _connect(db)
    cur = conn.cursor()
    ts = _now()
    cur.execute(
        "INSERT INTO datasets(dataset_id,created_at,path,row_count) VALUES(?,?,?,?)",
        (dataset_id, ts, path, row_count),
    )
    conn.commit()
    conn.close()


def get_dataset_path(db: DB, dataset_id: str) -> Optional[str]:
    conn = _connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT path FROM datasets WHERE dataset_id=?", (dataset_id,))
    row = cur.fetchone()
    conn.close()
    return row["path"] if row else None


# -----------------------
# Training run helpers
# -----------------------
def create_train_run(db: DB, run_id: str, dataset_id: str, model: str) -> None:
    conn = _connect(db)
    cur = conn.cursor()
    ts = _now()
    cur.execute(
        "INSERT INTO train_runs(run_id,model,status,created_at,updated_at,dataset_id) VALUES(?,?,?,?,?,?)",
        (run_id, model, "queued", ts, ts, dataset_id),
    )
    conn.commit()
    conn.close()


def update_train_run(
    db: DB,
    run_id: str,
    status: str,
    *,
    outdir: str | None = None,
    metrics_json: str | None = None,
    error: str | None = None,
) -> None:
    conn = _connect(db)
    cur = conn.cursor()
    ts = _now()
    cur.execute(
        "UPDATE train_runs SET status=?, updated_at=?, outdir=COALESCE(?, outdir), metrics_json=COALESCE(?, metrics_json), error=? WHERE run_id=?",
        (status, ts, outdir, metrics_json, error, run_id),
    )
    conn.commit()
    conn.close()


def get_train_run(db: DB, run_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM train_runs WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None