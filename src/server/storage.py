# src/server/storage.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class Storage:
    root: Path
    uploads: Path
    results: Path
    training_data: Path
    models_registry: Path
    models_active_root: Path  # storage/models_registry/active


def init_storage(root: str | Path = "storage") -> Storage:
    root = Path(root)
    uploads = root / "uploads"
    results = root / "results"
    training_data = root / "training_data"
    models_registry = root / "models_registry"
    models_active_root = models_registry / "active"

    for p in [root, uploads, results, training_data, models_registry, models_active_root]:
        p.mkdir(parents=True, exist_ok=True)

    # Create per-model active directories (may stay empty until training succeeds)
    for model in ["nb", "logreg", "distilbert"]:
        (models_active_root / model).mkdir(parents=True, exist_ok=True)

    return Storage(
        root=root,
        uploads=uploads,
        results=results,
        training_data=training_data,
        models_registry=models_registry,
        models_active_root=models_active_root,
    )


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))