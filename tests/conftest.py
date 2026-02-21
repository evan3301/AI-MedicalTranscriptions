import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import pytest
from fastapi.testclient import TestClient

from src.server import app as app_module
from src.server.db import init_db
from src.server.storage import init_storage


@pytest.fixture()
def temp_env(tmp_path, monkeypatch):
    """
    Creates an isolated storage/DB per test and patches src.server.app globals.

    Important: In FastAPI TestClient there is already a running event loop.
    So we must NOT call asyncio.run() from inside request handling.
    Instead we schedule tasks onto the currently running loop.
    """
    storage_root = tmp_path / "storage"
    storage = init_storage(storage_root)
    db = init_db(storage_root / "server.db")

    # Patch server globals
    monkeypatch.setattr(app_module, "STORAGE", storage, raising=True)
    monkeypatch.setattr(app_module, "DB", db, raising=True)

    # Patch create_task to schedule onto the current loop (no asyncio.run)
    def _create_task(coro):
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)

    monkeypatch.setattr(app_module.asyncio, "create_task", _create_task, raising=True)

    return {"storage": storage, "db": db, "root": tmp_path}


@pytest.fixture()
def client(temp_env):
    return TestClient(app_module.app)


@pytest.fixture()
def sample_dataset_csv(tmp_path) -> Path:
    p = tmp_path / "dataset.csv"
    p.write_text(
        "text,specialty\n"
        "patient has cough and fever,Family Medicine\n"
        "follow-up for hypertension,Family Medicine\n"
        "skin rash and itching,Dermatology\n"
        "eczema flare and topical steroid,Dermatology\n"
        "annual physical exam,Family Medicine\n"
        "acne treatment started,Dermatology\n"
        "URI symptoms and rest advised,Family Medicine\n"
        "psoriasis plaque noted,Dermatology\n"
        "flu-like illness supportive care,Family Medicine\n"
        "dermatitis counseling,Dermatology\n",
        encoding="utf-8",
    )
    return p