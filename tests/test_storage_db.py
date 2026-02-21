from pathlib import Path

from src.server.storage import init_storage
from src.server.db import init_db, create_note, get_note, update_note_status, create_dataset, get_dataset_path, create_train_run, get_train_run, update_train_run


def test_storage_init_creates_expected_dirs(tmp_path):
    s = init_storage(tmp_path / "storage")
    assert s.uploads.exists()
    assert s.results.exists()
    assert s.training_data.exists()
    assert s.models_registry.exists()
    assert s.models_active_root.exists()
    # per-model active dirs exist
    for m in ["nb", "logreg", "distilbert"]:
        assert (s.models_active_root / m).exists()


def test_db_note_lifecycle(tmp_path):
    db = init_db(tmp_path / "server.db")
    create_note(db, "n1", "/tmp/input.txt")
    row = get_note(db, "n1")
    assert row["note_id"] == "n1"
    assert row["status"] == "queued"

    update_note_status(db, "n1", "complete", result_path="/tmp/res.json")
    row = get_note(db, "n1")
    assert row["status"] == "complete"
    assert row["result_path"] == "/tmp/res.json"


def test_db_dataset_and_train_run(tmp_path):
    db = init_db(tmp_path / "server.db")
    create_dataset(db, "d1", "/tmp/d.csv", 123)
    assert get_dataset_path(db, "d1") == "/tmp/d.csv"

    create_train_run(db, "r1", "d1", "nb")
    r = get_train_run(db, "r1")
    assert r["run_id"] == "r1"
    assert r["dataset_id"] == "d1"
    assert r["model"] == "nb"
    assert r["status"] == "queued"

    update_train_run(db, "r1", "complete", outdir="/tmp/out", metrics_json='{"ok":true}')
    r = get_train_run(db, "r1")
    assert r["status"] == "complete"
    assert r["outdir"] == "/tmp/out"
    assert r["metrics_json"] == '{"ok":true}'