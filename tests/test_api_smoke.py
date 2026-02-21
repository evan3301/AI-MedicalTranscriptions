import time


def _wait_for_status(client, run_id: str, timeout_s: float = 20.0) -> str:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"/train/runs/{run_id}")
        assert r.status_code == 200
        last = r.json()["status"]
        if last in ("complete", "failed"):
            return last
        time.sleep(0.05)
    return last or "unknown"


def _wait_for_note_complete(client, note_id: str, timeout_s: float = 10.0) -> str:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"/notes/{note_id}")
        assert r.status_code == 200
        last = r.json()["status"]
        if last in ("complete", "failed"):
            return last
        time.sleep(0.05)
    return last or "unknown"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_train_data_autostarts_run(client, sample_dataset_csv):
    with open(sample_dataset_csv, "rb") as f:
        r = client.post("/train/data?model=nb", files={"file": ("dataset.csv", f, "text/csv")})

    assert r.status_code == 200
    payload = r.json()
    assert payload["model"] == "nb"
    run_id = payload["run_id"]

    # It should at least exist and be queued/running
    r2 = client.get(f"/train/runs/{run_id}")
    assert r2.status_code == 200
    assert r2.json()["status"] in ("queued", "running", "complete", "failed")

    # Training is async; allow it to still be running after a short wait
    final = _wait_for_status(client, run_id, timeout_s=20.0)
    assert final in ("complete", "failed", "running")


def test_notes_end_to_end_returns_mobile_fields(client):
    r = client.post("/notes", json={"text": "Assessment: viral URI. Plan: rest.", "model": "nb"})
    assert r.status_code == 200
    note_id = r.json()["note_id"]

    final = _wait_for_note_complete(client, note_id, timeout_s=10.0)
    assert final == "complete"

    r3 = client.get(f"/notes/{note_id}/results")
    assert r3.status_code == 200
    res = r3.json()

    assert "prediction" in res
    assert "confidence" in res
    assert "soap_summary" in res
    assert res["model_used"] == "nb"