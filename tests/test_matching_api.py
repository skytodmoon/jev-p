"""API contracts, reusable corpus handles, backpressure and job result pagination."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from examples import enterprise_api as api
from jevp.engine import DecisionEngine
from jevp.match_cache import ScoreCache
from jevp.match_jobs import JobManager
from jevp.retrieval import IndexRegistry
from test_matching_pipeline import COMPANIES, PRODUCT, FixtureBackend


@pytest.fixture
def client(monkeypatch, tmp_path):
    cache = ScoreCache()
    jobs = JobManager(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(api, "_ENGINES", {"auto": DecisionEngine(FixtureBackend(), None)})
    monkeypatch.setattr(api, "_CACHE", cache)
    monkeypatch.setattr(api, "_JOBS", jobs)
    monkeypatch.setattr(api, "_INDEXES", IndexRegistry())
    with TestClient(api.app) as client:
        yield client
    jobs.close()
    cache.close()


def wait_done(client, handle):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get(f"/api/match/jobs/{handle}").json()
        if status["status"] in {"completed", "failed"}:
            return status
        time.sleep(0.005)
    pytest.fail("background job did not complete")


def test_full_api_and_hot_cache(client):
    body = {"backend": "auto", "product": PRODUCT, "companies": COMPANIES}
    out = client.post("/api/match", json=body)
    assert out.status_code == 200
    assert out.json()["schema"] == "enterprise-match.v2"
    assert out.json()["meta"]["all_companies_scored"]
    hot = client.post("/api/match", json=body).json()
    assert hot["meta"]["call_est"] == 0
    assert hot["meta"]["cache_hits"] == 3


def test_demo_honors_product_override(client):
    out = client.post("/api/match", json={"backend": "auto", "demo": "en", "product": PRODUCT})
    assert out.status_code == 200
    assert out.json()["product"] == PRODUCT


def test_prepared_index_and_topk(client):
    created = client.post("/api/companies/index", json={"companies": COMPANIES})
    assert created.status_code == 201
    out = client.post("/api/match", json={"backend": "auto", "index_id": created.json()["index_id"],
                                         "product": PRODUCT, "mode": "topk", "top_k": 1, "candidate_limit": 2})
    assert out.status_code == 200
    assert out.json()["meta"]["candidate_count"] == 2
    assert not out.json()["meta"]["all_companies_scored"]
    assert len(out.json()["ranking"]) == 1


@pytest.mark.parametrize("changes", [{"companies": []}, {"batch_size": 0}, {"questions": "typo"},
                                     {"mode": "topk", "top_k": 20, "candidate_limit": 2},
                                     {"as_of": "invalid"}, {"demo": "zh", "companies": COMPANIES}, {"backend": "typo"}])
def test_invalid_requests_return_422(client, changes):
    out = client.post("/api/match", json={"backend": "auto", **changes})
    assert out.status_code == 422


def test_unknown_handles_and_invalid_vectors(client):
    assert client.post("/api/match", json={"index_id": "missing", "backend": "auto"}).status_code == 404
    assert client.get("/api/match/jobs/missing").status_code == 404
    assert client.post("/api/companies/index", json={"companies": [{"embedding": [0, 0]}]}).status_code == 422


def test_async_job_progress_and_result_pages(client):
    submitted = client.post("/api/match/jobs", json={"companies": COMPANIES, "product": PRODUCT, "backend": "auto"})
    assert submitted.status_code == 202
    handle = submitted.json()["job_id"]
    status = wait_done(client, handle)
    assert status["status"] == "completed"
    assert status["processed"] == status["total"] == 3
    first = client.get(f"/api/match/jobs/{handle}/results?offset=0&limit=2").json()
    second = client.get(f"/api/match/jobs/{handle}/results?offset=2&limit=2").json()
    assert len(first["ranking"]) == 2 and len(second["ranking"]) == 1
    assert first["pagination"]["next_offset"] == 2
    assert second["pagination"]["next_offset"] is None
    assert client.get(f"/api/match/jobs/{handle}/results?limit=1001").status_code == 422


def test_job_backpressure_failure_and_eviction():
    jobs = JobManager(workers=1, max_pending=1, max_jobs=1)
    ready = threading.Event()
    release = threading.Event()

    def block(progress):
        ready.set()
        release.wait(3)
        return {"ranking": [], "errors": [], "meta": {}}

    try:
        first = jobs.submit(block)
        assert ready.wait(2)
        with pytest.raises(OverflowError):
            jobs.submit(block)
        with pytest.raises(ValueError):
            jobs.results(first)
        release.set()
        deadline = time.monotonic() + 3
        while jobs.status(first)["status"] != "completed" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert jobs.status(first)["status"] == "completed"

        def fail(progress):
            raise RuntimeError("injected job failure")
        second = jobs.submit(fail)
        deadline = time.monotonic() + 3
        while jobs.status(second)["status"] != "failed" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert jobs.status(second)["error"] == "injected job failure"
        with pytest.raises(KeyError):
            jobs.status(first)
    finally:
        release.set()
        jobs.close()
