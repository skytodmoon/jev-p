"""Product-aware enterprise matching API; full/topk, reusable indexes and jobs.

Run: uvicorn examples.enterprise_api:app --port 8021
See docs/MATCHING_PERFORMANCE.md for coverage semantics and deployment limits.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Literal, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, os.path.join(_ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field, model_validator  # noqa: E402
from threading import RLock  # noqa: E402
from datetime import date  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402
from jevp.match_cache import ScoreCache  # noqa: E402
from jevp.match_jobs import JobManager  # noqa: E402
from jevp.matching import EnterpriseMatcher, MatchOptions  # noqa: E402
from jevp.retrieval import IndexRegistry  # noqa: E402
from scenarios.enterprise_match import frames as DEMO_ZH, PRODUCT  # noqa: E402
from scenarios.enterprise_match_en import frames as DEMO_EN, PRODUCT_EN  # noqa: E402

@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        if _JOBS is not None:
            await run_in_threadpool(_JOBS.close)
        if _CACHE is not None:
            _CACHE.close()


app = FastAPI(title="JevP 企业-产品匹配服务", version="enterprise-match.v2", lifespan=lifespan)

PRODUCT_TEMPLATE = {
    "name": "示例产品（例如：工业预测性维护平台）",
    "description": "一句话描述产品面向的行业、解决的痛点与目标客户特征",
}

COMPANY_TEMPLATE = {
    "name": "示例企业（双击修改名称）",
    "basic": {"industry": "机械制造", "scale": "中型", "location": "苏州",
              "employees": "500", "since": 2010},
    "bidding": [{"year": 2026, "title": "设备预测性维护系统招标", "amount": 80}],
    "website": {"strategy": "智能制造与设备全生命周期管理", "news": "新产线投产"},
    "supply_chain": {"customers": "汽车零部件制造商", "suppliers": "数控机床厂商",
                     "leverage": "设备密集、停机成本高，可作渠道切入"},
}


class MatchRequest(BaseModel):
    product: Optional[Dict[str, Any]] = None
    companies: Optional[List[Dict[str, Any]]] = Field(default=None, min_length=1, max_length=200000)
    index_id: Optional[str] = None
    demo: Optional[str] = Field(default=None, pattern="^(zh|en)$")
    questions: str = Field(default="core", pattern="^(core|all|min)$")
    apply_patch: bool = True
    backend: Literal["auto", "router", "openai", "ollama", "mlx", "transformers", "laya", "agentjev", "vllm"] = "auto"
    mode: str = Field(default="full", pattern="^(full|topk)$")
    top_k: int = Field(default=100, ge=1, le=100000)
    candidate_limit: int = Field(default=500, ge=1, le=100000)
    recall_limit: int = Field(default=5000, ge=1, le=200000)
    batch_size: int = Field(default=32, ge=1, le=512)
    max_batch_chars: int = Field(default=64000, ge=1024, le=2000000)
    use_cache: bool = True
    as_of: Optional[date] = None
    model_revision: str = Field(default="default", max_length=128)

    @model_validator(mode="after")
    def check_sources(self):
        if sum(x is not None for x in (self.companies, self.index_id, self.demo)) > 1:
            raise ValueError("choose only one of companies, index_id and demo")
        if self.mode == "topk" and self.candidate_limit < self.top_k:
            raise ValueError("candidate_limit must be >= top_k")
        return self


class IndexRequest(BaseModel):
    companies: List[Dict[str, Any]] = Field(min_length=1, max_length=200000)


_ENGINES: Dict[str, Any] = {}
_ENGINE_LOCK = RLock()
_LOCAL_MODEL_LOCKS: Dict[str, Any] = {}
_CACHE = None
_JOBS = None
_INDEXES = IndexRegistry()
_CACHE_DIR = os.environ.get("JEVP_CACHE_DIR", os.path.join(_ROOT, ".cache", "matching"))


def get_engine(backend_kind: str) -> Any:
    with _ENGINE_LOCK:
        if backend_kind not in _ENGINES:
            backend = auto_backend(backend=backend_kind)
            _ENGINES[backend_kind] = DecisionEngine(backend, getattr(backend, "tokenizer", None))
        return _ENGINES[backend_kind]


def get_cache():
    global _CACHE
    with _ENGINE_LOCK:
        if _CACHE is None:
            _CACHE = ScoreCache(os.environ.get("JEVP_MATCH_CACHE_PATH", os.path.join(_CACHE_DIR, "scores.sqlite3")),
                                ttl_seconds=int(os.environ.get("JEVP_CACHE_TTL", "3600")),
                                max_entries=int(os.environ.get("JEVP_CACHE_ENTRIES", "100000")))
        return _CACHE


def get_jobs():
    global _JOBS
    with _ENGINE_LOCK:
        if _JOBS is None:
            _JOBS = JobManager(os.path.join(_CACHE_DIR, "jobs.sqlite3"))
        return _JOBS


def _resolve_input(req: MatchRequest):
    index = None
    if req.index_id is not None:
        try:
            index = _INDEXES.get(req.index_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        companies, default_product = index.companies, PRODUCT
    elif req.demo == "en":
        companies, default_product = DEMO_EN(), PRODUCT_EN
    elif req.demo == "zh" or req.companies is None:
        companies, default_product = DEMO_ZH(), PRODUCT
    else:
        companies, default_product = req.companies, PRODUCT
    return companies, req.product if req.product is not None else default_product, index


def _options(req):
    return MatchOptions(mode=req.mode, top_k=req.top_k, candidate_limit=req.candidate_limit,
                        recall_limit=req.recall_limit,
                        batch_size=req.batch_size, max_batch_chars=req.max_batch_chars,
                        question_set=req.questions, apply_patch=req.apply_patch,
                        use_cache=req.use_cache, as_of=req.as_of.isoformat() if req.as_of else "",
                        model_revision=req.model_revision + ":" + os.environ.get("JEVP_MODEL_REVISION", "default"))


def _run_match(req, inputs, progress=None):
    companies, product, index = inputs
    engine = get_engine(req.backend)
    matcher = EnterpriseMatcher(engine, get_cache() if req.use_cache else None)
    # Local tensor runtimes may not support overlapping forwards. HTTP backends
    # already enforce a shared request semaphore, so allow concurrent callers.
    with _ENGINE_LOCK:
        lock = _LOCAL_MODEL_LOCKS.setdefault(req.backend, RLock())
    if hasattr(engine.backend, "_request_gate"):
        out = matcher.match(companies, product, _options(req), index=index, progress=progress)
    else:
        with lock:
            out = matcher.match(companies, product, _options(req), index=index, progress=progress)
    out["demo"] = bool(req.demo)
    return out


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "enterprise-match", "schema": "enterprise-match.v2"}


@app.get("/api/schema", response_model=None)
def api_schema() -> dict:
    return {"schema": "enterprise-match.v2", "input": MatchRequest.model_json_schema(),
            "notes": ["full scores every input company; topk scores only retrieved candidates",
                      "min + apply_patch=true includes industry_fit (2 questions)",
                      "as_of defaults to server calendar date; include an explicit date for reproducibility",
                      "embeddings are optional precomputed features, not generated by this service",
                      "index/job handles are bounded and process-local; use a single worker or sticky routing"],
            "endpoints": ["POST /api/companies/index", "POST /api/match", "POST /api/match/jobs",
                          "GET /api/match/jobs/{job_id}", "GET /api/match/jobs/{job_id}/results?offset=0&limit=100"]}


@app.post("/api/companies/index", status_code=201)
def create_index(req: IndexRequest):
    try:
        handle, index = _INDEXES.add(req.companies)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"index_id": handle, "company_count": len(index.companies), "digest": index.digest,
            "vector_dimension": index.dimension, "scope": "process", "ttl_seconds": _INDEXES.ttl}


@app.get("/api/match/template")
def match_template() -> dict:
    """输入默认模板（调试台自动预填，减少手写）：单企业 + 产品，字段齐全可直接改值。"""
    return {"product": PRODUCT_TEMPLATE, "company": COMPANY_TEMPLATE}


@app.get("/api/match/example")
def example(lang: str = "zh") -> dict:
    """返回内置示例企业列表（供调试页填充输入框）：lang=zh|en。"""
    if lang == "en":
        return {"lang": "en", "companies": DEMO_EN()}
    return {"lang": "zh", "companies": DEMO_ZH()}


@app.post("/api/match")
def api_match(req: MatchRequest) -> dict:
    try:
        return _run_match(req, _resolve_input(req))
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/match/jobs", status_code=202)
def submit_match_job(req: MatchRequest):
    inputs = _resolve_input(req)
    try:
        handle = get_jobs().submit(lambda progress: _run_match(req, inputs, progress))
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc
    return {"job_id": handle, "status_url": f"/api/match/jobs/{handle}",
            "results_url": f"/api/match/jobs/{handle}/results"}


@app.get("/api/match/jobs/{job_id}")
def match_job_status(job_id: str):
    try:
        return get_jobs().status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/match/jobs/{job_id}/results")
def match_job_results(job_id: str, offset: int = 0, limit: int = 100):
    try:
        return get_jobs().results(job_id, offset, limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409 if "completion" in str(exc) else 422, detail=str(exc)) from exc


@app.post("/api/match/one")
def api_match_one(body: Dict[str, Any]) -> dict:
    # Preserve the old raw-company shape while accepting {product, company}.
    company = body.get("company", body)
    req = MatchRequest(companies=[company], product=body.get("product") if "company" in body else None,
                       questions="all", apply_patch=True)
    out = api_match(req)
    return {"schema": out["schema"], "company": out["ranking"][0], "meta": out["meta"]}


@app.get("/")
def debug_page() -> FileResponse:
    return FileResponse(os.path.join(_THIS_DIR, "enterprise_debug.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("JEVP_API_PORT", "8021")))
