"""企业-产品匹配服务 API + 调试页面（FastAPI）。

端点：
    GET  /                   网页调试台（examples/enterprise_debug.html）
    GET  /healthz            健康检查
    GET  /api/schema         输入/输出结构契约（文档）
    POST /api/match          批量企业匹配打分 + 排序（核心）
    POST /api/match/one      单企业快查

输入（POST /api/match）：
{
  "product":    {"name": str, "description": str},   // 可选，缺省用内置示例产品
  "companies":  [企业档案...],                        // 可选（缺省=demo:zh 示例）
  "demo":       "zh" | "en" | null,                  // 快捷加载内置中/英企业示例
  "questions":  "core" | "all" | "min",              // 默认 core（3 问/家）
  "apply_patch": true,                               // 默认 true：域先验硬校正
  "backend":    "auto" | "router" | "openai" | ...   // 默认 auto（VLM_* 环境 -> 27B）
}

企业档案 schema（字段全部可选，最少 name+basic.industry）：
{
  "name": str, "basic": {"industry","scale","location","employees","since"},
  "bidding": [{"year": int, "title": str, "amount": num?}],
  "website": {"strategy": str, "news": str},
  "supply_chain": {"customers","suppliers","leverage"}
}

输出：
{
  "schema": "enterprise-match.v1",
  "demo": bool, "product": {...}, "questions_used": [...],
  "ranking": [ {
      "name", "lang": "zh|en",
      "match_score": 校正后分, "score_raw": 原始分, "patched": bool,
      "patch_note": "...", "confidence", "margin",
      "advice": "🔥 优先触达|→ 可跟进|◇ 观察待唤醒|✋ 暂不触达（记录线索）",
      "per_question": {...}
  } ...],  // 按 match_score 降序
  "meta": {"backend", "call_est": int, "latency_ms": float}
}
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, os.path.join(_ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from jevp.backend import auto_backend, is_chinese_like  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402
from scenarios.enterprise_match import build_questions, frames as DEMO_ZH, PRODUCT  # noqa: E402
from scenarios.enterprise_match_en import frames as DEMO_EN, PRODUCT_EN  # noqa: E402
from rule_baseline_enterprise import domain_patch, is_non_commercial  # noqa: E402

app = FastAPI(title="JevP 企业-产品匹配服务", version="enterprise-match.v1")

_QUESTION_SETS = {
    "all": None,
    "core": ("industry_fit", "demand_signal", "match_score"),
    "min": ("match_score",),
}


def route_advice(score: float | None, conf: float) -> str:
    s = score if score is not None else 0.0
    if s >= 2.5 and conf >= 0.9:
        return "🔥 优先触达"
    if s >= 2.0 and conf >= 0.7:
        return "→ 可跟进"
    if s >= 1.5:
        return "◇ 观察待唤醒"
    return "✋ 暂不触达（记录线索）"


class MatchRequest(BaseModel):
    product: Optional[Dict[str, Any]] = None
    companies: Optional[List[Dict[str, Any]]] = None
    demo: Optional[str] = None  # "zh" | "en"
    questions: str = "core"
    apply_patch: bool = True
    backend: str = "auto"


# -- 引擎 ---------------------------------------------------------------- #
_ENGINES: Dict[str, Any] = {}


def get_engine(backend_kind: str) -> Any:
    if backend_kind not in _ENGINES:
        backend = auto_backend(backend=backend_kind)
        _ENGINES[backend_kind] = DecisionEngine(
            backend=backend, tokenizer=getattr(backend, "tokenizer", None))
    return _ENGINES[backend_kind]


def _resolve_input(req: MatchRequest):
    """解析请求：示例加载、产品、企业列表、问题集。"""
    if req.demo == "en":
        companies, product = DEMO_EN(), PRODUCT_EN
    elif req.demo == "zh" or (req.demo is None and not req.companies):
        companies, product = DEMO_ZH(), PRODUCT
    else:
        companies = req.companies or []
        product = req.product or PRODUCT
    if not companies:
        raise HTTPException(status_code=422, detail="companies 不能为空（或使用 demo=zh/en 加载示例）")
    qtype = req.questions if req.questions in _QUESTION_SETS else "core"
    keep = _QUESTION_SETS[qtype]
    questions = build_questions()
    if keep:
        questions = {k: questions[k] for k in keep}
    return companies, product, questions, qtype


def _score_one(engine, ent: Dict[str, Any], questions, apply_patch: bool) -> Dict[str, Any]:
    ans = engine.decide(ent, questions)
    flat = {k: v.to_dict() for k, v in ans.items()}
    ms = flat.get("match_score", {})
    score_raw = ms.get("score")
    conf = ms.get("confidence")
    probs = ms.get("probabilities") or {}
    margin = 0.0
    if probs:
        vals = sorted(probs.values(), reverse=True)
        margin = (vals[0] - vals[1]) if len(vals) > 1 else vals[0]

    ind_fit = flat.get("industry_fit", {}).get("noul")
    row_in = {
        "industry_fit": (ind_fit or 0.0) >= 0.5 if ind_fit is not None else None,
        "n_bids": len(ent.get("bidding", [])),
        "non_commercial": is_non_commercial(ent),
    }
    patch_note = ""
    patched = False
    score = score_raw
    if apply_patch and "industry_fit" in questions:
        s = domain_patch(score_raw, row_in)
        if abs((s or 0) - (score_raw or 0)) > 1e-9:
            patched = True
            if row_in["non_commercial"]:
                patch_note = "非企业实体→≤0.5"
            elif row_in["industry_fit"] is False and row_in["n_bids"] == 0:
                patch_note = "行业无关且无招投标→≤0.5"
            else:
                patch_note = "行业不匹配→≤1.0"
        score = s

    lang = "zh" if is_chinese_like(ent, questions) else "en"
    return {
        "name": ent.get("name", "?"),
        "lang": lang,
        "match_score": (round(score, 4) if score is not None else None),
        "score_raw": (round(score_raw, 4) if score_raw is not None else None),
        "patched": patched,
        "patch_note": patch_note,
        "confidence": (round(conf, 4) if conf is not None else None),
        "margin": round(margin, 4),
        "advice": route_advice(score, conf or 0.0),
        "per_question": {k: v for k, v in flat.items()},
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "enterprise-match", "schema": "enterprise-match.v1"}


@app.get("/api/schema", response_model=None)
def api_schema() -> dict:
    """输入/输出结构契约（供前端与集成方取用）。"""
    return {
        "schema": "enterprise-match.v1",
        "input": {
            "product": {"name": "str(可选)", "description": "str(可选，缺省内置 Titan-PHM)"},
            "companies": ["企业档案：name/basic{industry,scale,location,employees,since}/"
                          "bidding[{year,title,amount?}]/website{strategy,news}/supply_chain{...}"],
            "demo": '"zh"|"en"|null（快捷加载内置示例）',
            "questions": '"core"|"all"|"min"（问数/家：3/6/1，默认 core）',
            "apply_patch": "bool（默认 true：域先验硬校正行业✗≤1/无关≤0.5/非企业≤0.5）",
            "backend": '"auto"(VLM_*→27B)|"router"(语言路由)|"openai"|"laya"|"agentjev"',
        },
        "output": {
            "schema": '"enterprise-match.v1"',
            "demo": "bool", "product": "object", "questions_used": ["str"],
            "ranking": [{
                "name": "str", "lang": '"zh"|"en"',
                "match_score": "校正后匹配分 0-3（小数，可排序）",
                "score_raw": "原始分（未校正）", "patched": "bool 是否经校正",
                "patch_note": "str 校正原因", "confidence": "0-1 边际置信",
                "margin": "top1-top2 差值", "advice": "🔥/→/◇/✋ 触达建议",
                "per_question": "{question: 各维度答案对象}",
            }],
            "meta": {"backend": "str", "call_est": "问数×企业数，int", "latency_ms": "float"},
        },
        "examples": ["POST /api/match  demo=zh", "POST /api/match  demo=en",
                     "POST /api/match/one"],
    }


@app.get("/api/match/example")
def example(lang: str = "zh") -> dict:
    """返回内置示例企业列表（供调试页填充输入框）：lang=zh|en。"""
    if lang == "en":
        return {"lang": "en", "companies": DEMO_EN()}
    return {"lang": "zh", "companies": DEMO_ZH()}


@app.post("/api/match")
def api_match(req: MatchRequest) -> dict:
    t0 = time.perf_counter()
    engine = get_engine(req.backend if req.backend != "auto" else "auto")
    companies, product, questions, qtype = _resolve_input(req)

    ranking = []
    for ent in companies:
        try:
            ranking.append(_score_one(engine, ent, questions, req.apply_patch))
        except Exception as exc:  # 单企业失败不拖垮整批
            ranking.append({
                "name": ent.get("name", "?"), "error": str(exc)[:200],
                "match_score": None, "confidence": None,
            })
    ranking.sort(key=lambda r: (r.get("match_score") if r.get("match_score") is not None else -1,
                                r.get("confidence") or 0), reverse=True)

    return {
        "schema": "enterprise-match.v1",
        "demo": bool(req.demo),
        "product": product,
        "questions_used": list(questions),
        "ranking": ranking,
        "meta": {
            "backend": getattr(getattr(engine, "backend", None), "model_id", req.backend),
            "call_est": len(companies) * len(questions),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        },
    }


@app.post("/api/match/one")
def api_match_one(company: Dict[str, Any]) -> dict:
    req = MatchRequest(companies=[company], questions="all", apply_patch=True)
    out = api_match(req)
    single = out["ranking"][0] if out["ranking"] else {}
    return {"schema": "enterprise-match.v1", "company": single, "meta": out["meta"]}


@app.get("/")
def debug_page() -> FileResponse:
    return FileResponse(os.path.join(_THIS_DIR, "enterprise_debug.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("JEVP_API_PORT", "8021")))