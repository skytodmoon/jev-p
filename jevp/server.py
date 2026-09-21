"""FastAPI 服务：暴露对齐 Jev 的 POST /v1/systemone。

请求：{"state": <str|JSON>, "questions": {name: {type, instructions, criteria}}}
响应：{"model", "answers": {name: {...}}, "usage": {"input_tokens", "output_tokens"}}
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from .backend import auto_backend
from .engine import DecisionEngine
from .protocol import validate_questions

app = FastAPI(title="JevP", version="0.1.0", description="System One decision engine on small LLMs")

_engine: Optional[DecisionEngine] = None


def get_engine() -> DecisionEngine:
    global _engine
    if _engine is None:
        backend_kind = os.environ.get("JEVP_BACKEND", "auto")  # auto|mlx|transformers
        model = os.environ.get("JEVP_MODEL")  # 缺省时 auto_backend 取 7B 默认
        backend = auto_backend(model=model, backend=backend_kind)
        _engine = DecisionEngine(backend=backend, tokenizer=backend.tokenizer)
        print(f"[JevP] backend={type(backend).__name__} model={backend.model_id}")
    return _engine


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": get_engine().backend.model_id}


@app.post("/v1/systemone")
def systemone(body: Dict[str, Any]) -> dict:
    engine = get_engine()
    state = body.get("state")
    if state is None:
        raise HTTPException(status_code=422, detail="缺少字段: state")
    raw_questions = body.get("questions")
    if not raw_questions:
        raise HTTPException(status_code=422, detail="缺少字段: questions")
    try:
        questions = validate_questions(raw_questions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    answers = engine.decide(state, questions)
    # 统计输入 token：重新 tokenize prompts 的开销可接受
    from .protocol import build_prompt

    names = list(questions)
    prompts_text = [build_prompt(state, n, questions[n]) for n in names]
    input_tokens = engine.count_input_tokens(prompts_text)

    return {
        "model": engine.backend.model_id,
        "answers": {name: ans.to_dict() for name, ans in answers.items()},
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("JEVP_PORT", "8000")))