"""把 Laya（Jev 同赛道开源平替）部署成 JevP 形状的 HTTP 服务。

JevP 生态兼容：`POST /v1/systemone`（{"state", "questions"} -> answers），
返回 Laya 的答案（choice/score/noul + probabilities/confidence），
并附 routing 信息（english / multilingual / typed-decisions 由 Router 自动选）。

用法：
    pip install "laya[serve]"
    python examples/laya_server.py --device cpu --port 8010
    curl -s localhost:8010/v1/systemone -H 'content-type: application/json' -d @examples/request.json
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Laya (Jev alternative) server", version="0.3")

_router = None
_device = "cpu"


def get_router():
    global _router
    if _router is None:
        from laya import Router

        _router = Router(preload=False, device=_device)
        print(f"[laya-server] Router ready (device={_device})")
    return _router


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": "laya-router", "device": _device}


@app.post("/v1/systemone")
def systemone(body: Dict[str, Any]) -> dict:
    state = body.get("state")
    questions = body.get("questions")
    if state is None or not questions:
        raise HTTPException(status_code=422, detail="需要 state 与 questions")
    try:
        result = get_router().predict(state, questions)
    except Exception as exc:  # 下载失败/权重缺失等
        raise HTTPException(status_code=502, detail=f"Laya 推理失败: {exc}") from exc
    return {
        "model": "laya/" + str((result.get("routing") or {}).get("model", "router")),
        "answers": result.get("answers", {}),
        "routing": result.get("routing", {}),
        "usage": {"output_tokens": 0, "note": "判别式模型，无文本生成"},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Laya(Jev 平替) 部署服务")
    ap.add_argument("--device", default=os.environ.get("LAYA_DEVICE", "cpu"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", "8010")))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    global _device
    _device = args.device

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()