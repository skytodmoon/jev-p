"""企业-产品匹配评价演示：潜在客户挖掘 → 打分 → 排序（含 LLM 调用降频控制）。

架构（System One 批量精排）：
    企业库/招投标/官网 ──recall──▶ 候选企业（本 demo 用 8 家合成档案）
        ──▶ JevP 原子问题打分 ──▶ match_score + confidence/margin ──▶ 排序 Top-N

【LLM 调用降频设计】（本脚本核心诉求）
    每批候选的 LLM 调用次数 = 问题数 × 企业数：
      --questions all  → 6 问/家（48 次/8家）：维度最全，精评档
      --questions core → 3 问/家（24 次/8家，默认）：行业+需求+综合，性价比档
      --questions min  → 1 问/家（ 8 次/8家）：极速粗筛档，只问综合匹配分
    三级漏斗配合（README §3.12）：
      ① L0 规则粗筛（0 LLM：行业/招投标年份/体量字段直接过滤或本地判别式 Laya 粗排）
      ② L1 本脚本打分（core/min 档）
      ③ L2 仅对 Top-N 精评（all 档）+ 结果缓存（同一企业 7 天内不重复调用）

用法：
    # 默认 core（24 次调用），后端 VLM_*（qwen3.8-27b 等）
    python examples/enterprise_match_demo.py
    # 极速粗筛（8 次调用）
    python examples/enterprise_match_demo.py --questions min
    # 精评档（48 次调用）
    python examples/enterprise_match_demo.py --questions all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, os.path.join(_ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from scenarios.enterprise_match import PRODUCT, describe, frames, build_questions  # noqa: E402
from rule_baseline_enterprise import domain_patch, is_non_commercial  # noqa: E402
from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402

# 问题集裁剪：all / core / min
_QUESTION_SETS = {
    "all": None,  # 全部 6 问
    "core": ("industry_fit", "demand_signal", "match_score"),
    "min": ("match_score",),
}


def route_advice(score: float | None, conf: float) -> str:
    """分数×置信度联合路由（避免低分高置信误触达）。"""
    s = score if score is not None else 0.0
    if s >= 2.5 and conf >= 0.9:
        return "🔥 优先触达"
    if s >= 2.0 and conf >= 0.7:
        return "→ 可跟进"
    if s >= 1.5:
        return "◇ 观察待唤醒"
    return "✋ 暂不触达（记录线索）"


def main() -> None:
    ap = argparse.ArgumentParser(description="企业-产品匹配打分排序")
    ap.add_argument("--questions", choices=list(_QUESTION_SETS), default="core",
                    help="问题集：all=6问/core=3问(默认)/min=1问 —— 决定 LLM 调用次数")
    ap.add_argument("--backend", default=os.environ.get("JEVP_BACKEND", "auto"))
    ap.add_argument("--no-patch", action="store_true",
                    help="关闭域先验硬校正（默认开启：行业不匹配→≤1、无关→≤0.5、非企业→≤0.5）")
    args = ap.parse_args()

    backend = auto_backend(model=os.environ.get("JEVP_MODEL"), backend=args.backend)
    engine = DecisionEngine(backend=backend, tokenizer=getattr(backend, "tokenizer", None))

    keep = _QUESTION_SETS[args.questions]
    questions = build_questions()
    if keep:
        questions = {k: questions[k] for k in keep}

    n_call = len(questions) * len(frames())
    print(f"[match] 后端: {type(backend).__name__} ({backend.model_id})")
    print(f"[match] 问题集: {args.questions}（{len(questions)} 问/家 × {len(frames())} 家 = "
          f"~{n_call} 次 LLM 调用/批）")
    print(f"[match] 产品: {PRODUCT['name']}\n       {PRODUCT['description']}\n")

    rows: List[Dict[str, Any]] = []
    for idx, ent in enumerate(frames()):
        answers = engine.decide(ent, questions)
        flat = {k: v.to_dict() for k, v in answers.items()}
        ms = flat.get("match_score", {})
        score = ms.get("score")
        conf = ms.get("confidence")
        probs = ms.get("probabilities") or {}
        margin = 0.0
        if probs:
            vals = sorted(probs.values(), reverse=True)
            margin = (vals[0] - vals[1]) if len(vals) > 1 else vals[0]
        rows.append({
            "name": ent["name"], "desc": describe(ent, idx),
            "score": score, "score_raw": score, "confidence": conf, "margin": margin,
            "industry_fit": flat["industry_fit"].get("noul") if "industry_fit" in flat else None,
            "demand_signal": flat["demand_signal"].get("noul") if "demand_signal" in flat else None,
            "urgency": flat["urgency"].get("noul") if "urgency" in flat else None,
            "n_bids": len(ent.get("bidding", [])),
            "non_commercial": is_non_commercial(ent),
        })

    if not args.no_patch:
        for r in rows:
            r["score"] = domain_patch(r["score"], r)

    rows.sort(key=lambda r: (r["score"] or -1, r["confidence"] or 0), reverse=True)

    patched = not args.no_patch
    print(f"{'排名':<4}{'企业':<18}{'匹配分':>7}{'确认度':>8}{'margin':>8}  建议")
    print("-" * 66)
    for i, r in enumerate(rows, 1):
        score_txt = f"{r['score']:>7.2f}"
        if patched and abs((r['score'] or 0) - (r['score_raw'] or 0)) > 1e-9:
            score_txt = f"{r['score']:>5.2f}⬅{r['score_raw']:.2f}"
        print(f"{i:<4}{r['name']:<18}{score_txt}{(r['confidence'] or 0):>8.3f}"
              f"{r['margin']:>8.3f}  {route_advice(r['score'], r['confidence'] or 0)}")
    if patched:
        print("\n* 匹配分为域先验校正后（← 表示经校正回退；--no-patch 查看原始分）")

    print("\n== Top3 明细 ==")
    for r in rows[:3]:
        extra = {k: v for k, v in r.items()
                 if k in ("industry_fit", "demand_signal", "urgency") and v is not None}
        print(json.dumps({"name": r["name"], "score": r["score"],
                          "confidence": r["confidence"], **extra}, ensure_ascii=False))


if __name__ == "__main__":
    main()