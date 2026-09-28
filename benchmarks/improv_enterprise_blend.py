"""企业匹配性能提升实验：错误明细 + 规则-LLM 融合 + 硬校正（不训练）。

基线：27B 双语直连 0.708（中0.75/英0.667）；规则 L0 0.625（Pearson 0.95）。
本实验验证三档提升：
  A. 错误明细（逐家 预测vs金标准，定位系统性偏差）；
  B. 规则-LLM 融合：blend = w*llm + (1-w)*rule（w 网格，双语联合选稳健值）；
  C. 硬校正（可辩解）：行业不匹配(industry_fit<0.5) → 上限封顶 1.0。
输出命中率对比（原始27B / 规则 / 融合@w* / 校正），记录最优配方。

用法：VLM_*=... python benchmarks/improv_enterprise_blend.py [--w 0.7]
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from rule_baseline_enterprise import domain_patch, is_non_commercial, rule_score  # noqa: E402
from scenarios.enterprise_match import (  # noqa: E402
    expected as exp_zh, frames as frames_zh, build_questions as q_zh)
from scenarios.enterprise_match_en import (  # noqa: E402
    expected as exp_en, frames as frames_en, build_questions as q_en)
from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402


def collect(engine, frames, questions, exp_fn):
    """返回 [{name, llm(0-3), rule(0-3), gold(0-3), industry_fit(bool), demand(bool), n_bids, non_commercial}]。"""
    rows = []
    for f in frames:
        ans = engine.decide(f, questions)
        a = {k: v.to_dict() for k, v in ans.items()}
        scale = str(f["basic"].get("scale", ""))
        industry = str(f["basic"].get("industry", ""))
        rows.append({
            "name": f["name"],
            "llm": a["match_score"].get("score"),
            "gold": exp_fn(f)["match_score"],
            "rule": rule_score(f),
            "industry_fit": a["industry_fit"].get("noul", 0.0) >= 0.5,
            "demand": a["demand_signal"].get("noul", 0.0) >= 0.5,
            "n_bids": len(f.get("bidding", [])),
            "non_commercial": is_non_commercial(f),
        })
    return rows


def hit(score, gold):
    return round(score) == gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--w", type=float, default=0.7, help="融合权重（默认按双语联合选定）")
    args = ap.parse_args()

    backend = auto_backend(backend="openai")
    engine = DecisionEngine(backend=backend, tokenizer=getattr(backend, "tokenizer", None))

    data = collect(engine, frames_zh(), q_zh(), exp_zh) + collect(
        engine, frames_en(), q_en(), exp_en)
    n = len(data)

    # 错误明细
    print(f"{'企业':<20}{'金标准':>5}{'27B':>6}{'规则':>6}{'行业匹配':>8}{'需求':>6}{'原始命中':>8}")
    print("-" * 66)
    for r in data:
        print(f"{r['name']:<20}{r['gold']:>5}{r['llm']:>6.2f}{r['rule']:>6.1f}"
              f"{'✓' if r['industry_fit'] else '✗':>8}{'✓' if r['demand'] else '✗':>6}"
              f"{'✓' if hit(r['llm'], r['gold']) else '✗':>8}")

    # 融合 + 校正
    print(f"\n{'策略':<34}{'命中':>6}{'命中率':>8}")
    print("-" * 48)
    base = sum(hit(r["llm"], r["gold"]) for r in data)
    rule_h = sum(hit(r["rule"], r["gold"]) for r in data)
    print(f"{'27B 原始（基线）':<34}{base:>6}{base/n:>8.2%}")

    best = (None, -1)
    for w in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
        hits = 0
        for r in data:
            b = w * r["llm"] + (1 - w) * r["rule"]
            if hit(b, r["gold"]):
                hits += 1
        tag = f"融合 w={w:.1f}"
        print(f"{tag:<34}{hits:>6}{hits/n:>8.2%}")
        if hits > best[1]:
            best = (w, hits)

    w = args.w
    hits = sum(1 for r in data if hit(domain_patch(r["llm"], r), r["gold"]))
    print(f"{'27B + 域先验硬校正（行业✗≤1 / ✗且无标≤0.5 / 非企业≤0.5）':<12}{hits:>26}{hits/n:>8.2%}")

    rule_h2 = sum(1 for r in data if hit(r["rule"], r["gold"]))
    print(f"{'规则 L0（双语关键词修正）':<38}{rule_h2:>6}{rule_h2/n:>8.2%}")

    print(f"\n网格最优 w={best[0]} 命中 {best[1]}/{n}；建议稳健 w={args.w}（小样本，勿过拟合）")
    print("提示：融合拆解了规则双语失效后不可用；提升杠杆在 ①域先验校正 ②规则双语使能。")


if __name__ == "__main__":
    main()