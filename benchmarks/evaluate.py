"""综合评测：多场景 × 多指标打分（对 JevP=类 Jev 方案做系统评定）。

指标（默认连本地推理服务跑真实数据）：
- schema_valid       类型安全：程序化拼装的答案永远结构合法；
- accuracy           与**规则金标准**的一致率（choice 精确 / noul 阈值 0.5 / score 四舍五入）；
- consistency        同一帧重复采样的主决策一致率（模型/服务端稳定性）；
- calibrated_signal  正确时平均置信度 vs 错误时平均置信度（应 "对>错"，即置信度是正确性的代理）；
- mass_mean          raw_mass 均值（候选在 top-k 中的质量占比）；
- latency_ms        每决策平均延迟。

金标准说明：mario/guardrail/ticket 场景均内置"可辩解金标准"（规则标注，
非苛求的学术标注），用于趋势性评定而非宣称绝对精度。

用法：
    python benchmarks/evaluate.py                     # 默认全部场景 × repeat=3
    python benchmarks/evaluate.py --scenarios mario,guardrail --repeat 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from scenarios import SCENARIOS  # noqa: E402
from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402


def _prediction(atype: str, payload: Dict[str, Any], expected):
    """把答案转成可判定的预测值（与金标准同型）。"""
    if atype == "choice":
        return payload["choice"], payload.get("confidence", 0.0)
    if atype == "noul":
        conf = max(payload.get("noul", 0.0), 1.0 - payload.get("noul", 0.0))
        return bool(payload.get("noul", 0.0) >= 0.5), conf
    # score
    return round(payload.get("score", 0.0)), payload.get("confidence", 0.0)


def evaluate_scenario(
    engine: DecisionEngine,
    scenario,
    repeat: int,
    warmup: int = 1,
) -> Dict[str, Any]:
    questions = scenario.build_questions()
    qtypes = {k: v.type for k, v in questions.items()}

    rows: List[Dict[str, Any]] = []
    decisions = 0
    latency_ms_all: List[float] = []
    correct_conf: List[float] = []
    wrong_conf: List[float] = []
    masses: List[float] = []
    schema_valid = True
    per_frame_modes: List[float] = []

    frames = scenario.frames()
    # warmup
    for _ in range(max(0, warmup)):
        engine.decide(frames[0], questions)

    for state in frames:
        exp = scenario.expected(state)
        trial_results: List[Dict[str, Any]] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            answers = engine.decide(state, questions)
            latency_ms_all.append((time.perf_counter() - t0) * 1000.0)
            schema_valid = schema_valid and set(answers) == set(questions)
            trial: Dict[str, Any] = {"correct": [], "confs": [], "masses": []}
            for key, exp_val in exp.items():
                atype = qtypes[key]
                if atype not in ("choice", "noul", "score"):
                    continue
                payload = answers[key].payload
                pred, conf = _prediction(atype, payload, exp_val)
                ok = pred == exp_val
                trial["correct"].append(ok)
                trial["confs"].append(conf)
                trial["masses"].append(payload.get("raw_mass", 0.0))
                decisions += 1
            trial_results.append(trial)

        # 一致性：主决策（第一个判分键）在 repeat 次中的 mode 占比
        first_key = next(k for k in exp if qtypes[k] in ("choice", "noul", "score"))
        preds = [r["correct"][0] for r in trial_results]
        # 以"是否与金标准一致"的众数占比近似主决策一致性
        mode_count = max(preds.count(True), preds.count(False))
        per_frame_modes.append(mode_count / repeat)

        for tr in trial_results:
            for ok, conf, mass in zip(tr["correct"], tr["confs"], tr["masses"]):
                (correct_conf if ok else wrong_conf).append(conf)
                masses.append(mass)

    n_correct = len(correct_conf)
    n_total = n_correct + len(wrong_conf)
    summary = {
        "frames": len(frames),
        "decisions": n_total,
        "schema_valid": bool(schema_valid),
        "accuracy": round(n_correct / n_total, 4) if n_total else None,
        "consistency": round(statistics.mean(per_frame_modes), 4) if per_frame_modes else None,
        "conf_correct_mean": round(statistics.mean(correct_conf), 4) if correct_conf else None,
        "conf_wrong_mean": round(statistics.mean(wrong_conf), 4) if wrong_conf else None,
        "mass_mean": round(statistics.mean(masses), 4) if masses else None,
        "latency_ms_per_decision": round(
            statistics.mean(latency_ms_all) / len(questions), 2) if latency_ms_all else None,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="JevP 综合评测（默认连本地推理服务）")
    ap.add_argument("--scenarios", default="mario,guardrail,ticket")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--backend", default=os.environ.get("JEVP_BACKEND", "auto"))
    ap.add_argument("--model", default=os.environ.get("JEVP_MODEL"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    backend = auto_backend(model=args.model, backend=args.backend)
    engine = DecisionEngine(backend=backend, tokenizer=backend.tokenizer)
    print(f"[eval] backend={type(backend).__name__} model={backend.model_id} "
          f"scenarios={names} repeat={args.repeat}\n")

    results: Dict[str, Any] = {"scenarios": {}}
    header = (f"{'场景':<12}{'决策数':>6}{'schema':>7}{'准确率':>9}{'一致性':>8}"
              f"{'conf对':>9}{'conf错':>9}{'mass':>8}{'ms/决策':>9}")
    print(header)
    print("-" * len(header))
    for name in names:
        scenario = SCENARIOS[name]
        r = evaluate_scenario(engine, scenario, args.repeat, args.warmup)
        results["scenarios"][name] = r
        print(f"{name:<12}{r['decisions']:>6}{'✅' if r['schema_valid'] else '❌':>7}"
              f"{r['accuracy']:>9}{r['consistency']:>8}"
              f"{r['conf_correct_mean']:>9}{r['conf_wrong_mean']:>9}"
              f"{r['mass_mean']:>8}{r['latency_ms_per_decision']:>9}")

    results["meta"] = {
        "backend": type(backend).__name__,
        "model": backend.model_id,
        "repeat": args.repeat,
        "note": "金标准为规则标注（可辩解，非学术精确标注）；accuracy 用于趋势评定。",
    }
    if args.out:
        out_path = args.out
    else:
        os.makedirs(os.path.join(_THIS_DIR, "..", "results"), exist_ok=True)
        out_path = os.path.join(_THIS_DIR, "..", "results",
                                f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()