"""Laya（Apache-2.0 开源决策模型，Jev 同赛道平替）对比评测。

复用 JevP 场景（benchmarks/scenarios 的 mario / guardrail / ticket，
含规则金标准），把同一组 questions 喂给 Laya 的 Router.predict，
输出与 `evaluate.py` 同格式指标，便于横向对比：

    JevP 方案（LLM logits 约束打分） vs Laya（非自回归判别式模型）

用法：
    python benchmarks/evaluate_laya.py                # 首跑会自动下载 checkpoint
    python benchmarks/evaluate_laya.py --scenarios mario,guardrail --device cpu
    python benchmarks/evaluate_laya.py --no-preload   # 懒加载（按语言路由）
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


def _prediction(atype: str, answer: Dict[str, Any], expected):
    """把 Laya 单个答案转成可判定预测（与金标准同型）。"""
    if atype == "choice":
        choice = answer.get("choice")
        probabilities = answer.get("probabilities") or {}
        conf = answer.get("confidence")
        if conf is None and probabilities:
            conf = max(probabilities.values())
        return choice, conf or 0.0
    if atype == "noul":
        noul = answer.get("noul", 0.0)
        conf = max(noul, 1.0 - noul)
        return bool(noul >= 0.5), conf
    score = answer.get("score", 0.0)
    conf = answer.get("confidence") or 0.0
    return round(score), conf


def evaluate_scenario(
    router,
    scenario,
    repeat: int,
    warmup: int = 1,
) -> Dict[str, Any]:
    questions = scenario.build_questions()
    qdict = {k: v.to_dict() for k, v in questions.items()}
    qtypes = {k: v.type for k, v in questions.items()}

    decisions = 0
    schema_valid = True
    correct_conf: List[float] = []
    wrong_conf: List[float] = []
    masses: List[float] = []
    latency_ms_all: List[float] = []
    per_frame_modes: List[float] = []
    routed: Dict[str, int] = {}

    frames = scenario.frames()
    for _ in range(max(0, warmup)):
        router.predict(_state(frames[0]), qdict)

    for state in frames:
        exp = scenario.expected(state)
        trial_flags: List[bool] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            result = router.predict(_state(state), qdict)
            latency_ms_all.append((time.perf_counter() - t0) * 1000.0)
            answers = result.get("answers", {})
            schema_valid = schema_valid and set(answers) == set(questions)
            agent = (result.get("routing") or {}).get("model", "?")
            routed[agent] = routed.get(agent, 0) + 1

            first = True
            for key, exp_val in exp.items():
                atype = qtypes[key]
                if atype not in ("choice", "noul", "score") or key not in answers:
                    continue
                pred, conf = _prediction(atype, answers[key], exp_val)
                ok = pred == exp_val
                if first:
                    trial_flags.append(ok)
                    first = False
                decisions += 1
                (correct_conf if ok else wrong_conf).append(conf)
                mass = answers[key].get("mass") or answers[key].get("raw_mass")
                if mass is not None:
                    masses.append(float(mass))
        if trial_flags:
            mode_count = max(trial_flags.count(True), trial_flags.count(False))
            per_frame_modes.append(mode_count / len(trial_flags))

    n_correct = len(correct_conf)
    n_total = n_correct + len(wrong_conf)
    per_decision_ms = (statistics.mean(latency_ms_all) / max(len(questions), 1)
                       if latency_ms_all else None)
    return {
        "frames": len(frames),
        "decisions": n_total,
        "schema_valid": bool(schema_valid),
        "accuracy": round(n_correct / n_total, 4) if n_total else None,
        "consistency": round(statistics.mean(per_frame_modes), 4) if per_frame_modes else None,
        "conf_correct_mean": round(statistics.mean(correct_conf), 4) if correct_conf else None,
        "conf_wrong_mean": round(statistics.mean(wrong_conf), 4) if wrong_conf else None,
        "mass_mean": round(statistics.mean(masses), 4) if masses else None,
        "latency_ms_per_decision": round(per_decision_ms, 2) if per_decision_ms else None,
        "routed_to": routed,
        "routed_detail": str(routed),
    }


def _state(frame: Any) -> Any:
    """Laya 接受文本或 dict；JSON dict 直接传，否则字符串化。"""
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Laya(Jev 平替) 对比评测")
    ap.add_argument("--scenarios", default="mario,guardrail,ticket")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    ap.add_argument("--no-preload", dest="preload", action="store_false", default=True,
                    help="懒加载 checkpoint（首次按语言路由下载）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import laya
    from laya import Router

    print(f"[eval-laya] laya={laya.__version__} device={args.device} "
          f"preload={args.preload} scenarios={args.scenarios} repeat={args.repeat}")
    print("[eval-laya] 首次运行会自动从 HF 下载 checkpoint（421M 英文 / 322M 多语言）...")
    router = Router(preload=args.preload, device=args.device)

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    results: Dict[str, Any] = {"scenarios": {}}
    header = (f"{'场景':<12}{'决策数':>6}{'schema':>7}{'准确率':>9}{'一致性':>8}"
              f"{'conf对':>9}{'conf错':>9}{'ms/决策':>9}{'路由':>12}")
    print(header)
    print("-" * len(header))
    for name in names:
        scenario = SCENARIOS[name]
        r = evaluate_scenario(router, scenario, args.repeat, args.warmup)
        results["scenarios"][name] = r
        print(f"{name:<12}{r['decisions']:>6}{'✅' if r['schema_valid'] else '❌':>7}"
              f"{r['accuracy']:>9}{r['consistency']:>8}"
              f"{r['conf_correct_mean']:>9}{r['conf_wrong_mean']:>9}"
              f"{r['latency_ms_per_decision']:>9}{r['routed_detail']:>12}")

    results["meta"] = {
        "model": "laya (Router: english 421M / multilingual 322M / typed-decisions)",
        "device": args.device,
        "repeat": args.repeat,
        "note": "金标准为规则标注（可辩解）；Laya 输出 confidence/probabilities 与 JevP 定义略有差异，对比仅供参考。",
    }
    if args.out:
        out_path = args.out
    else:
        os.makedirs(os.path.join(_THIS_DIR, "..", "results"), exist_ok=True)
        out_path = os.path.join(_THIS_DIR, "..", "results",
                                f"eval-laya-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()