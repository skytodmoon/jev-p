"""基准测试：默认面向 Mac mini 4（M4 / 16GB）+ 7B 基线的目标机器。

场景驱动：默认 **mario**（Super Mario Bros. 1-1 快速决策场景），可选 ticket。

测量目标：
- 每帧决策的端到端延迟（中位数）；
- schema 有效性（程序化拼装恒为 True，用于对冲"朴素生成 JSON"对比）；
- 峰值内存（max RSS，仅 macOS/Linux）。

用法：
    python benchmarks/benchmark.py                          # 默认 mario 场景
    python benchmarks/benchmark.py --scenario ticket --fields 4,16
    python benchmarks/benchmark.py --repeat 5
    JEVP_BACKEND=transformers python benchmarks/benchmark.py

说明：首版 MLX 后端为逐条前向（每问题一次 forward）；Transformers 后端为
一次 batch 前向。两者数字因此不完全可比，报告会标注后端类型。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from typing import Any, Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from scenarios import load  # noqa: E402
from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402
from jevp.protocol import validate_questions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="JevP 基准测试（目标：Mac mini 4 / 16GB / 7B）")
    ap.add_argument("--scenario", default="mario", help="mario（默认）| ticket | 自定义")
    ap.add_argument("--fields", default="4,8,16,28", help="ticket 场景的字段数档位，逗号分隔")
    ap.add_argument("--repeat", type=int, default=3, help="每档重复次数（取中位数）")
    ap.add_argument("--warmup", type=int, default=1, help="预热轮数")
    ap.add_argument("--backend", default=os.environ.get("JEVP_BACKEND", "auto"))
    ap.add_argument("--model", default=os.environ.get("JEVP_MODEL"))
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    scenario = load(args.scenario)
    print(f"[bench] 场景={scenario.NAME}（{scenario.DESC}） backend={args.backend} "
          f"model={args.model or '(默认 7B)'} repeat={args.repeat}")

    # 预热：构造引擎 + 一轮判断
    backend = auto_backend(model=args.model, backend=args.backend)
    engine = DecisionEngine(backend=backend, tokenizer=backend.tokenizer)
    print(f"[bench] 实际后端: {type(backend).__name__} ({backend.model_id})")

    def run(frames: list, questions: Dict[str, Any]) -> Dict[str, float]:
        # 预热
        for _ in range(max(0, args.warmup)):
            validate_questions({k: v.to_dict() for k, v in questions.items()})
            engine.decide(frames[0], questions)
        per_frame: list[float] = []
        valid = True
        last = {}
        for state in frames:
            median = _median_of(lambda: engine.decide(state, questions), args.repeat)
            per_frame.append(median)
            last = engine.decide(state, questions)
            valid = valid and set(last) == set(questions)
        total = sum(per_frame)
        return {
            "frames": len(frames),
            "fields_per_frame": len(questions),
            "median_s_per_frame": total / len(frames),
            "ms_per_decision": total * 1000.0 / (len(frames) * len(questions)),
            "total_s_all_frames": total,
            "schema_valid": valid,
        }

    results: Dict[str, Any] = {"scenarios": {}}
    print(f"\n{'场景/档位':<28} 帧数 | 每帧(s) | 每决策(ms) | schema | 后端")
    print("-" * 76)

    if scenario.NAME == "mario":
        questions = scenario.build_questions()
        r = run(scenario.frames(), questions)
        key = "mario(1-1 关卡)"
        print(f"{key:<28} {r['frames']:>2}   | {r['median_s_per_frame']:7.3f} | "
              f"{r['ms_per_decision']:9.1f} | {'✅' if r['schema_valid'] else '❌'} | {type(backend).__name__}")
        results["scenarios"][key] = r
    else:
        field_counts = [int(x) for x in args.fields.split(",") if x.strip()]
        for n in field_counts:
            questions = scenario.build_questions(n_choice=n)
            frames = scenario.frames(n_choice=n)
            r = run(frames, questions)
            key = f"ticket({len(questions)} 字段)"
            print(f"{key:<28} {r['frames']:>2}   | {r['median_s_per_frame']:7.3f} | "
                  f"{r['ms_per_decision']:9.1f} | {'✅' if r['schema_valid'] else '❌'} | {type(backend).__name__}")
            results["scenarios"][key] = r

    results["meta"] = {
        "scenario": scenario.NAME,
        "backend": type(backend).__name__,
        "model": backend.model_id,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "max_rss_mb": _max_rss_mb(),
        "note": "首版 MLX 后端为逐条前向；Transformers 后端为一次 batch 前向。",
    }
    print(f"\n平台: {results['meta']['platform']} / {results['meta']['machine']}")
    if results["meta"]["max_rss_mb"] is not None:
        print(f"峰值内存(max RSS): {results['meta']['max_rss_mb']:.0f} MB")

    if args.out:
        out_path = args.out
    else:
        os.makedirs(os.path.join(_THIS_DIR, "..", "results"), exist_ok=True)
        out_path = os.path.join(_THIS_DIR, "..", "results",
                                f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


def _median_of(fn, repeat: int) -> float:
    times = [fn_and_measure(fn) for _ in range(repeat)]
    return statistics.median(times)


def fn_and_measure(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _max_rss_mb() -> float | None:
    if platform.system() in ("Darwin", "Linux"):
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024.0 if platform.system() == "Linux" else kb / (1024.0 * 1024.0)
    return None


if __name__ == "__main__":
    main()