"""基准测试：默认面向 Mac mini 4（M4 / 16GB）+ 7B 基线的目标机器。

测量目标：
- 不同"单批问题数"（字段数）下的端到端延迟（中位数）；
- schema 有效性（程序化拼装恒为 True，用于对冲"朴素生成 JSON"对比）；
- 峰值内存（max RSS，仅 macOS/Linux）。

用法：
    python benchmarks/benchmark.py                          # 默认 4,8,16,28 字段
    python benchmarks/benchmark.py --fields 4,16 --repeat 5
    JEVP_BACKEND=transformers python benchmarks/benchmark.py   # 强制用 torch 后端

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402
from jevp.protocol import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    validate_questions,  # noqa: F401（校验每批构造的问题）
)  # noqa: E402

STATE = (
    "客服工单：客户报告 Stripe 支付回调整整 3 天没有成功，"
    "订单 499 美元已支付，要求尽快处理并退款。"
)

CHOICE_CRITERIA = {
    "billing": "账单、退款、发票相关",
    "technical": "产品故障、报错、无法使用",
    "account": "账号与登录",
    "sales": "售前咨询、报价",
    "other": "其他",
}


def make_questions(n_choice: int) -> Dict[str, Any]:
    """n_choice 个 choice + 1 个 noul + 1 个 score ≈ n_choice+2 个字段。"""
    qs: Dict[str, Any] = {}
    for i in range(n_choice):
        qs[f"dept_{i}"] = ChoiceQuestion(
            instructions=f"工单 {i}：应该路由到哪个部门？",
            criteria=CHOICE_CRITERIA,
        )
    qs["urgent"] = NoulQuestion(instructions="这条工单是否紧急、需要立即干预？")
    qs["frustration"] = ScoreQuestion(
        instructions="客户的不满程度如何？",
        criteria=["平静", "不满", "非常愤怒"],
    )
    _ = validate_questions({k: v.to_dict() for k, v in qs.items()})  # 协议自检
    return qs


def run_case(engine: DecisionEngine, questions: Dict[str, Any], repeat: int) -> float:
    times: list[float] = []
    last = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        last = engine.decide(STATE, questions)
        times.append(time.perf_counter() - t0)
    assert set(last) == set(questions), "schema 有效性（程序化拼装恒真）"
    return statistics.median(times)


def max_rss_mb() -> float | None:
    if platform.system() in ("Darwin", "Linux"):
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024.0 if platform.system() == "Linux" else kb / (1024.0 * 1024.0)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="JevP 基准测试（目标机器：Mac mini 4 / 16GB / 7B）")
    ap.add_argument("--fields", default="4,8,16,28", help="每批字段数列表，逗号分隔")
    ap.add_argument("--repeat", type=int, default=3, help="每档重复次数（取中位数）")
    ap.add_argument("--warmup", type=int, default=1, help="预热轮数")
    ap.add_argument("--backend", default=os.environ.get("JEVP_BACKEND", "auto"))
    ap.add_argument("--model", default=os.environ.get("JEVP_MODEL"))
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    field_counts = [int(x) for x in args.fields.split(",") if x.strip()]
    print(f"[bench] backend={args.backend} model={args.model or '(默认 7B)'} "
          f"fields={field_counts} repeat={args.repeat}")

    backend = auto_backend(model=args.model, backend=args.backend)
    engine = DecisionEngine(backend=backend, tokenizer=backend.tokenizer)
    print(f"[bench] 实际后端: {type(backend).__name__} ({backend.model_id})")

    # 预热（模型加载/首轮分配）
    warm = make_questions(4)
    for _ in range(max(0, args.warmup)):
        engine.decide(STATE, warm)

    results = {"factors": {}}
    print("\n字段数  | 中位延迟(s) | 每决策(ms) | schema-valid | 后端子类型")
    print("-" * 64)
    for n in field_counts:
        questions = make_questions(n)
        med = run_case(engine, questions, args.repeat)
        per = med * 1000.0 / len(questions)
        print(f"{len(questions):5d}  | {med:10.3f} | {per:9.1f} | {'✅' if True else '❌'} "
              f"| {type(backend).__name__}")
        results["factors"][str(len(questions))] = {
            "median_s": round(med, 4),
            "ms_per_decision": round(per, 2),
            "schema_valid": True,
        }

    results["meta"] = {
        "backend": type(backend).__name__,
        "model": backend.model_id,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "max_rss_mb": max_rss_mb(),
        "note": "首版 MLX 后端为逐条前向；Transformers 后端为一次 batch 前向。",
    }
    print(f"\n平台: {results['meta']['platform']} / {results['meta']['machine']}")
    if results["meta"]["max_rss_mb"] is not None:
        print(f"峰值内存(max RSS): {results['meta']['max_rss_mb']:.0f} MB")

    if args.out:
        out_path = args.out
    else:
        os.makedirs("results", exist_ok=True)
        out_path = f"results/bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()