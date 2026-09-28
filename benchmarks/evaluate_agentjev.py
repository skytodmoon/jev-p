"""AgentJev-0.6B（System One 决策模型）接入 JevP 标准三场景评测。

与 evaluate.py（LLM logits 路线）/ evaluate_laya.py（Laya 判别式）同口径：
同一组 questions + 同一规则金标准 + 同一组指标，可直接横向对比。

前置：AgentJev 服务需已启动（默认 http://127.0.0.1:8149）。

    python -m jev_service.server --checkpoint ... --port 8149 --device cpu|cuda:0
    python benchmarks/evaluate_agentjev.py --repeat 3
    AGENTJEV_URL=http://<gpu-host>:8149 python benchmarks/evaluate_agentjev.py
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
for p in (_ROOT, _THIS_DIR, os.path.join(os.path.dirname(_ROOT), "agent-jev")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from scenarios import SCENARIOS  # noqa: E402
from agentjev_client import AgentJev  # noqa: E402

# JevP protocol 类型 -> AgentJev 类型
_TYPE_MAP = {"noul": "boolean", "choice": "choice", "score": "score"}


def to_agentjev_questions(questions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 JevP protocol questions 转成 AgentJev evaluate API 的 questions 列表。"""
    out: List[Dict[str, Any]] = []
    for name, q in questions.items():
        raw = q.to_dict()
        item: Dict[str, Any] = {
            "id": name,
            "type": _TYPE_MAP[q.type],
            "question": raw["instructions"],
        }
        if q.type == "noul":
            # JevP noul criteria {true,false 描述} 与 AgentJev boolean criteria 同构
            c = raw.get("criteria") or {"true": "The statement is true.",
                                        "false": "The statement is false."}
            item["criteria"] = {"true": c.get("true", ""), "false": c.get("false", "")}
        elif q.type == "choice":
            item["options"] = raw["criteria"]
        else:  # score
            item["levels"] = list(raw["criteria"])
        out.append(item)
    return out


def prediction(atype: str, ans: Dict[str, Any]):
    """从 AgentJev answer 提取 (预测值, 置信度)。"""
    if atype == "noul":
        dist = ans.get("distribution") or {}
        p_true = float(dist.get("true", 0.0))
        return bool(ans.get("value")), max(p_true, 1.0 - p_true)
    if atype == "choice":
        dist = ans.get("distribution") or {}
        best = max(dist, key=dist.get) if dist else None
        conf = float(dist.get(best, 0.0)) if best is not None else 0.0
        return best, conf
    # score
    level = ans.get("level")
    expected = ans.get("score")  # 连续期望分
    dist = ans.get("distribution") or {}
    conf = float(max(dist.values())) if dist else 0.0
    return (round(float(expected)) if expected is not None else level), conf


def evaluate_scenario(jev, scenario, repeat: int, warmup: int = 1) -> Dict[str, Any]:
    questions = scenario.build_questions()
    ev_questions = to_agentjev_questions(questions)
    qtypes = {k: v.type for k, v in questions.items()}

    frames = scenario.frames()
    for _ in range(max(0, warmup)):
        state = _state_str(frames[0])
        jev.evaluate(state, ev_questions)

    decisions = 0
    correct_conf: List[float] = []
    wrong_conf: List[float] = []
    lat_ms: List[float] = []
    per_frame_modes: List[float] = []
    for state in frames:
        exp = scenario.expected(state)
        trial_flags: List[bool] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            res = jev.evaluate(_state_str(state), ev_questions)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            ans_list = res.get("answers") or []
            first = True
            for ans_item in ans_list:
                key = ans_item.get("id")
                if key not in exp or qtypes.get(key) not in _TYPE_MAP:
                    continue
                pred, conf = prediction(qtypes[key], ans_item)
                ok = pred == exp[key]
                if first:
                    trial_flags.append(ok)
                    first = False
                decisions += 1
                (correct_conf if ok else wrong_conf).append(conf)
        if trial_flags:
            mode_count = max(trial_flags.count(True), trial_flags.count(False))
            per_frame_modes.append(mode_count / repeat)

    n_correct = len(correct_conf)
    n_total = n_correct + len(wrong_conf)
    return {
        "frames": len(frames),
        "decisions": n_total,
        "accuracy": round(n_correct / n_total, 4) if n_total else None,
        "consistency": round(statistics.mean(per_frame_modes), 4) if per_frame_modes else None,
        "conf_correct_mean": round(statistics.mean(correct_conf), 4) if correct_conf else None,
        "conf_wrong_mean": round(statistics.mean(wrong_conf), 4) if wrong_conf else None,
        "latency_ms_per_decision": round(statistics.mean(lat_ms) / max(len(questions), 1), 2)
        if lat_ms else None,
    }


def _state_str(frame: Any) -> Any:
    # client 会自动把 dict/list 转 JSON 字符串
    return frame


def _fmt(v) -> str:
    return "n/a" if v is None else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="AgentJev 接入 JevP 三场景评测")
    ap.add_argument("--scenarios", default="mario,guardrail,ticket")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--url", default=os.environ.get("AGENTJEV_URL", "http://127.0.0.1:8149"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jev = AgentJev(args.url)
    info = jev.info()
    print(f"[eval-agentjev] {info.get('model')} @ {args.url} "
          f"types={info.get('types')} repeat={args.repeat}")

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    results: Dict[str, Any] = {"scenarios": {}}
    header = (f"{'场景':<12}{'决策数':>6}{'准确率':>9}{'一致性':>8}"
              f"{'conf对':>9}{'conf错':>9}{'ms/决策':>9}")
    print(header)
    print("-" * len(header))
    for name in names:
        scenario = SCENARIOS[name]
        r = evaluate_scenario(jev, scenario, args.repeat, args.warmup if hasattr(args, 'warmup') else 1)
        results["scenarios"][name] = r
        print(f"{name:<12}{r['decisions']:>6}{_fmt(r['accuracy']):>9}"
              f"{_fmt(r['consistency']):>8}{_fmt(r['conf_correct_mean']):>9}"
              f"{_fmt(r['conf_wrong_mean']):>9}{_fmt(r['latency_ms_per_decision']):>9}")

    results["meta"] = {
        "model": info.get("model", "AgentJev"),
        "url": args.url,
        "repeat": args.repeat,
        "note": "金标准为规则标注（可辩解）；AgentJev 官方注明概率为模型分布，域校准不保证。",
    }
    if args.out:
        out_path = args.out
    else:
        os.makedirs(os.path.join(_THIS_DIR, "..", "results"), exist_ok=True)
        out_path = os.path.join(_THIS_DIR, "..", "results",
                                f"eval-agentjev-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()