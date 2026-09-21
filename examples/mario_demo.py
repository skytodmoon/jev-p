"""马里奥关卡演示：用 JevP 做 Super Mario Bros. 1-1 的"宏观动作决策"。

场景设计（System One 定位）：
- JevP 是**快决策层**：每 0.1–0.6s 输出一个宏观动作（跑/跳/蹲/等/回头）+ 概率
- 高频（60fps）的精确跳跃/碰撞交给确定性控制器（本演示只输出动作建议）
- 状态由"采样器"提供：真机接入模拟器（FCEUX Lua / Libretro / jsnes）时，
  把内存/画面转成同样的结构化字段即可，本演示使用合成关卡快照

用法（在目标机器：Mac mini 4 / 16GB / 7B）：
    python examples/mario_demo.py
    JEVP_BACKEND=transformers python examples/mario_demo.py
"""
from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, os.path.join(_ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scenarios.mario import ACTION_BUTTONS, build_questions, describe, frames  # noqa: E402
from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402
from jevp.protocol import ChoiceQuestion, NoulQuestion  # noqa: E402


def pick_action(answers) -> str:
    a = answers["action"].to_dict()
    return a.get("choice", "wait")


def main() -> None:
    backend = auto_backend(
        model=os.environ.get("JEVP_MODEL"),
        backend=os.environ.get("JEVP_BACKEND", "auto"),
    )
    print(f"[mario] backend={type(backend).__name__} model={backend.model_id}\n")

    engine = DecisionEngine(backend=backend, tokenizer=backend.tokenizer)
    questions = build_questions()

    action_seq = []
    print("帧 | 场景摘要 | 推荐动作 | confidence | raw_mass | 路由")
    print("-" * 88)
    for idx, state in enumerate(frames()):
        answers = engine.decide(state, questions)
        action = pick_action(answers)
        action_seq.append(action)
        act = answers["action"].to_dict()
        conf = act.get("confidence", 0.0)
        mass = act.get("raw_mass", 0.0)
        danger = answers["danger"].to_dict().get("score", 0.0)

        if mass < 0.2:
            route = "⚠️ 低质量，转人工/不执行"
        elif conf >= 0.9:
            route = "自动执行"
        elif conf >= 0.7:
            route = "复核"
        else:
            route = "不执行"
        print(f"{idx}  | {describe(state, idx)}")
        print(f"      -> {action:<8} conf={conf:.3f} mass={mass:.4f} danger={danger:.2f} | {route}")

    print("\n== 动作序列（可交给模拟器按钮映射）==")
    for i, a in enumerate(action_seq):
        print(f"  frame{i}: {a:<8} buttons={ACTION_BUTTONS.get(a, [])}")

    print("\n== 各问题明细（帧 1：敌人逼近）==")
    answers = engine.decide(frames()[1], questions)
    for name, ans in answers.items():
        print(f"  [{name}] {json.dumps(ans.to_dict(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()