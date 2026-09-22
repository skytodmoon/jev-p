"""本地 demo：默认测试基线——7B 模型 × Apple Silicon（Mac mini 4 / 16GB）。

用法：
    python examples/demo.py
        # auto 后端：Apple Silicon 自动用 MLX（mlx-community/Qwen2.5-7B-Instruct-4bit）
        # 其他平台回退 transformers（Qwen/Qwen2.5-7B-Instruct，需 torch）
    JEVP_BACKEND=transformers JEVP_MODEL=Qwen/Qwen2.5-7B-Instruct python examples/demo.py

问题设计注意：每问应是一个"懂行的人几秒内能做出的直觉判断"（原子化）。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK：强制 UTF-8 输出
for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402
from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion  # noqa: E402


def main() -> None:
    backend = auto_backend(
        model=os.environ.get("JEVP_MODEL"),
        backend=os.environ.get("JEVP_BACKEND", "auto"),
    )
    print(f"[JevP demo] backend={type(backend).__name__} model={backend.model_id}")
    if type(backend).__name__ == "MLXBackend":
        print("[JevP demo] 提示：Apple Silicon 建议先 `pip install -e '.[mlx]'`")

    engine = DecisionEngine(backend=backend, tokenizer=backend.tokenizer)

    state = "客户：我上周的订单还没发货，你们到底什么时候处理？我已经等不下去了。"

    questions = {
        "department": ChoiceQuestion(
            instructions="这条工单应该由哪个团队处理？",
            criteria={
                "billing": "账单、退款、发票相关",
                "technical": "产品故障、报错、无法使用",
                "sales": "售前咨询、报价、方案",
            },
        ),
        "frustration": ScoreQuestion(
            instructions="客户当前的挫败程度如何？",
            criteria=["平静", "不满", "非常愤怒"],
        ),
        "is_urgent": NoulQuestion(
            instructions="这条消息是否紧急、需要立即处理？",
            criteria={"true": "明确的时间敏感或影响生产", "false": "无紧迫表述"},
        ),
    }

    answers = engine.decide(state, questions)
    print("\n== 决策结果 ==")
    for name, ans in answers.items():
        print(f"\n[{name}]")
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))

    # 简单路由建议（选择性分类）
    print("\n== 路由建议 ==")
    for name, ans in answers.items():
        conf = ans.payload.get("confidence") or ans.payload.get("noul", 0.0)
        mass = ans.payload.get("raw_mass", 0.0)
        if mass < 0.2:
            advice = "⚠️ raw_mass 过低：模型对这个问题没底，建议转人工/不自动执行"
        elif conf >= 0.9:
            advice = "自动执行"
        elif conf >= 0.7:
            advice = "升级给推理模型复核"
        else:
            advice = "转人工"
        print(f"  {name}: conf={conf:.3f} mass={mass:.4f} -> {advice}")


if __name__ == "__main__":
    main()