"""双语企业匹配评测：同一产品 × 中英两套企业候选 × 多个后端 → 分语言对比表。

目标：验证"未来模型兼顾中英"的评测基准。
- 中文侧：`enterprise_match`（8 家中企，中文档案）
- 英文侧：`enterprise_match_en`（6 家海外企业，英文档案）
- 后端：router（语言感知路由，主角）/ openai(27B) / laya（判别式） / agentjev（可选，需服务）

指标（复用 evaluate.py 同口径）：档位命中率（round(score)==金标准 0-3）。
用法：
    VLM_*=... python benchmarks/eval_bilingual_enterprise.py [--with-agentjev]
"""
from __future__ import annotations

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

from evaluate import evaluate_scenario  # noqa: E402  # 复用统一指标
from scenarios import SCENARIOS  # noqa: E402
from jevp.backend import auto_backend  # noqa: E402
from jevp.engine import DecisionEngine  # noqa: E402


def main() -> None:
    with_agentjev = "--with-agentjev" in sys.argv[1:]

    backends = ["router", "openai", "laya"]
    if with_agentjev:
        backends.append("agentjev")

    groups = {"中文组": "enterprise_match", "英文组": "enterprise_match_en"}
    results = {}

    print(f"{'后端':<10}{'中文命中':>10}{'英文命中':>10}{'总体':>10}  说明")
    print("-" * 52)
    for bk in backends:
        backend = auto_backend(backend=bk)
        engine = DecisionEngine(backend=backend,
                                tokenizer=getattr(backend, "tokenizer", None))
        note = ""
        hits = {}
        for label, sc in groups.items():
            scenario = SCENARIOS[sc]
            r = evaluate_scenario(engine, scenario, repeat=1, warmup=0)
            acc = r.get("accuracy")
            hits[label] = acc
            if bk == "router":
                note = f"routed={backend.last_meta.get('lang','?')}"
        n_zh = hits["中文组"]
        n_en = hits["英文组"]
        overall = round((n_zh + n_en) / 2, 4) if n_zh is not None and n_en is not None else None
        results[bk] = {"zh": n_zh, "en": n_en, "overall": overall}
        print(f"{bk:<10}{(str(n_zh) if n_zh is not None else 'n/a'):>8}{'':>2}"
              f"{(str(n_en) if n_en is not None else 'n/a'):>8}{'':>2}"
              f"{(str(overall) if overall is not None else 'n/a'):>8}  {note}")

    print("\n结论参考：")
    print("- router 使用语言感知路由（中文→27B、英文→判别式），目标是兼顾两组的命中率；")
    print("- 中英双语数据集（国际企业加入后）即未来中英兼顾模型的持续评测基准。")


if __name__ == "__main__":
    main()