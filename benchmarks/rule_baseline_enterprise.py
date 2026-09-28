"""规则基线：企业-产品匹配 L0 粗筛（零 LLM 调用）参考实现。

用于对比：纯规则（0 次 LLM 调用）能达到与 System One 打分多接近的匹配档位，
支撑"两级漏斗"降频设计（L0 规则粗筛放行候选 → L1/L2 LLM 精排 Top-N）。

打分规则（透明、可辩解）：
  +1 主营行业是重资产制造业（制造/钢铁/化工/能源/电池/建材/汽车/机械等）
  +1 企业规模大型（中型 +0.5）
  +1 招投标/采购中出现需求信号（监测/维护/健康/智能/数字化/预测/联网/系统）
  +0.5 近期（2026）有采购动作
  +0.5 官网战略含数字化/智能制造/全生命周期等关键词
  上限 3.0 → 与金标准 match_level(0-3) 对齐。

用法：
    python benchmarks/rule_baseline_enterprise.py
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

from scenarios.enterprise_match import frames, expected  # noqa: E402

INDUSTRY_KW = ("制造", "钢铁", "化工", "能源", "电池", "建材", "汽车", "机械", "材料", "冶炼",
               "steel", "chemical", "energy", "battery", "auto", "manufactur",
               "machine", "material", "mining", "utility", "power", "food processing")
BID_KW = ("监测", "维护", "健康", "智能", "数字化", "预测", "联网", "系统",
          "monitor", "predict", "maintenance", "digital", "smart", "health",
          "condition", "analytics", "retrofit", "automation", "platform", "tooling")
STRAT_KW = ("智能", "数字化", "全生命周期", "灯塔", "智造", "本质安全",
            "digital", "smart", "lifecycle", "industry 4.0", "reliability",
            "asset", "analytics", "O&M")


def rule_score(f) -> float:
    s = 0.0
    ind = f["basic"]["industry"]
    if any(k in ind for k in INDUSTRY_KW):
        s += 1.0
    scale = f["basic"]["scale"]
    if "大型" in scale:
        s += 1.0
    elif "中型" in scale:
        s += 0.5
    bids = " ".join(b["title"] for b in f["bidding"])
    if any(k in bids for k in BID_KW):
        s += 1.0
    if any(b.get("year") == 2026 for b in f["bidding"]):
        s += 0.5
    strat = f["website"].get("strategy", "")
    if any(k in strat for k in STRAT_KW):
        s += 0.5
    return min(3.0, s)


def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else None


def domain_patch(llm_score, r) -> float:
    """域先验硬校正（可辩解业务规则，非调参过拟合；小样本注意留出验证）：
    - 行业不匹配(industry_fit 为 False) → 上限 1.0（低档封顶）；
    - 行业✗ 且 无任何招投标 → 上限 0.5（无关企业）；
    - 非企业实体（高校/科研院所） → 上限 0.5（非商业客户）。
    r 需含: industry_fit(bool), n_bids(int), non_commercial(bool)。
    """
    s = float(llm_score)
    if not r["industry_fit"]:
        s = min(s, 1.0)
        if r.get("n_bids", 1) == 0:
            s = min(s, 0.5)
    if r.get("non_commercial"):
        s = min(s, 0.5)
    return s


def is_non_commercial(frame) -> bool:
    scale = str(frame["basic"].get("scale", ""))
    industry = str(frame["basic"].get("industry", ""))
    return ("非企业" in scale or "高校" in scale
            or "research" in industry.lower() or "institute" in industry.lower())


def main() -> None:
    rows = []
    for i, f in enumerate(frames()):
        s = rule_score(f)
        e = expected(f)["match_score"]
        rows.append((f["name"], s, e, abs(s - e)))

    print(f"{'企业':<18}{'规则分':>8}{'金标准':>8}{'偏差':>6}")
    print("-" * 44)
    for name, s, e, d in rows:
        print(f"{name:<18}{s:>8.2f}{e:>8}{d:>6.2f}")

    acc = sum(1 for _, s, e, _ in rows if round(s) == e) / len(rows)
    corr = pearson([r[1] for r in rows], [r[2] for r in rows])
    print(f"\n档位命中率（round==金标准）: {acc:.2%}")
    print(f"与金标准排序相关 Pearson: {corr:.3f}" if corr else "排序相关: n/a")
    print("说明：L0 规则粗筛 0 次 LLM 调用；命中率可作两级漏斗的上限参照。")


if __name__ == "__main__":
    main()