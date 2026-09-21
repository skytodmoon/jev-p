"""校准工具：ECE、温度缩放、置信度路由（仅标准库，可离线测试）。

背景（务必阅读 README §5）：raw softmax 不是校准概率。本模块提供
- ECE（期望校准误差）量化校准质量；
- TemperatureScaler 学习一个温度系数，把概率拉回"0.9 就约 9 成对"；
- confidence_router 做选择性分类：高置信自动、低置信转人工。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# ECE：期望校准误差
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    confidences: Sequence[float],
    corrects: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """加权平均的 |准确率 - 平均置信度|。

    完美校准 -> 接近 0；过度自信 -> 明显 >0。
    """
    if len(confidences) != len(corrects) or len(confidences) == 0:
        raise ValueError("confidences 与 corrects 必须等长且非空")
    pairs = sorted(zip(confidences, corrects), key=lambda p: p[0])
    total = len(pairs)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # 最后一箱闭合右边界，确保包含 c == 1.0；其余为左闭右开
        if b == n_bins - 1:
            bin_pairs = [p for p in pairs if lo <= p[0] <= hi]
        else:
            bin_pairs = [p for p in pairs if lo <= p[0] < hi]
        if not bin_pairs:
            continue
        conf = sum(c for c, _ in bin_pairs) / len(bin_pairs)
        acc = sum(1 for _, ok in bin_pairs if ok) / len(bin_pairs)
        ece += (len(bin_pairs) / total) * abs(acc - conf)
    return ece


# --------------------------------------------------------------------------- #
# 温度缩放（Temperature Scaling）
# --------------------------------------------------------------------------- #
def _softmax_nll_tempered(
    probs: List[List[float]],
    labels: Sequence[int],
    temperature: float,
) -> float:
    """温度缩放后的平均负对数似然（多分类）。

    probs[i][k] 是样本 i 的候选 k 的原始概率；labels[i] 是正确下标。
    T=1 时即原始 NLL。最小化 NLL 可学习最优温度。
    """
    assert 0 < temperature
    nll = 0.0
    n = len(probs)
    for i, row in enumerate(probs):
        # 温度缩放：logits/T 后再 softmax（等价于 p^(1/T) 归一化）
        scaled = [p ** (1.0 / temperature) for p in row]
        z = sum(scaled)
        nll += -math.log(scaled[labels[i]] / z)
    return nll / max(n, 1)


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    @classmethod
    def fit(
        cls,
        probs: Sequence[Sequence[float]],
        labels: Sequence[int],
        grid: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                  1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0),
    ) -> "TemperatureScaler":
        """在验证集上网格搜索使 NLL 最小的温度。T>1 通常抑制过度自信。"""
        best_t, best_nll = 1.0, float("inf")
        for t in grid:
            nll = _softmax_nll_tempered([list(p) for p in probs], labels, t)
            if nll < best_nll:
                best_nll, best_t = nll, t
        return cls(temperature=best_t)

    def scale(self, probs: Sequence[Sequence[float]]) -> List[List[float]]:
        """应用温度缩放，返回每行归一化后的概率。"""
        out: List[List[float]] = []
        for row in probs:
            scaled = [p ** (1.0 / self.temperature) for p in row]
            z = sum(scaled)
            out.append([v / z for v in scaled])
        return out


# --------------------------------------------------------------------------- #
# 置信度路由（选择性分类）
# --------------------------------------------------------------------------- #
@dataclass
class RouteDecision:
    action: str  # "auto" | "review" | "human"
    confidence: float

    def to_dict(self) -> dict:
        return {"action": self.action, "confidence": self.confidence}


def confidence_router(confidence: float, high: float = 0.9, low: float = 0.7) -> str:
    """按置信度路由：>=high 自动执行；>=low 升级复核；<low 转人工。"""
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence 必须在 [0,1]，收到 {confidence}")
    if confidence >= high:
        return "auto"
    if confidence >= low:
        return "review"
    return "human"