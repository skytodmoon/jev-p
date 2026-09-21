"""Choice / Score / Noul 问题原语：数据模型、校验、prompt 构建与答案序列化。

对齐 TypeSafe Jev 的公开接口形状（见 Cloudflare 官方文档），
仅依赖标准库，可离线测试。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

QType = Literal["noul", "choice", "score"]

SYSTEM_PROMPT = (
    "You are JevP, a System One decision engine. Decide based strictly on the "
    "STATE. Do not explain. Only produce the single-token answer at the position."
)


# --------------------------------------------------------------------------- #
# 问题模型
# --------------------------------------------------------------------------- #
@dataclass
class Question:
    """一个问题：instructions + 类型相关 criteria。"""

    instructions: str
    criteria: Union[Dict[str, str], List[str], None] = None
    type: QType = "noul"  # 默认值仅为满足 dataclass 字段顺序；子类/工厂会强制正确类型

    def validate(self) -> None:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError(f"question instructions 不能为空: {self!r}")
        if self.type == "noul":
            if self.criteria is None:
                return  # 可省略；默认 true/false
            if not isinstance(self.criteria, dict):
                raise ValueError("noul 的 criteria 必须是 {true: 描述, false: 描述}")
            keys = set(self.criteria)
            if not keys.issubset({"true", "false"}):
                raise ValueError(f"noul 的 criteria 只能含 true/false，收到: {keys}")
        elif self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError("choice 的 criteria 必须是至少有 2 个选项的 dict {label: 描述}")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError("score 的 criteria 必须是至少有 2 个档位的 list [描述...]")
        else:
            raise ValueError(f"未知问题类型: {self.type!r}")

    def candidate_labels(self) -> List[Any]:
        """该问题在打分阶段的候选"键"序列。"""
        if self.type == "noul":
            return ["true", "false"]
        if self.type == "choice":
            return list(self.criteria.keys())  # type: ignore[union-attr]
        # score：档位索引的字符串形式（0..n-1）
        return [str(i) for i in range(len(self.criteria))]  # type: ignore[arg-type]

    def candidate_texts(self) -> List[str]:
        """候选在 prompt 中展示的文本（用于模型理解），打分 token 另见 engine。"""
        if self.type == "noul":
            c = self.criteria or {}
            return [
                c.get("true", "Yes") or "Yes",
                c.get("false", "No") or "No",
            ]
        if self.type == "choice":
            c = self.criteria or {}
            return [f"{k}: {v}" for k, v in c.items()]
        # score
        return [f"{i}: {d}" for i, d in enumerate(self.criteria or [])]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d


@dataclass
class NoulQuestion(Question):
    """Noul：是/否判断。构造后自动校验。"""

    def __post_init__(self) -> None:
        self.type = "noul"
        self.validate()


@dataclass
class ChoiceQuestion(Question):
    """Choice：多选一分类。构造后自动校验。"""

    def __post_init__(self) -> None:
        self.type = "choice"
        self.validate()


@dataclass
class ScoreQuestion(Question):
    """Score：有序量表打分。构造后自动校验。"""

    def __post_init__(self) -> None:
        self.type = "score"
        self.validate()


def parse_question(raw: Dict[str, Any]) -> Question:
    """从请求 JSON 解析一个问题。"""
    if not isinstance(raw, dict):
        raise ValueError("question 必须是 JSON 对象")
    qtype = raw.get("type")
    if qtype not in ("noul", "choice", "score"):
        raise ValueError(f"question.type 必须是 noul/choice/score，收到: {qtype!r}")
    criteria = raw.get("criteria")
    q = Question(
        type=qtype,
        instructions=raw.get("instructions", ""),
        criteria=criteria,
    )
    q.validate()
    return q


def validate_questions(questions: Dict[str, Any]) -> Dict[str, Question]:
    """解析并校验一组 questions（键名仅用于回传对齐，不参与推理）。"""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions 必须是非空对象")
    parsed: Dict[str, Question] = {}
    for name, raw in questions.items():
        parsed[str(name)] = parse_question(raw)
    return parsed


# --------------------------------------------------------------------------- #
# prompt 构建
# --------------------------------------------------------------------------- #
def _stringify_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(state)


def build_prompt(state: Any, name: str, question: Question) -> str:
    """为单个问题构建打分 prompt；候选展示帮助模型理解，但打分只用候选 token。"""
    lines = [SYSTEM_PROMPT, "", "STATE:", _stringify_state(state), ""]
    if question.type == "choice" and question.criteria:
        lines.append("OPTIONS:")
        for label, desc in question.criteria.items():
            lines.append(f"- {label}: {desc}")
        lines.append("")
    elif question.type == "score" and question.criteria:
        lines.append("SCALE:")
        for i, desc in enumerate(question.criteria):
            lines.append(f"- {i}: {desc}")
        lines.append("")
    lines.append(f"QUESTION ({name}) · {question.type}: {question.instructions}")
    lines.append("ANSWER (single token):")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 答案模型（与 Jev 输出形状对齐）
# --------------------------------------------------------------------------- #
@dataclass
class Answer:
    type: QType
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = {"type": self.type}
        out.update(self.payload)
        return out