"""JevP: Jev-like System One decision engine on stock small LLMs.

核心思路（对应 README §2 架构）：
state + questions -> build_prompt -> 单次前向取末位 logits -> 候选 token softmax
-> 程序化拼装类型化答案（不生成自由文本）。
"""

__version__ = "0.1.0"

from .protocol import (
    Answer,
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    build_prompt,
    parse_question,
    validate_questions,
)
from .engine import DecisionEngine, score_from_logits, softmax

__all__ = [
    "Answer",
    "ChoiceQuestion",
    "DecisionEngine",
    "NoulQuestion",
    "Question",
    "ScoreQuestion",
    "build_prompt",
    "parse_question",
    "score_from_logits",
    "softmax",
    "validate_questions",
]