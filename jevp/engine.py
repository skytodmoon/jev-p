"""约束解码引擎：把"判断"从"文本生成"中剥离。

对齐 Jev 的语义：
- 输入 state + 一组类型化 questions；
- 对每个问题构造 prompt，**一批**（batch）全部问题一起前向；
- 只读取每个样本"末尾位置"的 logits，对候选 token 打分（softmax）；
- **JSON 由程序拼装**，模型不生成任何自由文本 → 结构上不可能格式错误。

只依赖标准库；tokenizer 以鸭子类型注入（便于离线单测）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .protocol import Answer, Question, build_prompt


# --------------------------------------------------------------------------- #
# 打分原语
# --------------------------------------------------------------------------- #
def softmax(vals: Sequence[float]) -> List[float]:
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    return [e / z for e in exps]


def candidate_token_ids(tokenizer: Any, texts: Sequence[str]) -> List[List[int]]:
    """把候选文本转成 token id 序列（首 token 为主）。

    tokenizer 鸭子类型：``tokenizer(text, add_special_tokens=False)["input_ids"]``。
    """
    out: List[List[int]] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"候选文本 {text!r} 没有产生任何 token id")
        out.append([int(t) for t in ids])
    return out


def _resolve_unique_tokens(candidates_ids: Sequence[Sequence[int]]) -> List[int]:
    """为每个候选选一个唯一 token id：优先首 token，冲突则顺延下一个。

    候选全部冲突（两个 label 的 token 序列完全相同）时抛出错误——
    此时打分结果无法区分它们，属于协议设计错误。
    """
    chosen: List[int] = []
    used: set = set()
    for ids in candidates_ids:
        pick: int | None = None
        for t in ids:
            if t not in used:
                pick = int(t)
                break
        if pick is None:
            raise ValueError(
                "候选 token 序列完全重合，无法区分。请更换候选标签文本或打分 token 映射。"
            )
        used.add(pick)
        chosen.append(pick)
    return chosen


def score_from_logits(
    logits_row: Sequence[float],
    candidates_ids: Sequence[Sequence[int]],
) -> Tuple[List[int], List[float], float]:
    """对单个样本的末位 logits 给候选打分。

    返回 (chosen_token_ids, probs, raw_mass)：
    - chosen_token_ids：每个候选实际使用的 token id；
    - probs：候选间的 softmax 概率（和为 1）；
    - raw_mass：候选 token 在**完整词表** softmax 中分到的概率之和。
      raw_mass 低 = 模型对这个候选集"没底"，即使 confidence 很高也要警惕
      （实证：vllm-jev-decison 里 confidence 0.9985 而 mass 只有 0.000004）。
    """
    chosen = _resolve_unique_tokens(candidates_ids)
    logits = [float(logits_row[t]) for t in chosen]
    probs = softmax(logits)
    # 完整词表 softmax 一次（V 通常 3e4~1.5e5，B 小，可接受）
    full = softmax(list(logits_row))
    mass = sum(full[t] for t in chosen)
    return chosen, probs, mass


# --------------------------------------------------------------------------- #
# 字符串级打分（Ollama 后端：只暴露 top-k logprob，无完整词表）
# --------------------------------------------------------------------------- #
_NEG_LOGPROB = -100.0  # 候选未进 top-k 时的近似 logprob


def _string_variants(text: str) -> List[str]:
    """候选 token 字符串的常见拼写变体（BPE 空格前缀/大小写），保序去重。"""
    t = text.strip()
    out = [t, " " + t, t.lower(), " " + t.lower(), t.upper(), " " + t.upper()]
    seen: set = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def _score_string_logprobs(
    logprobs_row: Dict[str, float],
    candidate_variants: Sequence[Sequence[str]],
) -> Tuple[List[Optional[str]], List[float], float]:
    """对 top-k logprob 字典给候选打分（Ollama 后端路径）。

    返回 (chosen_token_strings, probs, raw_mass)：
    - 每个候选取变体中的最大 logprob（未命中 -> _NEG_LOGPROB）；
    - probs 为候选间 softmax；
    - raw_mass 为**top-k 近似**：候选在本次返回的 top-k 概率分布中的占比
      （非完整词表份额；README 已注明该差异）。
    """
    chosen_lp: List[float] = []
    chosen_name: List[Optional[str]] = []
    for variants in candidate_variants:
        lp, name = _NEG_LOGPROB, None
        for v in variants:
            if v in logprobs_row and logprobs_row[v] > lp:
                lp = logprobs_row[v]
                name = v
        chosen_lp.append(lp)
        chosen_name.append(name)

    m = max(chosen_lp)
    exps = [math.exp(x - m) for x in chosen_lp]
    z = sum(exps)
    probs = [e / z for e in exps]

    # raw_mass（top-k 近似）：候选占返回 top-k 概率的比例
    values = list(logprobs_row.values())
    mass = 0.0
    if values:
        am = max(values)
        total = sum(math.exp(v - am) for v in values)
        cand_total = sum(math.exp(x - am) for x in chosen_lp if x > _NEG_LOGPROB / 2)
        mass = cand_total / total if total > 0 else 0.0
    return chosen_name, probs, mass


# --------------------------------------------------------------------------- #
# 决策引擎
# --------------------------------------------------------------------------- #
class DecisionEngine:
    """底层对象只需提供 ``batch_last_logits(prompts) -> 每样本末位 logits``。"""

    def __init__(self, backend: Any, tokenizer: Any):
        self.backend = backend
        # 判别式后端（Laya 等）无 tokenizer；非 direct 路径才使用
        self.tokenizer = getattr(backend, "tokenizer", None) if tokenizer is None else tokenizer

    # -- public ------------------------------------------------------------ #
    def decide(self, state: Any, questions: Dict[str, Question]) -> Dict[str, Answer]:
        """并行评估所有问题，返回 {问题名: Answer}。

        路由后端（RouterBackend 等，实现了 ``route_decide``）优先接管；
        判别式后端（Laya 等，``backend.direct=True``）非词表打分，走 direct_decide；
        否则走"构造 prompt -> 一次 batch 前向 -> 候选打分"。
        """
        route = getattr(self.backend, "route_decide", None)
        if route is not None:
            return route(state, questions)

        if getattr(self.backend, "direct", False):
            return self.backend.direct_decide(state, questions)

        names = list(questions)
        prompts = [build_prompt(state, n, questions[n]) for n in names]
        rows = self.backend.batch_last_logits(prompts)  # (B, V)，B == len(questions)
        if len(rows) != len(names):
            raise RuntimeError(
                f"backend 返回 {len(rows)} 行，期望 {len(names)}（每问题一行）"
            )
        answers: Dict[str, Answer] = {}
        for i, name in enumerate(names):
            # 行数据形态由后端契约决定：token_id 路径是序列，
            # token_string(Ollama) 路径是 {token: logprob} 字典——原样传递
            answers[name] = self._answer_one(questions[name], rows[i])
        return answers

    def count_input_tokens(self, prompts: Sequence[str]) -> int:
        total = 0
        for p in prompts:
            total += len(self.tokenizer(p, add_special_tokens=False)["input_ids"])
        return total

    def decide_many(
        self, states: Sequence[Any], questions: Dict[str, Question], *,
        batch_size: int = 32, max_batch_chars: int = 64000,
    ) -> List[Any]:
        """Bounded cross-state batches, in input order; failures stay with their state.

        batch_size bounds *prompts*, not companies. The character budget bounds
        materialized input, not tokenizer tokens or GPU memory. A single oversized
        prompt is rejected instead of silently losing its question to truncation.
        Direct backends retain their existing serial semantics unless they provide
        a native route_decide_many hook.
        """
        if not 1 <= batch_size <= 512 or max_batch_chars < 1:
            raise ValueError("batch_size must be 1..512; max_batch_chars must be positive")
        if not questions:
            raise ValueError("questions cannot be empty")
        route = getattr(self.backend, "route_decide_many", None)
        if route is not None:
            return route(states, questions, batch_size=batch_size,
                         max_batch_chars=max_batch_chars)
        if getattr(self.backend, "direct", False) or hasattr(self.backend, "route_decide"):
            results = []
            for state in states:
                try:
                    results.append(self.decide(state, questions))
                except Exception as exc:
                    results.append(exc)
            return results

        results: List[Any] = [{} for _ in states]
        pending: List[Tuple[int, str, str]] = []
        chars = 0

        def flush() -> None:
            if not pending:
                return
            prompts = [p for _, _, p in pending]
            try:
                fn = getattr(self.backend, "batch_last_logits_safe",
                             self.backend.batch_last_logits)
                rows = fn(prompts)
                if len(rows) != len(pending):
                    raise RuntimeError(f"backend returned {len(rows)} rows, expected {len(pending)}")
            except Exception as exc:
                rows = [exc] * len(pending)
            for (idx, name, _), row in zip(pending, rows):
                if isinstance(results[idx], Exception):
                    continue
                try:
                    if isinstance(row, Exception):
                        raise row
                    results[idx][name] = self._answer_one(questions[name], row)
                except Exception as exc:
                    results[idx] = exc
            pending.clear()

        for idx, state in enumerate(states):
            for name, question in questions.items():
                if isinstance(results[idx], Exception):
                    break
                try:
                    prompt = build_prompt(state, name, question)
                    if len(prompt) > max_batch_chars:
                        raise ValueError("single prompt exceeds max_batch_chars; shorten the profile")
                    context_limit = getattr(self.backend, "max_length", None)
                    if context_limit and self.tokenizer is not None and getattr(
                        self.backend, "candidate_scoring", "token_id"
                    ) == "token_id":
                        count = len(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])
                        if count > context_limit:
                            raise ValueError("prompt exceeds backend token context; shorten the profile")
                except Exception as exc:
                    results[idx] = exc
                    break
                if pending and (len(pending) >= batch_size or chars + len(prompt) > max_batch_chars):
                    flush()
                    chars = 0
                pending.append((idx, name, prompt))
                chars += len(prompt)
        flush()
        return results

    # -- private ----------------------------------------------------------- #
    def _score(self, texts: Sequence[str], logits_row: Any):
        if getattr(self.backend, "candidate_scoring", "token_id") == "token_string":
            # Ollama 后端：行数据是 {token字符串: logprob} 稀疏字典
            return _score_string_logprobs(
                logits_row, [_string_variants(t) for t in texts]
            )
        ids = candidate_token_ids(self.tokenizer, texts)
        chosen, probs, mass = score_from_logits(logits_row, ids)
        return chosen, probs, mass

    def _answer_one(self, q: Question, row: Sequence[float]) -> Answer:
        if q.type == "noul":
            # 打分 token 用 Yes/No（比 true/false 更常见）；criteria 描述只用于提示
            chosen, probs, mass = self._score(["Yes", "No"], row)
            p_true = probs[0]
            return Answer("noul", {"noul": round(p_true, 4), "raw_mass": round(mass, 6)})

        if q.type == "choice":
            labels = list(q.criteria.keys())  # type: ignore[union-attr]
            chosen, probs, mass = self._score(labels, row)
            best = max(range(len(probs)), key=lambda i: probs[i])
            return Answer("choice", {
                "choice": labels[best],
                "confidence": round(probs[best], 4),
                "probabilities": {lbl: round(p, 4) for lbl, p in zip(labels, probs)},
                "raw_mass": round(mass, 6),
            })

        # score：候选即档位下标 "0..n-1"，加权求期望分（可落在档位之间）
        n = len(q.criteria)  # type: ignore[arg-type]
        chosen, probs, mass = self._score([str(i) for i in range(n)], row)
        score = sum(i * p for i, p in enumerate(probs))
        conf = max(probs)
        return Answer("score", {
            "score": round(score, 4),
            "confidence": round(conf, 4),
            "legend": {str(i): d for i, d in enumerate(q.criteria)},  # type: ignore[union-attr]
            "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
            "raw_mass": round(mass, 6),
        })
