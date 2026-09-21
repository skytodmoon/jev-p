"""Ollama 后端（字符串级 top-k logprob 打分）的离线单元测试。

不发起真实 HTTP：只测 engine 的字符串打分路径与变体展开逻辑，
用 fake 后端模拟 "候选打分 = token_string" 契约。
"""
import unittest

from jevp.engine import (
    DecisionEngine,
    _score_string_logprobs,
    _string_variants,
)
from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion


class FakeOllamaBackend:
    """模拟 OllamaBackend /v1/chat/completions 的返回：{token字符串: logprob}。"""

    candidate_scoring = "token_string"
    model_id = "ollama:qwen2.5:7b (fake)"

    def __init__(self, row: dict):
        self.row = row

    def batch_last_logits(self, prompts):
        return [dict(self.row) for _ in prompts]


class TestStringVariants(unittest.TestCase):
    def test_basic(self):
        v = _string_variants("Yes")
        self.assertEqual(v[0], "Yes")
        self.assertIn(" Yes", v)
        self.assertIn("yes", v)
        self.assertIn("YES", v)
        # 去重
        self.assertEqual(len(v), len(set(v)))

    def test_strip(self):
        v = _string_variants(" billing ")
        self.assertEqual(v[0], "billing")


class TestScoreStringLogprobs(unittest.TestCase):
    def test_hits_use_max_variant(self):
        row = {"billing": -0.2, "technical": -2.0, "sales": -1.0, "other": -0.5}
        variants = [_string_variants(t) for t in ("billing", "technical", "sales", "other")]
        names, probs, mass = _score_string_logprobs(row, variants)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[0], probs[3])  # billing(-0.2) > other(-0.5)

    def test_space_prefix_variant_rescues(self):
        # 返回里只有带空格前缀的 token
        row = {" Yes": -0.1, " No": -3.0}
        names, probs, mass = _score_string_logprobs(
            row, [[_string_variants("Yes")[0]], [_string_variants("No")[0]]]
        )
        # _string_variants 数组用整组（含 " Yes"）
        variants = [_string_variants("Yes"), _string_variants("No")]
        names2, probs2, mass2 = _score_string_logprobs(row, variants)
        self.assertGreater(probs2[0], 0.9)
        self.assertGreater(mass2, 0.0)

    def test_miss_becomes_low_confidence_and_zero_mass(self):
        # 候选都不在 top-k 中
        row = {" other": -0.5}
        variants = [_string_variants("billing"), _string_variants("technical")]
        names, probs, mass = _score_string_logprobs(row, variants)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertAlmostEqual(mass, 0.0, places=9)
        self.assertAlmostEqual(probs[0], 0.5, places=6)  # 均匀 → 低置信（会被路由拦截）

    def test_raw_mass_is_topk_share(self):
        row = {" Yes": -0.1, " No": -2.0, " maybe": -1.0}
        variants = [_string_variants("Yes"), _string_variants("No")]
        _, probs, mass = _score_string_logprobs(row, variants)
        total = sum(__import__("math").exp(v + 2.0) for v in row.values())
        cand = sum(__import__("math").exp(x + 2.0) for x in (-0.1, -2.0))
        self.assertAlmostEqual(mass, cand / total, places=4)


class TestEngineStringPath(unittest.TestCase):
    def test_full_workflow_via_string_scoring(self):
        # 一帧马里奥式决策：action(choice) + gap(Noul) + danger(Score)
        row = {
            " run": -0.3, " jump": -0.6, " crouch": -3.0,
            " wait": -2.0, " turn_back": -4.0,
            " Yes": -0.5, " No": -1.2,
            " 0": -1.5, " 1": -0.4, " 2": -1.8,
        }
        engine = DecisionEngine(backend=FakeOllamaBackend(row), tokenizer=None)
        questions = {
            "action": ChoiceQuestion(
                instructions="下一步宏观动作？",
                criteria={"run": "跑", "jump": "跳", "crouch": "蹲",
                          "wait": "等", "turn_back": "回头"},
            ),
            "gap": NoulQuestion(instructions="前方有坑吗？"),
            "danger": ScoreQuestion(instructions="危险程度？", criteria=["低", "中", "高"]),
        }
        answers = engine.decide("Maro 状态快照", questions)

        act = answers["action"].to_dict()
        self.assertEqual(act["choice"], "run")  # -0.3 最大（jump 次优，conf 略低于 0.5）
        self.assertGreater(act["confidence"], 0.45)

        gap = answers["gap"].to_dict()
        self.assertGreater(gap["noul"], 0.5)  # Yes(-0.5) > No(-1.2)

        dg = answers["danger"].to_dict()
        self.assertGreater(dg["score"], 0.5)  # 档位 1 胜出 → 分数在 0-2 之间偏 1

    def test_batch_shape_matches_questions(self):
        backend = FakeOllamaBackend({" Yes": -0.1, " No": -2.0})
        engine = DecisionEngine(backend=backend, tokenizer=None)
        qs = {"a": NoulQuestion(instructions="紧急吗？"), "b": NoulQuestion(instructions="要退款吗？")}
        engine.decide("s", qs)
        # FakeOllamaBackend 为每个 prompt 返回一行 → 行数 == 问题数（契约由 decide 校验）


if __name__ == "__main__":
    unittest.main()