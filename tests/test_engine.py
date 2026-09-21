"""约束解码引擎单元测试（mock tokenizer + mock backend，零依赖）。

核心验证点：
- Noul/Choice/Score 的数值打分正确；
- 程序化拼装的答案形状与 Jev 对齐（type/score/choice/noul/confidence/probabilities/raw_mass/legend）；
- raw_mass（完整词表 softmax 中候选概率之和）计算正确；
- 候选 token 冲突时顺延、全冲突时报错；
- 一次 decide = 一次 batch 前向（每问题一行）。
"""
import unittest

from jevp.engine import (
    DecisionEngine,
    _resolve_unique_tokens,
    candidate_token_ids,
    score_from_logits,
    softmax,
)
from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion

# 小型确定性 tokenizer：文本 -> 固定 token id 序列
ID_MAP = {
    "Yes": [1],
    "No": [2],
    "billing": [3],
    "technical": [4],
    "sales": [5],
    "0": [6],
    "1": [7],
    "2": [8],
}


class MockTokenizer:
    def __call__(self, text, add_special_tokens=False):
        key = text.strip()
        if key in ID_MAP:
            return {"input_ids": list(ID_MAP[key])}
        # 未注册文本：退化为基于首字符的伪 token（仅保证确定性）
        return {"input_ids": [200 + ord(key[0]) % 50]}


class MockBackend:
    def __init__(self, rows):
        self.rows = rows  # List[List[float]]
        self.model_id = "mock"

    def batch_last_logits(self, prompts):
        assert len(prompts) == len(self.rows), f"prompts={len(prompts)} rows={len(self.rows)}"
        return [list(r) for r in self.rows]


def make_logits_row(token_id, score, other=-5.0, vocab=10):
    row = [other] * vocab
    row[token_id] = score
    return row


class TestScoringPrimitives(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([1.0, 0.0, -1.0])), 1.0, places=6)
        # 单调
        p = softmax([2.0, 1.0, 0.0])
        self.assertGreater(p[0], p[1])
        self.assertGreater(p[1], p[2])

    def test_candidate_token_ids(self):
        ids = candidate_token_ids(MockTokenizer(), ["Yes", "No"])
        self.assertEqual(ids, [[1], [2]])

    def test_resolve_conflicts_uses_next_token(self):
        chosen = _resolve_unique_tokens([[1], [1, 2]])
        self.assertEqual(chosen, [1, 2])

    def test_resolve_total_conflict_raises(self):
        with self.assertRaises(ValueError):
            _resolve_unique_tokens([[1], [1]])

    def test_raw_mass_is_vocab_share(self):
        # vocab=4: token1=2.0, token2=1.5, 其余很低
        row = [0.0, 2.0, 1.5, 0.0]
        chosen, probs, mass = score_from_logits(row, [[1], [2]])
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        # 完整词表 softmax 下 token1+token2 的概率份额
        full = softmax(row)
        self.assertAlmostEqual(mass, full[1] + full[2], places=6)
        self.assertAlmostEqual(probs[0], full[1] / (full[1] + full[2]), places=6)


class TestDecisionEngine(unittest.TestCase):
    def test_choice_noul_score_full_workflow(self):
        # 三个问题，一次 batch（3 行 logits）
        rows = [
            make_logits_row(3, 3.0),  # billing 胜出
            make_logits_row(1, 2.0),  # Yes 胜出
            make_logits_row(7, 1.0),  # 档位 1 胜出（0:0, 1:1.0, 2:0.5 构造见下）
        ]
        rows[2][6] = 0.0
        rows[2][8] = 0.5
        backend = MockBackend(rows)
        engine = DecisionEngine(backend=backend, tokenizer=MockTokenizer())

        questions = {
            "department": ChoiceQuestion(instructions="哪个部门？",
                                         criteria={"billing": "计费", "technical": "技术", "sales": "销售"}),
            "is_urgent": NoulQuestion(instructions="紧急吗？"),
            "frustration": ScoreQuestion(instructions="挫败程度？", criteria=["平静", "不满", "非常愤怒"]),
        }
        answers = engine.decide("工单: 支付失败三天", questions)

        # Choice
        dep = answers["department"].to_dict()
        self.assertEqual(dep["type"], "choice")
        self.assertEqual(dep["choice"], "billing")
        cand_full = softmax([rows[0][3], rows[0][4], rows[0][5]])
        self.assertAlmostEqual(dep["confidence"], cand_full[0], places=4)
        self.assertAlmostEqual(sum(dep["probabilities"].values()), 1.0, places=3)
        self.assertIn("raw_mass", dep)

        # Noul
        urg = answers["is_urgent"].to_dict()
        self.assertEqual(urg["type"], "noul")
        self.assertGreater(urg["noul"], 0.5)  # Yes 胜出
        full = softmax(rows[1])
        self.assertAlmostEqual(urg["noul"], full[1] / (full[1] + full[2]), places=4)

        # Score：加权期望分、可小数、legend
        fr = answers["frustration"].to_dict()
        self.assertEqual(fr["type"], "score")
        p = softmax(rows[2][6:9])
        expected = 0 * p[0] + 1 * p[1] + 2 * p[2]
        self.assertAlmostEqual(fr["score"], expected, places=4)
        self.assertNotEqual(fr["score"], int(fr["score"]))  # 可落在档位之间
        self.assertEqual(fr["legend"], {"0": "平静", "1": "不满", "2": "非常愤怒"})

    def test_one_forward_per_decide(self):
        calls = {}

        class TrackingBackend:
            def __init__(self):
                self.model_id = "track"
                self.rows = [[0.0] * 10, [0.0] * 10, [0.0] * 10]

            def batch_last_logits(self, prompts):
                calls["n"] = len(prompts)
                return self.rows

        engine = DecisionEngine(backend=TrackingBackend(), tokenizer=MockTokenizer())
        engine.decide("s", {
            "a": NoulQuestion(instructions="x"),
            "b": ChoiceQuestion(instructions="x", criteria={"billing": "计费", "technical": "技术"}),
            "c": ScoreQuestion(instructions="x", criteria=["低", "高"]),
        })
        self.assertEqual(calls["n"], 3)  # 一次 batch 前向，每问题一行

    def test_noul_probability_equals_true_share(self):
        row = make_logits_row(1, 1.0)  # Yes 1.0, No 0.0(other=-5 已设定? make 中 other=-5)
        engine = DecisionEngine(backend=MockBackend([row]), tokenizer=MockTokenizer())
        ans = engine.decide("s", {"u": NoulQuestion(instructions="x")})["u"].to_dict()
        full = softmax(row)
        expected = full[1] / (full[1] + full[2])
        self.assertAlmostEqual(ans["noul"], expected, places=4)


if __name__ == "__main__":
    unittest.main()