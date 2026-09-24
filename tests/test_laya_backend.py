"""LayaBackend（判别式后端）单元测试：不加载真实模型、不联网。

用 FakeRouter 模拟 laya.Router.predict 的返回结构，验证：
- engine.decide 正确 dispatch 到 direct_decide；
- Laya 答案正确映射为 JevP Answer（choice/noul/score 形状与字段）；
- 缺问题键时报错。
"""
import unittest

from jevp.backend import LayaBackend, _laya_answer_to_jevp
from jevp.engine import DecisionEngine
from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion


class FakeRouter:
    """模拟 laya.Router：predict 返回 Laya 结构的 answers。"""

    def __init__(self, answers=None):
        self.answers = answers
        self.calls = []

    def predict(self, state, qdict, model=None):
        self.calls.append(model)
        if self.answers is not None:
            # 显式注入：只返回 answers 中存在的键（模拟 Laya 缺键场景）
            return {"answers": {k: v for k, v in self.answers.items() if k in qdict},
                    "routing": {"model": model or "english"}}
        out = {key: self._fake_answer(q["type"]) for key, q in qdict.items()}
        return {"answers": out, "routing": {"model": model or "english"}}

    @staticmethod
    def _fake_answer(qtype):
        if qtype == "choice":
            return {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.9, "technical": 0.1},
                    "confidence": 0.75, "answer_confidence": 0.9}
        if qtype == "noul":
            return {"type": "noul", "noul": 0.82, "confidence": 0.82}
        return {"type": "score", "score": 1.2, "legend": {"0": "低", "1": "中", "2": "高"},
                "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}, "confidence": 0.6}


class TestLayaAnswerMapping(unittest.TestCase):
    def test_choice(self):
        a = _laya_answer_to_jevp("choice", {
            "type": "choice", "choice": "billing",
            "probabilities": {"billing": 0.9, "technical": 0.1},
            "confidence": 0.75, "answer_confidence": 0.9})
        self.assertEqual(a.to_dict()["choice"], "billing")
        self.assertEqual(a.to_dict()["confidence"], 0.75)
        self.assertEqual(a.to_dict()["answer_confidence"], 0.9)  # 透传

    def test_noul_and_score(self):
        n = _laya_answer_to_jevp("noul", {"noul": 0.82, "confidence": 0.82})
        self.assertEqual(n.to_dict()["noul"], 0.82)
        s = _laya_answer_to_jevp("score", {"score": 1.2, "legend": {"0": "低"},
                                           "probabilities": {}, "confidence": 0.6})
        self.assertEqual(s.to_dict()["score"], 1.2)
        self.assertEqual(s.to_dict()["type"], "score")


class TestLayaBackendDispatch(unittest.TestCase):
    def setUp(self):
        self.router = FakeRouter()
        self.backend = LayaBackend(device="cpu", model="english", router=self.router)
        self.engine = DecisionEngine(backend=self.backend, tokenizer=None)

    def test_direct_flag(self):
        self.assertTrue(self.backend.direct)

    def test_decide_dispatches_direct(self):
        qs = {
            "dept": ChoiceQuestion(instructions="哪个部门？",
                                   criteria={"billing": "计费", "technical": "技术"}),
            "urgent": NoulQuestion(instructions="紧急吗？"),
            "frustration": ScoreQuestion(instructions="程度？", criteria=["低", "中", "高"]),
        }
        answers = self.engine.decide("工单: 支付失败", qs)
        self.assertTrue(self.router.calls)  # predict 被调用
        self.assertEqual(self.router.calls[-1], "english")  # model 透传
        self.assertEqual(answers["dept"].to_dict()["choice"], "billing")
        self.assertEqual(answers["urgent"].to_dict()["noul"], 0.82)
        self.assertEqual(answers["frustration"].to_dict()["score"], 1.2)

    def test_missing_key_raises(self):
        router = FakeRouter(answers={"dept": {"type": "choice", "choice": "billing"}})
        backend = LayaBackend(model="english", router=router)
        engine = DecisionEngine(backend=backend, tokenizer=None)
        qs = {
            "dept": ChoiceQuestion(instructions="x", criteria={"a": "A", "b": "B"}),
            "urgent": NoulQuestion(instructions="y"),
        }
        with self.assertRaises(RuntimeError):
            engine.decide("s", qs)


if __name__ == "__main__":
    unittest.main()