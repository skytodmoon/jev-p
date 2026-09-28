"""AgentJevBackend 单元测试（mock post，不发真实 HTTP）。

验证：JevP questions -> AgentJev 协议转换、AgentJev answers -> JevP Answer 映射、
engine.decide dispatch、dict state 序列化、缺键报错。
"""
import json
import unittest

from jevp.backend import (
    AgentJevBackend,
    _agentjev_answer_to_jevp,
    _jepq_to_agentjev,
)
from jevp.engine import DecisionEngine
from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion


def _fake_answers(ev_questions):
    """按 AgentJev 协议构造 answers（id 回填）。"""
    out = []
    for q in ev_questions:
        if q["type"] == "boolean":
            out.append({"id": q["id"], "type": "boolean", "value": True,
                        "distribution": {"true": 0.8, "false": 0.2}})
        elif q["type"] == "choice":
            opts = list(q["options"])
            dist = {o: (0.7 if o == opts[0] else 0.3 / max(len(opts) - 1, 1)) for o in opts}
            out.append({"id": q["id"], "type": "choice", "distribution": dist})
        else:
            n = len(q["levels"])
            out.append({"id": q["id"], "type": "score", "score": 1.0, "level": 1,
                        "distribution": {i: 1.0 / n for i in range(n)}})
    return {"results": [{"answers": out}]}


class TestProtocolConversion(unittest.TestCase):
    def test_question_mapping(self):
        qs = {
            "gap": NoulQuestion(instructions="有坑吗？"),
            "action": ChoiceQuestion(instructions="动作？", criteria={"run": "跑", "jump": "跳"}),
            "danger": ScoreQuestion(instructions="危险？", criteria=["低", "高"]),
        }
        ev = _jepq_to_agentjev(qs)
        self.assertEqual([e["id"] for e in ev], ["gap", "action", "danger"])
        self.assertEqual(ev[0]["type"], "boolean")
        self.assertEqual(ev[1]["type"], "choice")
        self.assertEqual(ev[1]["options"], {"run": "跑", "jump": "跳"})
        self.assertEqual(ev[2]["levels"], ["低", "高"])
        self.assertIn("criteria", ev[0])

    def test_answer_mapping(self):
        nq = NoulQuestion(instructions="x")
        a = _agentjev_answer_to_jevp(nq, {"value": True, "distribution": {"true": 0.8, "false": 0.2}})
        self.assertEqual(a.to_dict()["noul"], 0.8)
        cq = ChoiceQuestion(instructions="x", criteria={"run": "跑", "jump": "跳"})
        a2 = _agentjev_answer_to_jevp(cq, {"distribution": {"run": 0.7, "jump": 0.3}})
        self.assertEqual(a2.to_dict()["choice"], "run")
        self.assertEqual(a2.to_dict()["confidence"], 0.7)
        sq = ScoreQuestion(instructions="x", criteria=["低", "中", "高"])
        a3 = _agentjev_answer_to_jevp(sq, {"score": 1.0, "level": 1,
                                           "distribution": {0: 0.1, 1: 0.8, 2: 0.1}})
        self.assertEqual(a3.to_dict()["score"], 1.0)
        self.assertEqual(a3.to_dict()["legend"], {"0": "低", "1": "中", "2": "高"})


class TestBackendDispatch(unittest.TestCase):
    def setUp(self):
        self.seen = {}

        def fake_post(payload):
            self.seen["payload"] = payload
            return _fake_answers(payload["questions"])

        self.backend = AgentJevBackend(endpoint="http://test:8149", post=fake_post)
        self.engine = DecisionEngine(backend=self.backend, tokenizer=None)

    def test_direct_flag_and_endpoint(self):
        self.assertTrue(self.backend.direct)
        self.assertEqual(self.backend.endpoint, "http://test:8149")

    def test_decide_full_flow(self):
        qs = {
            "gap": NoulQuestion(instructions="有坑吗？"),
            "action": ChoiceQuestion(instructions="动作？", criteria={"run": "跑", "jump": "跳"}),
            "danger": ScoreQuestion(instructions="危险？", criteria=["低", "高"]),
        }
        answers = self.engine.decide({"tile_x": 14.0}, qs)  # dict state
        # dict state 被序列化为 JSON 字符串
        state = self.seen["payload"]["state"]
        self.assertIsInstance(state, str)
        self.assertIn("tile_x", state)
        self.assertEqual(answers["gap"].to_dict()["noul"], 0.8)
        self.assertEqual(answers["action"].to_dict()["choice"], "run")
        self.assertEqual(answers["danger"].to_dict()["score"], 1.0)

    def test_missing_id_raises(self):
        def fake_post(payload):
            return {"results": [{"answers": [{"id": "other", "type": "boolean",
                                              "distribution": {"true": 1.0, "false": 0.0}}]}]}

        backend = AgentJevBackend(post=fake_post)
        engine = DecisionEngine(backend=backend, tokenizer=None)
        with self.assertRaises(RuntimeError):
            engine.decide("s", {"gap": NoulQuestion(instructions="x")})


if __name__ == "__main__":
    unittest.main()