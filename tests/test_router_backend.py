"""RouterBackend（语言感知路由）单元测试：fake 后端注入，不联网。

验证：中文决策路由到 zh 后端、英文到 en 后端、en 缺失回退 zh、
低置信/低 mass 降级标记、engine.decide 钩子接管。
"""
import unittest

from jevp.backend import RouterBackend, _answer_confidence, is_chinese_like
from jevp.engine import DecisionEngine
from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion


class FakeBackend:
    direct = True

    def __init__(self, tag, payload_override=None):
        self.tag = tag
        self.model_id = f"fake-{tag}"
        self.payload = payload_override or {"confidence": 0.9}

    def direct_decide(self, state, questions):
        answers = {}
        for k, q in questions.items():
            if q.type == "noul":
                answers[k] = __import__("jevp.protocol", fromlist=["Answer"]).Answer(
                    "noul", {"noul": 0.8, "confidence": 0.8})
            elif q.type == "choice":
                answers[k] = __import__("jevp.protocol", fromlist=["Answer"]).Answer(
                    "choice", {"choice": "x", "confidence": self.payload.get("confidence"),
                               "probabilities": {"x": self.payload.get("confidence", 0.5)},
                               "raw_mass": self.payload.get("mass")})
            else:
                answers[k] = __import__("jevp.protocol", fromlist=["Answer"]).Answer(
                    "score", {"score": 1.0, "confidence": self.payload.get("confidence"),
                              "legend": {}})
        return answers


ZH_Q = {"m": ChoiceQuestion(instructions="该企业属于哪个行业？", criteria={"a": "钢铁", "b": "化工"})}
EN_Q = {"m": ChoiceQuestion(instructions="Which industry?", criteria={"a": "steel", "b": "chem"})}


class TestLanguageDetection(unittest.TestCase):
    def test_chinese_zh(self):
        self.assertTrue(is_chinese_like("客户：订单一直没发货", ZH_Q))
        # 中文 criteria 起决定性作用（即使 state 是英文 JSON）
        self.assertTrue(is_chinese_like({"tile_x": 14.0}, ZH_Q))

    def test_english_en(self):
        self.assertFalse(is_chinese_like("The payment failed three days ago", EN_Q))


class TestRouterBackend(unittest.TestCase):
    def test_zh_route(self):
        zh, en = FakeBackend("zh"), FakeBackend("en")
        r = RouterBackend(zh_backend=zh, en_backend=en)
        engine = DecisionEngine(backend=r, tokenizer=None)
        ans = engine.decide("工单：支付失败三天", ZH_Q)  # 走 route_decide
        self.assertEqual(r.last_meta["lang"], "zh")
        self.assertEqual(r.last_meta["backend"], "fake-zh")

    def test_en_route(self):
        zh, en = FakeBackend("zh"), FakeBackend("en")
        r = RouterBackend(zh_backend=zh, en_backend=en)
        engine = DecisionEngine(backend=r, tokenizer=None)
        engine.decide("Refund request, please pay back today", EN_Q)
        self.assertEqual(r.last_meta["lang"], "en")
        self.assertEqual(r.last_meta["backend"], "fake-en")

    def test_en_missing_falls_back_to_zh(self):
        zh = FakeBackend("zh")
        r = RouterBackend(zh_backend=zh, en_backend=None)
        engine = DecisionEngine(backend=r, tokenizer=None)
        engine.decide("Refund today please", EN_Q)
        self.assertEqual(r.last_meta["lang"], "en")
        self.assertEqual(r.last_meta["backend"], "fake-zh")  # 回退 zh

    def test_degraded_flag_low_conf(self):
        zh = FakeBackend("zh", payload_override={"confidence": 0.3})
        r = RouterBackend(zh_backend=zh, en_backend=None, low_conf=0.5)
        engine = DecisionEngine(backend=r, tokenizer=None)
        ans = engine.decide("工单：支付失败", ZH_Q)
        self.assertTrue(r.last_meta["degraded"])
        self.assertIn("m", r.last_meta["low_confidence_keys"])
        # noul 答案的置信度取 max(n, 1-n)=0.8，不受低 conf 影响
        noul_q = {"u": NoulQuestion(instructions="紧急吗？")}
        ans2 = engine.decide("工单", noul_q)
        self.assertGreater(_answer_confidence(ans2["u"]), 0.3)

    def test_no_degrade_high_conf(self):
        zh = FakeBackend("zh", payload_override={"confidence": 0.95})
        r = RouterBackend(zh_backend=zh, en_backend=None, low_conf=0.5)
        DecisionEngine(backend=r, tokenizer=None).decide("工单", ZH_Q)
        self.assertFalse(r.last_meta["degraded"])


if __name__ == "__main__":
    unittest.main()