"""企业-产品匹配场景的离线校验（单元测试，无需模型/GPU）。"""
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, os.path.join(_ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from jevp.protocol import validate_questions  # noqa: E402
from scenarios.enterprise_match import (  # noqa: E402
    PRODUCT,
    build_questions,
    describe,
    expected,
    frames,
)


class TestEnterpriseMatchScenario(unittest.TestCase):
    def test_product_configured(self):
        self.assertIn("name", PRODUCT)
        self.assertGreater(len(PRODUCT["description"]), 20)

    def test_questions_valid_and_atomic(self):
        qs = build_questions()
        parsed = validate_questions({k: v.to_dict() for k, v in qs.items()})
        self.assertEqual(set(parsed), {
            "industry_fit", "demand_signal", "budget_scale",
            "strategic_fit", "urgency", "match_score",
        })
        # match_score 档位 0-3
        self.assertEqual(parsed["match_score"].candidate_labels(), ["0", "1", "2", "3"])

    def test_frames_cover_levels(self):
        fs = frames()
        self.assertGreaterEqual(len(fs), 8)
        levels = {expected(f)["match_score"] for f in fs}
        # 覆盖全部 4 档（0/1/2/3），保证排序有区分度
        self.assertEqual(levels, {0, 1, 2, 3})

    def test_frames_structure(self):
        for f in frames():
            for key in ("name", "basic", "bidding", "website", "supply_chain"):
                self.assertIn(key, f)

    def test_describe(self):
        self.assertIn("华银", describe(frames()[0], 0))


if __name__ == "__main__":
    unittest.main()