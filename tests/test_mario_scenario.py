"""马里奥关卡场景的离线校验（不需要模型/GPU）。

验证：问题集可通过协议校验、候选标签正确、prompt 包含状态字段、
关卡快照数量与结构符合设计、动作按钮映射与 choice 选项一致。
"""
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for p in (os.path.join(_ROOT, "benchmarks"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from jevp.protocol import build_prompt, validate_questions
from scenarios.mario import ACTION_BUTTONS, build_questions, describe, frames


class TestMarioScenario(unittest.TestCase):
    def test_questions_validate_and_shapes(self):
        qs = build_questions()
        parsed = validate_questions({k: v.to_dict() for k, v in qs.items()})
        self.assertEqual(set(parsed), {"action", "obstacle_ahead", "gap_ahead", "danger"})
        # choice 候选项原子化
        self.assertEqual(
            parsed["action"].candidate_labels(),
            ["run", "jump", "crouch", "wait", "turn_back"],
        )
        self.assertEqual(parsed["gap_ahead"].candidate_labels(), ["true", "false"])
        self.assertEqual(parsed["danger"].candidate_labels(), ["0", "1", "2"])
        self.assertEqual(
            set(ACTION_BUTTONS),
            set(parsed["action"].candidate_labels()),
            "按钮映射必须与 choice 选项一一对应",
        )

    def test_frames_struct(self):
        fs = frames()
        self.assertGreaterEqual(len(fs), 4)
        for f in fs:
            self.assertIn("mario", f)
            self.assertIn("terrain_scan", f)
            self.assertIn("enemies", f)
            self.assertIn("tile_x", f["mario"])

    def test_frame3_has_gap(self):
        f3 = frames()[3]
        self.assertIsNotNone(f3["terrain_scan"]["gap_ahead"])

    def test_prompt_contains_state_fields(self):
        q = build_questions()["action"]
        p = build_prompt(frames()[0], "action", q)
        self.assertIn('"mario"', p)
        self.assertIn('"tile_x"', p)
        self.assertIn("run: 保持/加速向前奔跑", p)

    def test_describe(self):
        s = describe(frames()[0], 0)
        self.assertIn("帧00", s)
        self.assertIn("Mario", s)


if __name__ == "__main__":
    unittest.main()