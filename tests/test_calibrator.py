"""校准器单元测试（零依赖）。"""
import unittest

from jevp.calibrator import (
    TemperatureScaler,
    confidence_router,
    expected_calibration_error,
)


class TestECE(unittest.TestCase):
    def test_perfect_calibration_zero(self):
        # 10 个样本 conf=0.5，恰好 5 个正确 → ECE≈0
        confs = [0.5] * 10
        corrects = [True] * 5 + [False] * 5
        self.assertAlmostEqual(expected_calibration_error(confs, corrects), 0.0, places=6)

    def test_overconfident_positive(self):
        # 全部 0.9 置信但全部错误 → 明显过度自信
        ece = expected_calibration_error([0.9] * 10, [False] * 10)
        self.assertGreater(ece, 0.5)

    def test_requires_equal_length(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([0.5], [True, False])


class TestTemperatureScaler(unittest.TestCase):
    def test_scale_normalizes(self):
        s = TemperatureScaler(temperature=2.0)
        out = s.scale([[0.9, 0.09, 0.01], [0.5, 0.3, 0.2]])
        for row in out:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_fit_finds_high_temperature_on_overconfident(self):
        # 模型给"错误预测"也打出 0.999 的极端概率（过度自信且犯错）：
        # 一半样本正确、一半样本错误，都报 0.999 置信 → 软化（大 T）能显著降低 NLL
        probs = [[0.999, 0.001]] * 50
        labels = [0] * 25 + [1] * 25
        scaler = TemperatureScaler.fit(probs, labels)
        self.assertEqual(scaler.temperature, 10.0)  # 网格上限被选中（软化最有利）

    def test_fit_keeps_sharp_on_perfect_predictions(self):
        # 全部预测正确时，锐化（小 T）降低 NLL —— 不应被过度软化
        probs = [[0.999, 0.001]] * 50
        labels = [0] * 50
        scaler = TemperatureScaler.fit(probs, labels)
        self.assertLessEqual(scaler.temperature, 1.0)

    def test_temperature_one_is_identity(self):
        s = TemperatureScaler(temperature=1.0)
        out = s.scale([[0.9, 0.1]])
        self.assertAlmostEqual(out[0][0], 0.9, places=6)


class TestRouter(unittest.TestCase):
    def test_routes(self):
        self.assertEqual(confidence_router(0.95), "auto")
        self.assertEqual(confidence_router(0.8), "review")
        self.assertEqual(confidence_router(0.5), "human")
        self.assertEqual(confidence_router(0.9), "auto")

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            confidence_router(1.5)


if __name__ == "__main__":
    unittest.main()