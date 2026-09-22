"""OpenAI 兼容后端（vLLM/SGLang 局域网服务）的离线单元测试。

不发起真实 HTTP：只测 URL 规范化、logprobs 响应解析、tokenizer 兜底与
估算 tokenizer 的行为；真实服务连通性用 examples/mario_demo.py --backend openai 实测。
"""
import os
import unittest
from unittest import mock

from jevp.backend import (
    OpenAICompatibleBackend,
    _EstimateTokenizer,
    _ensure_v1,
    _parse_openai_logprobs,
    _resolve_openai_config,
)


class TestEnsureV1(unittest.TestCase):
    def test_already_v1(self):
        self.assertEqual(_ensure_v1("http://192.168.1.52:8001/v1"),
                         "http://192.168.1.52:8001/v1")

    def test_appends_v1(self):
        self.assertEqual(_ensure_v1("http://192.168.1.52:8001"),
                         "http://192.168.1.52:8001/v1")

    def test_trailing_slash_stripped(self):
        self.assertEqual(_ensure_v1("http://h:8001/"),
                         "http://h:8001/v1")


class TestParseOpenAILogprobs(unittest.TestCase):
    def test_normal(self):
        resp = {
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": " run",
                        "logprob": -0.2,
                        "top_logprobs": [
                            {"token": " run", "logprob": -0.2},
                            {"token": " jump", "logprob": -1.0},
                        ],
                    }]
                }
            }]
        }
        out = _parse_openai_logprobs(resp)
        self.assertEqual(out, {" run": -0.2, " jump": -1.0})

    def test_malformed_returns_empty(self):
        self.assertEqual(_parse_openai_logprobs({}), {})
        self.assertEqual(_parse_openai_logprobs({"choices": []}), {})


class TestEstimateTokenizer(unittest.TestCase):
    def test_returns_ids(self):
        tok = _EstimateTokenizer()
        r = tok("hello world", add_special_tokens=False)
        self.assertIn("input_ids", r)
        self.assertGreaterEqual(len(r["input_ids"]), 1)


class TestOpenAICompatibleBackendInit(unittest.TestCase):
    def test_default_thinking_disabled(self):
        # 不触发 HTTP：只校验构造参数解析（通过替换 _client 不做网络调用）
        b = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
        self.assertEqual(OpenAICompatibleBackend(
            model="m", base_url="http://127.0.0.1:1/v1").chat_template_kwargs,
            {"enable_thinking": False})

    def test_custom_template_kwargs_preserved(self):
        b = OpenAICompatibleBackend(
            model="m", base_url="http://127.0.0.1:1/v1",
            chat_template_kwargs={"enable_thinking": True, "foo": 1})
        self.assertEqual(b.chat_template_kwargs, {"enable_thinking": True, "foo": 1})


class TestVLMEnvConfig(unittest.TestCase):
    """标准工作流变量组（VLM_*）的解析，不发起任何网络请求。"""

    def test_vlm_vars_used(self):
        with mock.patch.dict(os.environ, {
            "VLM_MODEL_NAME": "qwen3.8-27b",
            "VLM_BASE_URL": "http://192.168.1.52:8001/v1",
            "VLM_API_KEY": "sk-x",
            "VLM_PROVIDER": "openai",
        }, clear=False):
            cfg = _resolve_openai_config()
        self.assertEqual(cfg["model"], "qwen3.8-27b")
        self.assertEqual(cfg["base_url"], "http://192.168.1.52:8001/v1")
        self.assertEqual(cfg["api_key"], "sk-x")

    def test_jevp_vars_fallback_and_explicit_wins(self):
        with mock.patch.dict(os.environ, {
            "VLM_MODEL_NAME": "vlm-model",
            "JEVP_OPENAI_MODEL": "jevp-model",
        }, clear=False):
            cfg = _resolve_openai_config()
        self.assertEqual(cfg["model"], "jevp-model")  # JEVP_* 优先于 VLM_*
        cfg2 = _resolve_openai_config("explicit")
        self.assertEqual(cfg2["model"], "explicit")   # 显式参数最高优先

    def test_missing_model_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                _resolve_openai_config()


if __name__ == "__main__":
    unittest.main()