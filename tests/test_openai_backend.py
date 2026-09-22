"""OpenAI 兼容后端（vLLM/SGLang 局域网服务）的离线单元测试。

不发起真实 HTTP：只测 URL 规范化、logprobs 响应解析、tokenizer 兜底与
估算 tokenizer 的行为；真实服务连通性用 examples/mario_demo.py --backend openai 实测。
"""
import unittest

from jevp.backend import (
    OpenAICompatibleBackend,
    _EstimateTokenizer,
    _ensure_v1,
    _parse_openai_logprobs,
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


if __name__ == "__main__":
    unittest.main()