"""模型后端：把 prompt 列表变成"每样本末尾位置的打分"。

默认测试基线：**7B 模型 × Mac mini 4（M4 / 16GB）**，模型用 **Ollama** 下载与管理。

后端（auto_backend 自动选择，优先级从上到下）：
1. OllamaBackend（本机 Ollama 服务可用时默认）：``ollama pull qwen2.5:7b`` 后
   走 OpenAI 兼容 ``/v1/chat/completions`` 的 ``logprobs``（top-k）做**字符串级候选打分**
   —— Apple Silicon 上由 Ollama 自动 Metal 加速，模型由 ``ollama`` CLI 管理/更新；
2. MLXBackend（Apple Silicon、无 Ollama 时）：mlx-lm 直接加载 4bit 量化权重
   （``mlx-community/Qwen2.5-7B-Instruct-4bit``）；
3. TransformersBackend（其他平台回退）：torch + transformers。
4. vLLMBackend：占位（raw_logprobs 路线）。

注：Ollama 的后端只返回 top-k logprob，因此 raw_mass 为 **top-k 近似**；
MLX/Transformers 后端返回完整词表 logits，raw_mass 为精确份额。
"""
from __future__ import annotations

import json
import os
import platform
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

# 默认测试基线：7B
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"          # ollama pull qwen2.5:7b
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class Backend:
    """打分后端基类。

    candidate_scoring：
      "token_id"    —— batch_last_logits 返回 (B, V) 稠密/logits 序列，engine 按 token id 打分
                         （MLX / Transformers）
      "token_string"—— batch_last_logits 返回 {token字符串: logprob} 稀疏字典，
                         engine 按候选 token 字符串（含 BPE 空格变体）打分（Ollama top-k）

    direct：判别式后端（如 Laya）不产生词表 logits，实现
            ``direct_decide(state, questions) -> Dict[str, Answer]``；
            DecisionEngine.decide 检测到 ``direct=True`` 时走该路径。
    """

    model_id = "unknown"
    candidate_scoring = "token_id"
    direct = False

    def batch_last_logits(self, prompts: Sequence[str]) -> List[Any]:
        raise NotImplementedError

    def direct_decide(self, state: Any, questions: "Dict[str, Any]") -> "Dict[str, Any]":
        raise NotImplementedError("该后端不是判别式后端，不支持直接判定")


# --------------------------------------------------------------------------- #
# Ollama 后端（默认，模型下载/管理走 ollama CLI）
# --------------------------------------------------------------------------- #
class _JsonClient:
    """最小 HTTP JSON 客户端（urllib，无第三方依赖）。"""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        url = self.base_url + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, body: dict) -> Any:
        return self._request("POST", path, body)


class OllamaTokenizer:
    """Ollama 的远程 tokenizer 适配：engine 只用到 ``tokenizer(text)["input_ids"]``。"""

    def __init__(self, model: str, base_url: str):
        self._client = _JsonClient(base_url)
        self.model = model

    def __call__(self, text, add_special_tokens: bool = False) -> Dict[str, Any]:
        resp = self._client.post("/api/tokenize", {"model": self.model, "content": str(text)})
        return {"input_ids": [int(t) for t in resp.get("tokens", [])]}


class OllamaBackend(Backend):
    """Ollama 后端：OpenAI 兼容 /v1/chat/completions + logprobs。

    只取 top-k logprob（不生成文本），候选打分在 engine 内按 token 字符串完成。
    限制：Ollama 不暴露完整词表 logits，原始 raw_mass 是 top-k 近似；
    候选若未进 top-k，其 logprob 视为 -100（低置信）。
    """

    candidate_scoring = "token_string"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE,
        num_ctx: int = 2048,
        top_logprobs: int = 16,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.top_logprobs = top_logprobs
        self.model_id = f"ollama:{model}"
        self._client = _JsonClient(base_url, timeout=timeout)
        self._ensure_model_installed()
        self.tokenizer = OllamaTokenizer(model, base_url)

    def _ensure_model_installed(self) -> None:
        try:
            tags = self._client.get("/api/tags")
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"无法连接 Ollama（{self.base_url}）：{exc}。请先安装并启动 Ollama，"
                f"然后 `ollama pull {self.model}`"
            ) from exc
        names = {t.get("name", "") for t in tags.get("models", [])}
        if self.model not in names:
            raise RuntimeError(
                f"Ollama 未安装模型 {self.model!r}，请先执行: ollama pull {self.model}"
            )

    def batch_last_logits(self, prompts: Sequence[str]) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for p in prompts:
            rows.append(self._top_logprobs(p) or {})
        return rows

    def _top_logprobs(self, prompt: str) -> Dict[str, float]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "options": {"num_ctx": self.num_ctx},
        }
        resp = self._client.post("/v1/chat/completions", body)
        try:
            content = resp["choices"][0]["logprobs"]["content"]
            top = content[0].get("top_logprobs", [])
        except (KeyError, IndexError, TypeError):
            return {}
        return {t["token"]: float(t["logprob"]) for t in top}


# --------------------------------------------------------------------------- #
# MLX / Transformers 本地后端
# --------------------------------------------------------------------------- #
def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


class MLXBackend(Backend):
    """Apple Silicon（MLX）后端：本地跑 4bit 量化模型（无 Ollama 时的备选）。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MLX_MODEL,
        max_length: int = 2048,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model_id = model_name
        self._load()

    def _load(self) -> None:
        import mlx_lm.utils as mlxu

        self.model, self.tokenizer = mlxu.load(self.model_name)

    def batch_last_logits(self, prompts: Sequence[str]) -> List[List[float]]:
        import mlx.core as mx

        rows: List[List[float]] = []
        for p in prompts:
            ids = self.tokenizer.encode(p)[: self.max_length]
            x = mx.array([ids])
            out = self.model(x)
            if isinstance(out, tuple):
                out = out[0]
            mx.eval(out)
            logits = out[0, -1, :] if out.ndim == 3 else out[-1, :]
            rows.append([float(v) for v in logits.tolist()])
        return rows


class TransformersBackend(Backend):
    """torch + transformers 后端（通用平台，batch 一次前向）。"""

    @property
    def candidate_scoring(self) -> str:
        return "token_id"

    def __init__(
        self,
        model_name: str = DEFAULT_TRANSFORMERS_MODEL,
        device: str = "auto",
        max_length: int = 1024,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model_id = model_name
        self._load(device=device, load_in_4bit=load_in_4bit)

    def _load(self, device: str, load_in_4bit: bool) -> None:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def batch_last_logits(self, prompts: Sequence[str]) -> List[List[float]]:
        import torch

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        rows = out.logits[torch.arange(len(prompts), device=self.device), last_idx]
        return rows.detach().cpu().float().tolist()


class vLLMBackend(Backend):
    """占位：vLLM raw_logprobs 路线。"""

    model_id = "vllm (not implemented)"

    def batch_last_logits(self, prompts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError(
            "vLLMBackend 为占位实现。可改用 vLLM 的 raw_logprobs 接口（见 README Roadmap）。"
        )


# --------------------------------------------------------------------------- #
# OpenAI 兼容后端（vLLM / SGLang / Xinference 等，局域网自建服务）
# --------------------------------------------------------------------------- #
def _ensure_v1(base_url: str) -> str:
    """确保 base_url 以 /v1 结尾（chat completions 端点路径）。"""
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def _parse_openai_logprobs(resp: Any) -> Dict[str, float]:
    """解析 OpenAI 兼容响应的 top-logprobs -> {token字符串: logprob}。"""
    try:
        content = resp["choices"][0]["logprobs"]["content"]
        top = content[0].get("top_logprobs", [])
    except (KeyError, IndexError, TypeError):
        return {}
    return {t["token"]: float(t["logprob"]) for t in top}


class _EstimateTokenizer:
    """无本地 tokenizer 时的兜底：按字符数估算（输入 token 统计用，非打分）。"""

    def __call__(self, text, add_special_tokens: bool = False) -> Dict[str, Any]:
        return {"input_ids": list(range(max(1, len(str(text)) // 4 + 1)))}


def _resolve_openai_config(model_arg: Optional[str] = None) -> Dict[str, Any]:
    """解析 openai 后端配置：显式参数 > JEVP_OPENAI_* > VLM_*（标准工作流变量）。

    变量组一（JevP 原生）：JEVP_OPENAI_MODEL / JEVP_OPENAI_BASE_URL / JEVP_OPENAI_API_KEY
    变量组二（通用标准）：VLM_MODEL_NAME / VLM_BASE_URL / VLM_API_KEY / VLM_PROVIDER=openai
    """
    env = os.environ
    model = model_arg or env.get("JEVP_OPENAI_MODEL") or env.get("VLM_MODEL_NAME")
    if not model:
        raise ValueError(
            "backend=openai 需要模型：--model / JEVP_OPENAI_MODEL / VLM_MODEL_NAME"
        )
    base_url = (
        env.get("JEVP_OPENAI_BASE_URL")
        or env.get("VLM_BASE_URL")
        or "http://127.0.0.1:8001/v1"
    )
    api_key = env.get("JEVP_OPENAI_API_KEY") or env.get("VLM_API_KEY")
    top_logprobs = int(env.get("JEVP_OPENAI_TOP_LOGPROBS", "20"))
    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "top_logprobs": top_logprobs,
    }


class OpenAICompatibleBackend(Backend):
    """OpenAI 兼容服务后端（vLLM 等）：/v1/chat/completions + top-level logprobs。

    - 只取 top-k logprob（max_tokens=1，不生成文本），engine 按 token 字符串打分；
    - 面向 Qwen3 等"带思考链"模型：默认带 ``chat_template_kwargs={"enable_thinking": False}``
      关闭 thinking（确定性打分任务必需，否则 max_tokens=1 会被思考 token 吃掉）；
    - raw_mass 为 top-k 近似（同 Ollama 后端，不暴露完整词表）；
    - API Key 可选（服务无鉴权时留空）。

    配置：JEVP_OPENAI_BASE_URL / JEVP_OPENAI_MODEL / JEVP_OPENAI_API_KEY
    """

    candidate_scoring = "token_string"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        top_logprobs: int = 20,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = _ensure_v1(base_url)
        self.api_key = api_key
        self.top_logprobs = top_logprobs
        self.chat_template_kwargs = (
            chat_template_kwargs
            if chat_template_kwargs is not None
            else {"enable_thinking": False}
        )
        self.model_id = f"openai:{model}@{self.base_url}"
        self._client = _JsonClient(self.base_url, timeout=timeout)
        self.tokenizer = self._make_tokenizer()

    def _make_tokenizer(self) -> Any:
        # 尝试 vLLM /tokenize；失败退化为估算（仅影响 usage 统计，不影响打分）
        for path in ("/tokenize", "/v1/tokenize"):
            try:
                self._client.post(path, {"model": self.model, "prompt": "hello", "add_special_tokens": False})
                return _VLLMTokenizer(self._client, self.model)
            except Exception:
                continue
        return _EstimateTokenizer()

    def batch_last_logits(self, prompts: Sequence[str]) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for p in prompts:
            rows.append(self._top_logprobs(p) or {})
        return rows

    def _top_logprobs(self, prompt: str) -> Dict[str, float]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        resp = self._post_chat(body)
        return _parse_openai_logprobs(resp)

    def _post_chat(self, body: Dict[str, Any]) -> Any:
        import urllib.request

        url = self.base_url + "/chat/completions"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"OpenAI 兼容服务返回 HTTP {exc.code}: {detail}") from exc
        return json.loads(raw) if raw else {}


class _VLLMTokenizer:
    """vLLM 的 /tokenize 远程 tokenizer（仅用于输入 token 统计）。"""

    def __init__(self, client: _JsonClient, model: str):
        self._client = client
        self.model = model

    def __call__(self, text, add_special_tokens: bool = False) -> Dict[str, Any]:
        try:
            resp = self._client.post(
                "/tokenize",
                {"model": self.model, "prompt": str(text), "add_special_tokens": add_special_tokens},
            )
            tokens = resp.get("tokens") or resp.get("input_ids") or []
            if tokens:
                return {"input_ids": [int(t) for t in tokens]}
        except Exception:
            pass
        return _EstimateTokenizer()(text)


# --------------------------------------------------------------------------- #
# Laya 后端（Apache-2.0 判别式 System1 决策模型，Jev 开源平替）
# questions 协议与 JevP 完全同构：直接 router.predict(state, questions) 取答案，
# 不经过 LLM logits 打分。借鉴其工程实践：一键接入、按 checkpoint 路由。
# --------------------------------------------------------------------------- #
def _laya_answer_to_jevp(qtype: str, raw: Dict[str, Any]) -> Any:
    """把 Laya 单个答案映射为 JevP Answer payload（保持输出契约一致）。"""
    from .protocol import Answer

    out: Dict[str, Any] = {"type": qtype}
    if qtype == "choice":
        out["choice"] = raw.get("choice")
        out["probabilities"] = raw.get("probabilities") or {}
        out["confidence"] = raw.get("confidence")
    elif qtype == "noul":
        out["noul"] = raw.get("noul")
        if raw.get("confidence") is not None:
            out["confidence"] = raw["confidence"]
    elif qtype == "score":
        out["score"] = raw.get("score")
        out["legend"] = raw.get("legend") or {}
        out["probabilities"] = raw.get("probabilities") or {}
        out["confidence"] = raw.get("confidence")
    # Laya 额外字段原样透传（answer_confidence / action 等）
    for k, v in raw.items():
        if k not in out:
            out[k] = v
    return Answer(qtype, out)


class LayaBackend(Backend):
    """Laya（判别式决策模型）后端：直接输出 typed answers，无 LLM 服务依赖。

    配置：JEVP_LAYA_DEVICE（cpu|cuda|mps，默认 cpu）、JEVP_LAYA_MODEL
    （auto|english|multilingual|typed-decisions，默认 auto 由 Router 语言检测路由）。
    router 参数供测试注入；生产环境懒加载 ``laya.Router``（首调才下载/加载 checkpoint）。
    """

    direct = True

    def __init__(
        self,
        device: str = "cpu",
        model: Optional[str] = None,
        router: Optional[Any] = None,
    ):
        self.device = device
        self.model = model or "auto"
        self.model_id = f"laya/{self.model}@{device}"
        self._router = router

    def _get_router(self) -> Any:
        if self._router is None:
            from laya import Router

            self._router = Router(preload=False, device=self.device)
        return self._router

    def direct_decide(self, state: Any, questions: "Dict[str, Any]") -> "Dict[str, Any]":
        qdict = {k: q.to_dict() for k, q in questions.items()}
        # "auto"/None -> 不传 model，由 Router 自动语言检测路由；否则显式指定 checkpoint
        predict_model = None if self.model in (None, "auto") else self.model
        result = self._get_router().predict(state, qdict, model=predict_model)
        answers_raw = result.get("answers", {})
        answers: Dict[str, Any] = {}
        for k, q in questions.items():
            if k not in answers_raw:
                raise RuntimeError(f"Laya 未返回问题 {k!r} 的答案")
            answers[k] = _laya_answer_to_jevp(q.type, answers_raw[k])
        return answers


# --------------------------------------------------------------------------- #
# 自动选择
# --------------------------------------------------------------------------- #
def _laya_env_device() -> str:
    return os.environ.get("JEVP_LAYA_DEVICE", "cpu")
def _ollama_available(base_url: str = DEFAULT_OLLAMA_BASE, timeout: float = 1.5) -> bool:
    try:
        _JsonClient(base_url, timeout=timeout).get("/api/tags")
        return True
    except Exception:
        return False


def auto_backend(
    model: Optional[str] = None,
    backend: str = "auto",
    max_length: Optional[int] = None,
    device: str = "auto",
    load_in_4bit: bool = False,
) -> Backend:
    """按运行环境自动选择后端（默认测试基线：7B × Mac mini 4，模型走 Ollama）。

    backend: "auto"（Ollama → MLX → Transformers 依可用性降级）| "ollama" | "mlx"
             | "transformers" | "vllm"
    model:   显式模型/Tag；None 时按后端取默认 7B。
    """
    if backend == "ollama":
        return OllamaBackend(model=model or os.environ.get("JEVP_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))

    if backend == "laya":
        # Laya 判别式后端（Jev 开源平替）：模型选择 auto|english|multilingual|typed-decisions
        laya_model = model or os.environ.get("JEVP_LAYA_MODEL")
        return LayaBackend(
            device=os.environ.get("JEVP_LAYA_DEVICE", "cpu"),
            model=laya_model,
        )

    if backend == "openai":
        cfg = _resolve_openai_config(model)
        return OpenAICompatibleBackend(**cfg)

    if backend == "auto":
        # 0) 标准工作流变量：VLM_PROVIDER=openai 且给出 base_url -> 直接用
        if os.environ.get("VLM_PROVIDER") == "openai" and os.environ.get("VLM_BASE_URL"):
            return OpenAICompatibleBackend(**_resolve_openai_config(model))
        # 1) Ollama 可用（默认管理方式）
        if model is None and _ollama_available():
            try:
                return OllamaBackend(
                    model=os.environ.get("JEVP_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
                )
            except RuntimeError as exc:
                print(f"[JevP] Ollama 不可用（{exc}），尝试其他后端…")
        # 2) Apple Silicon：MLX
        if _is_apple_silicon():
            try:
                import mlx_lm  # noqa: F401
            except ImportError:
                pass
            else:
                if model is None:
                    model = os.environ.get("JEVP_MLX_MODEL", DEFAULT_MLX_MODEL)
                return MLXBackend(model_name=model, max_length=max_length or 2048)

    # 3) transformers（含显式指定）
    if backend == "mlx":
        if model is None:
            model = os.environ.get("JEVP_MLX_MODEL", DEFAULT_MLX_MODEL)
        return MLXBackend(model_name=model, max_length=max_length or 2048)
    if backend == "vllm":
        return vLLMBackend()
    if model is None:
        model = os.environ.get("JEVP_MODEL", DEFAULT_TRANSFORMERS_MODEL)
    return TransformersBackend(
        model_name=model,
        device=device,
        max_length=max_length or int(os.environ.get("JEVP_MAX_LENGTH", "1024")),
        load_in_4bit=load_in_4bit,
    )