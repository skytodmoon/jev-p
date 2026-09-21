"""模型后端：把 prompt 列表变成"每样本末尾位置的 logits"（B, V）。

默认测试基线：**7B 模型 × Mac mini 4（M4 / 16GB）**。

- MLXBackend（Apple Silicon 首选）：mlx-lm 加载 4bit 量化模型
  （默认 ``mlx-community/Qwen2.5-7B-Instruct-4bit``），单次前向取末位 logits，
  不调用生成流程。首版为逐条前向（正确性优先，无 padding 复杂度）。
- TransformersBackend（通用回退）：torch + transformers，单次前向取末位 logits。
- vLLMBackend：占位（raw_logprobs 路线，见 README Roadmap）。

默认选择逻辑 auto_backend()：
  Apple Silicon 且可 import mlx_lm -> MLXBackend（推荐）；
  否则 -> TransformersBackend。
"""
from __future__ import annotations

import os
import platform
from typing import Any, List, Optional, Sequence

# 默认测试基线：7B
DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class Backend:
    model_id = "unknown"

    def batch_last_logits(self, prompts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


class MLXBackend(Backend):
    """Apple Silicon（MLX）后端：本地跑 4bit 量化小模型（推荐 7B）。

    依赖（仅目标机安装）::

        pip install mlx-lm

    惰性导入 mlx / mlx_lm：在非 Apple 平台 import 本模块不会报错。
    注：首版为逐条前向（每个 prompt 一次 forward），一次 batch 全部问题的
    并行优化见 README Roadmap（Transformers/vLLM 后端已支持 batch）。
    """

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

        # 返回 (model, tokenizer)；model 是 mlx 的 ModelForCausalLM
        self.model, self.tokenizer = mlxu.load(self.model_name)

    def batch_last_logits(self, prompts: Sequence[str]) -> List[List[float]]:
        import mlx.core as mx

        rows: List[List[float]] = []
        for p in prompts:
            ids = self.tokenizer.encode(p)[: self.max_length]
            x = mx.array([ids])
            out = self.model(x)
            if isinstance(out, tuple):
                out = out[0]  # 某些实现返回 (logits, cache)
            mx.eval(out)
            logits = out[0, -1, :] if out.ndim == 3 else out[-1, :]
            rows.append([float(v) for v in logits.tolist()])
        return rows


class TransformersBackend(Backend):
    """本地 transformers 模型后端（torch，通用平台）。torch/transformers 惰性导入。

    关键细节：batch 内 padding 后，末尾位置是 pad token——
    必须按 attention_mask 取每个样本真实最后一个 token 的 logits。
    """

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
        import torch  # noqa: F401（确保 torch 已安装）
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
        # 每个样本真实最后位置（非 pad）
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        rows = out.logits[torch.arange(len(prompts), device=self.device), last_idx]
        return rows.detach().cpu().float().tolist()


class vLLMBackend(Backend):
    """占位：vLLM raw_logprobs 路线。

    工程要点（详见 vllm-jev-decison）：
    - 候选标签需是 tokenizer 校验过的单 token（A–P 等）；
    - 读取 ``--logprobs-mode raw_logprobs`` 的候选打分，argmax 选值；
    - 已知坑：logits 不是校准概率；选项顺序有偏置；state 可被提示注入。
    """

    model_id = "vllm (not implemented)"

    def batch_last_logits(self, prompts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError(
            "vLLMBackend 为占位实现。可改用 vLLM 的 raw_logprobs 接口（见 README Roadmap）。"
        )


def auto_backend(
    model: Optional[str] = None,
    backend: str = "auto",
    max_length: Optional[int] = None,
    device: str = "auto",
    load_in_4bit: bool = False,
) -> Backend:
    """按运行环境自动选择后端（默认测试基线：7B × Mac mini 4）。

    backend: "auto"（Apple Silicon 且有 mlx-lm -> MLX，否则 transformers）
             | "mlx" | "transformers" | "vllm"
    model:   显式模型名；None 时按后端取默认 7B。
    """
    from .engine import DecisionEngine  # noqa: F401（保持导入习惯一致）

    mlx = backend == "mlx" or (backend == "auto" and _is_apple_silicon())
    if mlx:
        if model is None:
            model = os.environ.get("JEVP_MLX_MODEL", DEFAULT_MLX_MODEL)
        return MLXBackend(
            model_name=model,
            max_length=max_length or int(os.environ.get("JEVP_MAX_LENGTH", "2048")),
        )

    if backend == "vllm":
        return vLLMBackend()

    # transformers（含 auto 但非 Apple Silicon 的情况）
    if model is None:
        model = os.environ.get("JEVP_MODEL", DEFAULT_TRANSFORMERS_MODEL)
    return TransformersBackend(
        model_name=model,
        device=device,
        max_length=max_length or int(os.environ.get("JEVP_MAX_LENGTH", "1024")),
        load_in_4bit=load_in_4bit,
    )