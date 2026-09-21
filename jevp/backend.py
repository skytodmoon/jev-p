"""模型后端：把 prompt 列表变成"每样本末尾位置的 logits"（B, V）。

- TransformersBackend：本地加载开源小模型（Qwen2.5 等），单次前向取末位 logits，
  **不调用 generate()**——这就是"约束解码"的引擎侧实现；
- vLLMBackend：占位，说明如何换成 raw_logprobs 路线（见 vllm-jev-decison）。
"""
from __future__ import annotations

from typing import Any, List, Sequence


class Backend:
    model_id = "unknown"

    def batch_last_logits(self, prompts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class TransformersBackend(Backend):
    """本地 transformers 模型后端。torch/transformers 惰性导入，未装也可 import 本模块。

    关键细节：batch 内 padding 后，末尾位置是 pad token——
    必须按 attention_mask 取每个样本真实最后一个 token 的 logits。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
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