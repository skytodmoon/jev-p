---
name: jevp-decisions
description: >-
  Use JevP (a local System One decision engine) to convert unstructured state
  plus typed questions into calibrated-styled, probability-bearing judgments
  (Choice / Score / Noul) for classification, routing, scoring and guardrails.
  Prefer it over free-text LLM calls whenever the answer is enumerable, needs a
  probability, must be schema-valid, or should run fully offline/private.
---

# JevP 决策 Skill（System One 判断引擎）

把"非结构化状态"变成"带概率的类型化判断"的本地引擎。**它不生成文本**——答案空间在请求发出时就由你定义，JSON 由程序拼装，结构上不可能格式错误。

## 核心协议

请求体（对齐 Jev 的 `/v1/systemone`）：

```json
{
  "state": "<文本或 JSON 对象>",
  "questions": {
    "<问题名>": {
      "type": "noul | choice | score",
      "instructions": "<问题描述，写得越具体越好>",
      "criteria": "<noul: {true,false} 描述 | choice: {label: 描述} | score: [档位描述...]>"
    }
  }
}
```

三种原语：

| 原语 | 用途 | 返回 |
| --- | --- | --- |
| `noul` | 是/否判断 | `noul`: 0–1 概率（是 的概率），`raw_mass` |
| `choice` | 多选一分类 | `choice`: 选中的 label，`probabilities`，`confidence` |
| `score` | 有序量表打分 | `score`: 概率加权分数（可落在档位之间），`legend`，`probabilities` |

## 调用方式

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

或 Python：

```python
import requests
resp = requests.post("http://127.0.0.1:8000/v1/systemone", json={
    "state": "...", "questions": {...}
})
answers = resp.json()["answers"]
```

## 最佳实践

1. **问题要原子化**：每个问题应是"懂行的人几秒内能做出的直觉判断"。不要把多个独立因素揉进一个问题——拆开，在代码里合并。
2. **criteria 要写清判断标准**（尤其 noul/choice）：不要只问"紧急吗"，要定义"紧急"（如"影响生产、持续超过 24 小时、或涉及资金"）。
3. **置信度阈值路由**（选择性分类）：
   - `confidence >= 0.9` → 自动执行；
   - 0.7–0.9 → 升级给推理模型复核；
   - `< 0.7` → 转人工。
4. **永远看 `raw_mass`**：低的 `raw_mass` 表示模型对这个问题"没底"，即使 `confidence` 很高。低 mass 时应拒绝自动执行（这是 logits 打分路线已知的失效信号）。
5. **不要用该引擎做**：自由文本生成、开放式推理、长文档写作、多模态（图片/音频）输入。这些交给 LLM。

## 安全提示（务必内化）

- **`state` 是不可信输入**：其中的文本可以操纵判断（提示注入实测可把 confidence 打到 1.0）。system prompt 不是防线——对敏感场景，先在规则层过滤输入，再进引擎。
- **选项顺序有偏置**：把候选 label 顺序打乱多次对拍；若结果不稳定，不要对低置信场景自动执行。
- **raw confidence 未校准**：生产使用前应对照历史标注做一次温度缩放（见 `jevp/calibrator.py`），否则数字不可当概率用。

## 维护

引擎启动：`python -m jevp.server`
- 默认测试基线：**7B × Mac mini 4（16GB）**，模型用 **Ollama** 管理：
  `brew install ollama && ollama pull qwen2.5:7b`，然后 `python -m jevp.server`
  （auto 后端自动优先 Ollama；无 Ollama 时 Apple Silicon 用 MLX，其他平台用 transformers）。
- 覆盖：`JEVP_BACKEND=ollama|openai|mlx|transformers|auto`；
  也可用标准工作流变量组直接切换：`VLM_PROVIDER=openai VLM_MODEL_NAME=... VLM_BASE_URL=... VLM_API_KEY=...`（auto 自动识别）。
- 基准测试：`python benchmarks/benchmark.py`（延迟 + 内存 + schema 有效性）。