# JevP

> 在现成的开源小 LLM 上，用**约束解码（constrained decoding）**复刻 TypeSafe AI 的 **Jev（System One 模型）**：把非结构化状态（state）+ 类型化问题（Choice / Score / Noul）一次性转成**带概率的结构化判断**，程序化拼装 JSON，**绝不生成自由文本**。

JevP 是一个**最小可运行**的实现：本地推理、零云端成本、数据不出域。它复刻的是 Jev 的**功能形态**（类型安全输出 + 单次前向并行判定 + 低延迟），但**不追求也不宣称复刻 Jev 的模型权重与校准质量**——差距是真实的，见文末「与 Jev 的差距」。

---

## 1. 为什么这么做（背景）

TypeSafe AI 的 Jev 是一个"System One 模型"：输入 `state` + 类型化 `questions`，输出带概率的答案；不生成文本，因此在结构上不可能产生坏 JSON；多个问题对同一 state **并行评估**，一次前向完成。

社区已有两条被实测验证的复刻路线：

- **[jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop)**：在 Apple Silicon 上用 Qwen2.5-1.5B/7B、Qwen3-8B + "prefill 一次、KV cache 广播、每字段单 token 打分"，28 字段判定 0.4–2.0s，**三个模型朴素写 JSON 全部失败、约束路径全部 schema 有效**；但在 TypeSafe 官方 373 对公开问题上，本地 7B 与 Jev 的差距约 13 个点（73.8% vs 86.6% 一致率）。
- **[vllm-jev-decison](https://github.com/siliconkernel/vllm-jev-decison)**：候选标签映射单 token → 读取 `logprobs` → argmax；并实测了**选项顺序偏置**与 **state 提示注入**两个坑。

JevP 把这两条路线的内核（约束打分 + 显式暴露未校准的 `raw_mass`）收敛成一个自包含的 Python 工程：协议层、打分引擎、校准器、`/v1/systemone` 服务、Agent Skill，全部可见可改。

## 2. 架构

```
state + questions (Choice/Score/Noul)
        │
        ▼
   build_prompt()           每问一 prompt，批量组成一个 batch
        │
        ▼
   Backend.batch_last_logits()   只取每样本"末尾位置"的 logits（不生成）
        │
        ▼
   score_candidates()       候选→token id→softmax；程序化拼装 JSON
        │
        ▼
   类型化答案 + raw_confidence + raw_mass             （可经 calibrator 校准后路由）
```

## 3. 快速开始

### 3.1 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt     # 推理需 torch + transformers
pip install -e .                    # 本项目（协议/校准部分零依赖）
```

### 3.2 跑测试（不需要模型、不需要 GPU）

```bash
python -m unittest discover -s tests -v
```

### 3.3 起服务（对齐 Jev 的 `/v1/systemone` 形状）

```bash
export JEVP_MODEL=Qwen/Qwen2.5-1.5B-Instruct   # 默认；建议 7B 起步获得可用质量
python -m jevp.server
```

调用：

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

示例请求（来自 Cloudflare 官方 Jev 文档的工单场景）：

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

示例响应（字段与官方 Jev 对齐，额外暴露未校准信号）：

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "answers": {
    "is_urgent":  { "type": "noul",   "noul": 0.61,  "raw_mass": 0.012 },
    "department": { "type": "choice", "choice": "billing", "confidence": 0.82,
                    "probabilities": {"billing": 0.82, "technical": 0.10, "sales": 0.08},
                    "raw_mass": 0.31 },
    "frustration":{ "type": "score",  "score": 1.13,
                    "confidence": 0.72, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.04, "1": 0.79, "2": 0.17}, "raw_mass": 0.28 }
  },
  "usage": { "input_tokens": 312, "output_tokens": 0 }
}
```

### 3.4 单文件 demo（本地模型推理）

```bash
python examples/demo.py                      # 默认 Qwen2.5-1.5B-Instruct，CPU 可跑
JEVP_MODEL=Qwen/Qwen2.5-7B-Instruct python examples/demo.py
```

### 3.5 作为 Agent Skill 使用

`SKILL.md` 定义了"JevP 决策 Skill"（Claude Code 等 agent 可直接引用）：协议、调用方式、最佳实践与安全提示，见项目根目录。

## 4. 包结构与设计

| 模块 | 职责 | 依赖 |
|---|---|---|
| `jevp/protocol.py` | Choice/Score/Noul 数据模型、校验、prompt 构建 | 仅标准库 |
| `jevp/engine.py` | 约束解码：候选 token 打分、softmax、程序化拼装答案 | 仅标准库（tokenizer 鸭子类型注入） |
| `jevp/backend.py` | `TransformersBackend`（本地模型，单次前向取末位 logits） | torch + transformers（惰性导入） |
| `jevp/calibrator.py` | ECE、温度缩放、置信度路由 | 仅标准库 |
| `jevp/server.py` | FastAPI `/v1/systemone`、`/healthz` | fastapi + uvicorn |
| `SKILL.md` | Agent Skill 封装（协议 + 最佳实践 + 安全提示） | — |
| `tests/` | 协议/校准/打分引擎单元测试（可离线运行） | unittest |

## 5. 校准与决策路由（重要）

> **raw logits 不是校准概率。** 这是社区实测反复确认的事实：
> - jev-on-a-laptop 实测 7B 模型在 20 个预测错误的字段里有 13 个置信度 > 0.90；
> - vllm-jev-decison 实测**低 `candidate_mass` 比高 `confidence` 更能暴露失败**（问一个候选标签答不了的问题，confidence 0.9985 而 mass 只有 0.000004）。

因此 JevP 在两个层面做了防御：

1. **输出 `raw_mass`**：候选 token 在完整词表 softmax 中分到的概率之和。值很低 = 模型"没懂这个问题"，即使 confidence 很高也要当心。
2. **`calibrator.py` 提供温度缩放 + ECE 评估**：收集 200–500 条带标注样本，学一个温度系数把概率拉回"0.9 就该九成对"，再用 ECE 量化；最后用 `confidence_router()` 做**选择性分类**：高置信自动执行、低置信转人工。

接入生产前请务必在自己数据上做一次校准验证（见 `examples/calibrate_demo.md` 思路）。

## 6. 与 Jev 的量化差距（诚实清单）

| 维度 | JevP（本地 7B） | Jev 1.13 | 说明 |
|---|---|---|---|
| 类型安全输出 | ✅ 结构保证 | ✅ 结构保证 | 约束解码 vs 专用采样器，殊途同归 |
| 单请求延迟 | 0.4–2.0s（28 字段，M系/Mac） | 0.07–0.5s | 同量级，取决于硬件 |
| 判定质量（官方 373 对基准一致率） | ~73.8% | 86.6% | **差约 13 个点**，实测见 jev-on-a-laptop |
| 概率校准 | ❌ 未校准（需外挂校准器） | ✅ RLCD 训练校准 | 最大的结构性差距 |
| 成本/隐私 | ~$0、数据不出域 | $0.042/M 输入 | JevP 的独特优势 |
| 文本/多模态 | ❌ 仅文本 | ❌ 仅文本 | 一致 |

**结论**：JevP 适合隐私敏感、离线、成本敏感、对"落后 10+ 点判定质量"可容忍的场景（初筛、粗分类、护栏原型）；需要**可信概率做自动决策**或要求 Jev 级质量时，请直接使用官方 Jev 或走训练路线（如 openJev-verdict-2.0）。

## 7. Roadmap

- [ ] vLLM `logprobs` 后端（复用 vllm-jev-decison 的 raw_logprobs 路线）
- [ ] 判别式引擎选项（ModernBERT + GLiClass 布局，openJev 路线，更接近"单前向打分"本质）
- [ ] 中文校准语料与验证
- [ ] 选项顺序扰动（Permutation-KL 的对拍测试，缓解顺序偏置）
- [ ] 模型输出 `raw_mass` 阈值告警的接入示例

## 8. 参考

- TypeSafe Jev 官方文档：https://docs.typesafe.ai / https://typesafe.ai
- Cloudflare Jev 模型页（接口形状参照）：https://developers.cloudflare.com/ai/models/typesafe/jev/
- jev-on-a-laptop（并行约束解码实测）：https://github.com/rorshopping/jev-on-a-laptop
- vllm-jev-decison（logprobs 路线与坑）：https://github.com/siliconkernel/vllm-jev-decison
- openJev-verdict-2.0（训练路线可对标）：https://github.com/Heman10x-NGU/openJev-verdict-2.0

License：待定（初始提交，由仓库所有者决定）。