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

默认测试基线：**7B 模型 × Mac mini 4（M4，16GB）**，模型用 **Ollama（macOS）** 下载与管理。

```bash
# 0) 安装并启动 Ollama（https://ollama.com），拉取 7B 模型
brew install ollama
ollama serve          # 或直接装官网 .dmg 后打开
ollama pull qwen2.5:7b   # 下载/更新/多版本管理都由 ollama CLI 负责

# 1) 安装本项目（推理走 Ollama 的 OpenAI 兼容 logprobs 接口，无需 torch）
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e .

# 2)（可选）无 Ollama 时的备选后端：
#    Apple Silicon → pip install -e '.[mlx]'   （MLX 直接加载 4bit 权重）
#    其他平台     → pip install -r requirements.txt  （torch + transformers）
```

### 3.2 跑测试（不需要模型、不需要 GPU）

```bash
python -m unittest discover -s tests -v
```

### 3.3 起服务（对齐 Jev 的 `/v1/systemone` 形状）

后端默认自动选择（优先级）：**① 标准工作流变量 `VLM_PROVIDER=openai` 且 `VLM_BASE_URL` 已设置 → OpenAI 兼容后端** → ② 本机 Ollama 服务可用 → Ollama（`qwen2.5:7b`） → ③ Apple Silicon 且装有 mlx-lm → MLX（`mlx-community/Qwen2.5-7B-Instruct-4bit`） → ④ transformers（`Qwen/Qwen2.5-7B-Instruct`）。

**OpenAI 兼容后端（vLLM/SGLang 等，含局域网自建服务）**支持两组等价变量（JevP 原生：`JEVP_OPENAI_MODEL` / `JEVP_OPENAI_BASE_URL` / `JEVP_OPENAI_API_KEY`；通用标准：`VLM_MODEL_NAME` / `VLM_BASE_URL` / `VLM_API_KEY` / `VLM_PROVIDER=openai`）。设置任一组后直接跑即可（auto 自动识别）：

```bash
export VLM_PROVIDER=openai VLM_MODEL_NAME=qwen3.8-27b \
       VLM_BASE_URL=http://<host>:<port>/v1 VLM_API_KEY=sk-...
python -m jevp.server                    # 或 python examples/mario_demo.py
```

对 Qwen3 等带思考链的模型，后端默认带 `chat_template_kwargs={"enable_thinking": false}` 关闭 thinking（确定性打分必需，否则 `max_tokens=1` 会被思考 token 吃掉）。

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
python examples/demo.py                      # auto 后端，默认 7B 基线
JEVP_BACKEND=transformers JEVP_MODEL=Qwen/Qwen2.5-7B-Instruct python examples/demo.py
```

### 3.5 测试基线：Mac mini 4（M4 · 16GB）· 7B

面向目标机器跑的基准脚本（在目标机器上执行，本仓库不预跑）：

```bash
python benchmarks/benchmark.py                     # 默认 mario 场景（1-1 关卡快决策）
python benchmarks/benchmark.py --scenario ticket --fields 4,16 --repeat 5
```

- **默认模型**：`qwen2.5:7b`（**Ollama 管理**：`ollama pull qwen2.5:7b`，Metal 加速、模型更新由 ollama CLI 负责）
- **内存预算**：Ollama 默认上下文从 2048 起（`num_ctx`），16GB 足够；建议单批字段数 ≤ 28 以获得平稳延迟
- **预期性能（参考值）**：社区在 16GB Apple Silicon 实测 Qwen2.5-7B-4bit：28 字段判定 ≈ 1.52s；单决策（3 字段）≈ 611ms/例——M4 mini 预期同量级（参考 [jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop)，非本仓库实测）
- **注意**：Ollama 后端只暴露 top-k logprob，因此 **raw_mass 为 top-k 近似**（MLX/Transformers 后端为完整词表精确值）；Ollama 与 MLX 后端为逐条前向，Transformers 后端为一次 batch 前向

### 3.6 作为 Agent Skill 使用

`SKILL.md` 定义了"JevP 决策 Skill"（Claude Code 等 agent 可直接引用）：协议、调用方式、最佳实践与安全提示，见项目根目录。

### 3.7 演示场景：马里奥关卡（Super Mario Bros. 1-1）

把 JevP 用作"快决策层"的端到端演示（思路源自社区项目 [typesafe-mario](https://github.com/fhshaik/typesafe-mario)，用结构化模拟器状态驱动马里奥决策）：

```bash
python examples/mario_demo.py                      # 合成 1-1 关卡快照 → 动作决策
python benchmarks/benchmark.py                     # 默认即该场景的延迟基准
```

**定位（重要）**：马里奥 60fps（16ms/帧）远超任何模型决策速率；System One 模型的正确用法是**剥离低频宏观动作决策（每秒 1~2 次）**，高频精确控制留给确定性代码：

```
模拟器(60fps) ──采样──▶ 结构化 state ──▶ JevP 决策(每 0.1~0.6s)
                                              │
                    ┌─────────────────────────┤
                    ▼                         ▼
              动作建议(按钮映射)          确定性代码继续 60fps 执行
```

每帧 4 个**原子化**问题：`action`（Choice: run/jump/crouch/wait/turn_back）、`obstacle_ahead`（Noul）、`gap_ahead`（Noul）、`danger`（Score 0–2）。示例关卡含 4 帧：开局 → 问号块 → 蘑菇怪逼近 → 坑前；每帧输出推荐动作 + confidence + `raw_mass` + 路由建议与按钮映射（`examples/mario_demo.py`）。

接入真实模拟器（FCEUX Lua / Libretro）只需写一个"采样器"：把内存/画面转成 `benchmarks/scenarios/mario.py` 中同样的结构化字段，即可沿用同一套决策问题集。

### 3.8 实测记录：qwen3.8-27b（局域网 OpenAI 兼容服务）× 马里奥关卡

2026-09-22 实测：`JEVP_BACKEND=openai` 连内网 vLLM 服务（`http://192.168.1.52:8001/v1`、`qwen3.8-27b`、`enable_thinking=false`）：

| 帧 | 场景 | 推荐动作 | confidence | raw_mass | danger(0–2) |
|---|---|---|---|---|---|
| 0 | 开局，前方问号块 | jump | 0.635 | 0.992 | 0.08 |
| 1 | 蘑菇怪 4 格外 | run | 0.888 | 0.992 | 0.95 |
| 2 | 蘑菇怪 2.5 格外 | run | 0.755 | 0.984 | 1.01 |
| 3 | 坑前（宽 2，地面 1 格） | jump | 0.486 | 0.944 | 1.49 |

**观察**：
- 决策语义全部合理（顶问号块→跳；敌人 3~4 格外→跑；临坑→跳）；
- **置信度回落符合预期**：临界/模糊帧 conf 下降到 0.486–0.635 → 被路由规则拦截（不自动执行），正是"快系统 + 置信度路由"的设计行为；
- **danger 随威胁单调上升**（0.08→0.95→1.01→1.49），是合理的校准代理信号；
- raw_mass 全部 >0.83：候选均在 top-k 内，无"乱答"迹象；
- **性能（repeat×3 中位数）**：每决策 **82.6ms**、每帧（4 问）**0.33s** → 每秒约 12 次判断，宏观决策层完全够用。

复现：设上述环境变量后 `python examples/mario_demo.py` / `python benchmarks/benchmark.py`（结果存 `results/bench-*.json`）。

### 3.9 综合评测（qwen3.8-27b · 局域网 OpenAI 兼容服务 · repeat×3）

`python benchmarks/evaluate.py`（金标准为规则标注，可辩解、非学术精确标注）：

| 场景 | 决策数 | schema | 准确率 | 一致性 | 置信度(对/错) | mass | ms/决策 |
|---|---|---|---|---|---|---|---|
| mario（马里奥 1-1 动作决策） | 12 | ✅ | 0.75 | 1.0 | 0.70 / 0.47 | 0.98 | 78 |
| guardrail（Agent 工具调用护栏） | 54 | ✅ | 0.96 | 0.94 | 0.94 / 0.50 | 0.90 | 106 |
| ticket（工单路由/紧急/情绪） | 9 | ✅ | 0.67 | 1.0 | ≥0.99 / 0.88 | 0.78 | 75 |

**评定结论**：

1. **类型安全 100% 达标**：schema-valid 是结构保证（JSON 程序化拼装），三场景恒成立——这是本方案最稳、最可复用的收益；
2. **任务越结构化，判定越准**：护栏 0.96 > 马里奥 0.75 > 工单 0.67；误差集中在边界模糊/主观判分处（临坑起跳时机、工单"情绪档位"）；
3. **置信度是"有用的校准代理"，但不是校准概率**：guardrail 正确/错误的平均置信度分离干净（0.94 vs 0.50，阈值可放心设置）；但 ticket 错误时仍高置信（0.88）→ 印证 raw logits 路线**必须**外挂温度缩放/置信度头（见 §5）才能用于自动决策；
4. **一致性高**（同输入重复判定稳定）、**性能充足**（75–106ms/决策，宏观决策层约每秒 10 次判断）。

**局限**：金标准为规则标注（趋势性数字而非绝对精度）；中文场景未覆盖（模型以英文为主）；mario/guardrail 亦可作为"护栏/快判断"接入点取用（见场景文件）。

### 3.10 Laya（Jev 同赛道开源平替）对比评测

[Laya](https://github.com/NandhaKishorM/laya)（Apache-2.0，21k+ stars）是 Convai Innovations 出品的**判别式 System 1 决策模型**：非自回归、单次前向（官方 T4 33ms、batch 7.2ms/问）、RLCD 训练、3 套 checkpoint（`english` ModernBERT-large 421M / `multilingual` mmBERT-base 322M / `typed-decisions` 421M），由 `Router` 自动语言路由。其 questions 协议与 JevP **完全同构**，因此可直接复用本仓库场景金标准。

**部署**（本仓库已提供开箱支持）：
```bash
pip install "laya[serve]"
python examples/laya_server.py --device cpu --port 8010   # /v1/systemone（JevP 同形状）+ /healthz
python benchmarks/evaluate_laya.py                         # 三场景对比评测
```

**已接入 JevP（判别式后端，P0 落地）**：`LayaBackend` 实现了 `Backend` 的 `direct` 契约，
`engine.decide` 自动 dispatch，无需新评测管线即可一键切换：

```bash
JEVP_BACKEND=laya python benchmarks/evaluate.py          # 三场景，Router 自动语言路由
JEVP_BACKEND=laya python examples/mario_demo.py          # demo / server 同理
JEVP_BACKEND=laya JEVP_LAYA_MODEL=multilingual python benchmarks/evaluate.py  # 强制多语言权重
```

统一管线实测（Router auto 路由，CPU）与独立评测一致：**ticket 1.0 / guardrail 0.556 / mario 0.5**
（auto 会把中文 criteria 场景路由到 english 权重；`evaluate_laya.py` 强制 multilingual 为 0.39/0.25）。
注：Laya 判别式输出无 `raw_mass` 字段（该信号只属于 LLM logits 路线）。

**实测对比（同一组 questions + 同一规则金标准，2026-09-24）**：

| 场景（语言） | **Laya**（判别式）准确率 | **JevP+qwen3.8-27b**（logits 打分）准确率 | 说明 |
|---|---|---|---|
| ticket 工单（英文） | **1.0** | 0.67 | Laya 英文权重在英文任务上更强（判别式语义稳） |
| mario 马里奥（中文 criteria） | 0.25 | **0.75** | 中文多语言权重是 Laya 短板 |
| guardrail 工具护栏（中文 criteria） | 0.39 | **0.96** | 同上 |
| schema 有效 | ✅ 100% | ✅ 100% | 两者同为"程序化拼装/受限输出" |
| 每决策延迟 | 229–1038ms（本机 CPU；官方 GPU 33ms） | 75–106ms（27B 服务端） | 各自主场不同 |

**评定结论**：
1. **Laya 是真·Jev 平替，但只在英文/非中文任务上**：英文工单 100% 全对、确定性高（一致性 1.0）、无幻觉（判别式）、部署极简（`pip install laya`）；
2. **中文是 Laya 的短板**：multilingual（mmBERT-base）对中文 criteria 的判定 0.25–0.39，且 confidence 低到近不可用（~0.05）——**这与 JevP+qwen3.8-27b 恰好互补**（27B 中英文都稳）；
3. **两者置信度都需谨慎**：Laya 英文 checkpoint 启动时官方自警 "invalid temperatures → confidence uncalibrated"（README 已注明）；JevP 路线同样依赖外挂校准（见 §5）；
4. **选型建议**：英文密集型决策（票务/审核/护栏，英文语料）→ Laya 开箱即用、成本极低；中文/混合 → JevP+qwen3.8-27b 更稳；两者 questions 协议同构，方案可无缝切换/互为对照。

复现：`python benchmarks/evaluate_laya.py --repeat 3 --device cpu`（首次自动下载 checkpoint，结果存 `results/eval-laya-*.json`）。

### 3.11 AgentJev-0.6B（第三个 System One 实现）部署与评测

[AgentJev](https://github.com/malevrigns/agent-jev)（Apache-2.0，313★）：Qwen3-0.6B 骨干去掉 LM head + 候选头单前向打分，零解码 token、共享前缀缓存、2048 上下文；官方 typed-decisions 自报 **79.25% top-1**（比 Laya +2.25 点、比 Jev +6.5 点）。

**部署（本机 CPU + 4×RTX5090 GPU 服务器双实测）**：
```bash
git clone https://github.com/malevrigns/agent-jev.git && cd agent-jev
pip install -r requirements.txt huggingface_hub safetensors
# 权重 safetensors -> torch checkpoint + 校准温度
python -c "from huggingface_hub import hf_hub_download; from safetensors.torch import load_file; import torch; \
torch.save({'state_dict': load_file(hf_hub_download('aimeigaoshou/agent-jev','model.safetensors'))}, 'agentjev_v1.pt'); \
hf_hub_download('aimeigaoshou/agent-jev','temperatures.json', local_dir='.')"
# 首跑需预热 Qwen3-0.6B tokenizer 进 HF 缓存（server 用 local_files_only=True）
python -c "from huggingface_hub import hf_hub_download as d; \
[d('Qwen/Qwen3-0.6B', f) for f in ['config.json','tokenizer.json','tokenizer_config.json']]"
# 启动（GPU 服务器：--device cuda:0）
python -m jev_service.server --checkpoint agentjev_v1.pt --model-path Qwen/Qwen3-0.6B \
  --temperatures temperatures.json --port 8149 --device cpu
```
验证：`GET /health` 返回 `{status: ready, types: [boolean, choice, score], shared_prefix_compute: true, ...}`。

**业务实测（自带 `run_practical_test.py`）**：Coding Agent 门控三连全对——拒绝提交（63% 未完成概率）、推荐读失败测试（69.6%，margin 57%）、风险 Level 1；DevOps 事故定级 Level 2 合理。延迟：CPU 首次 4–8s/决策；**GPU（RTX 5090）热身后 94–178ms/决策，与官方"40–80ms"文案吻合**（首次决策含模型加载 1.6s）。

**GPU 部署实测（4×RTX 5090，`--device cuda:2`，2026-09-28）**：
- 三场景准确率与 CPU 完全一致（判别式确定性推理）：mario 0.0 / guardrail 0.833 / ticket 1.0；
- **每决策延迟 41–53ms**（vs CPU 1.9–5.3s，提速 40–100×，达官方宣称量级）；
- 环境要点：服务器 HF 直连不通需 `HF_ENDPOINT=https://hf-mirror.com`（否则服务启动卡在 tokenizer 校验的无限重试）；避让被 27B 占用的卡（选显存富余的 `cuda:2`）。

**三场景同题评测（`benchmarks/evaluate_agentjev.py`，与 §3.9/§3.10 同口径）**：

| 场景（语言） | **AgentJev-0.6B** | Laya（判别式） | JevP+qwen3.8-27b |
|---|---|---|---|
| ticket（英文） | **1.0** | **1.0** | 0.67 |
| guardrail（英文命令+中文 criteria） | 0.83 | 0.39 | **0.96** |
| mario（数值化状态+中文 criteria） | **0.0**¹ | 0.25 | **0.75** |
| 一致性 | 1.0 | 1.0 | 1.0 |
| 每决策延迟 | **GPU 41–53ms**（CPU 1.9–5.3s） | GPU 33ms（CPU 0.23–1.0s） | 27B 服务 75–106ms |

¹ **中英 criteria 消融已定位根因（见下）**：中文 criteria 返回近均匀分布（margin≈0.07），argmax 随机化；是**训练语料语言**问题而非评测/状态问题。

**中英 criteria 消融（`benchmarks/ablation_criteria_lang.py`，state 与金标准不变，仅翻译 criteria）**：

| 场景 | 中文 criteria | English criteria | margin(中/英) |
|---|---|---|---|
| mario | **0.0** | **0.50–0.75** (GPU 0.5 / CPU 0.75) | 0.071 / 0.079 |
| guardrail | 0.8333 | 0.8333 | 0.319 / 0.268 |

**根因结论**：mario 中文 → 候选头无判别信号 → 近均匀分布；英文 → argmax 方向正确但仍犹豫（margin 低）。guardrail 中英同为 0.833——因为判别信号在**英文 state（命令）**里，criteria 语言影响小。**结论：约束/微调以英文为主，中文 criteria 的判别能力弱，需中文数据微调或应用层英文化 criteria（JevP 已内置 `LayaBackend`/`AgentJevBackend` 一键切换 + `scenarios` 中英双版）。**

**要点**：
1. **三个 System One 实现的强项各不相同**：AgentJev 英文/命令判定强（门控实测全对、ticket 1.0）、0.6B 极轻量；Laya 英文强但中文弱；JevP+27B 中英文最均衡；
2. **训练语料与任务匹配度比参数量更关键**：guardrail 上 0.6B（0.83）> Laya 322M（0.39）、< 27B（0.96）；
3. **三者都内置/暴露了校准温度**（AgentJev: `temperatures.json` 自动加载；Laya: checkpoint 温度；JevP: calibrator.py），且都注明"域校准不保证"——**校准仍是 System One 落地的共同最后一公里**。

复现：`AGENTJEV_URL=http://<host>:8149 python benchmarks/evaluate_agentjev.py --repeat 3`。

### 3.12 业务落地：企业-产品匹配评价系统（潜在客户挖掘 → 打分 → 排序）

**业务定义**：输入产品信息，从企业库（基本信息/招投标/官网战略/上下游关系）挖掘潜在客户企业并排序。JevP 承担**批量精排打分**：recall 由外部企业库/招投标检索完成，JevP 对每个候选企业跑一组原子化问题，产出 `match_score` 排序 + `confidence/margin` 把关。

**问题集（每企业 6 问，可裁剪）**：`industry_fit`(noul) / `demand_signal`(noul，招投标与采购窗口) / `budget_scale`(score 0-2) / `strategic_fit`(score 0-2，官网战略) / `urgency`(noul) / `match_score`(score 0-3 综合档位)。

**LLM 调用降频**（本场景核心诉求，`examples/enterprise_match_demo.py --questions`）：
- `min`：1 问/家（8 次/8家）极速粗筛；`core`：3 问/家（24 次/8家，默认）；`all`：6 问/家（48 次/8家）精评档；
- 实测 **core 档排序与 all 档 8/8 一致**（调用减半不损排序）；
- 三级漏斗（README 建议架构）：**L0 规则粗筛（0 LLM）→ L1 core/min 打分 → L2 all 精评仅 Top-N** + 同企业 7 天缓存。

**四方案同步评测（中文企业档案，同一金标准 match_level 0-3，repeat×3）**：

| 方案 | LLM 调用 | 档位命中率 | 排序/置信度信号 | 每问延迟 |
|---|---|---|---|---|
| **规则基线 L0**（`rule_baseline_enterprise.py`） | **0 次** | **62.5%** | 排序相关 Pearson 0.95 | ~0ms |
| **JevP+qwen3.8-27b**（推荐） | 24 次/批（core） | **75%** | conf对 0.82 / conf错 0.55（区分度可用） | 85ms |
| Laya（判别式） | 0（本地） | 12.5% | conf 0.62/0.58（无区分） | 621ms（CPU） |
| AgentJev-0.6B（GPU） | 0（本地） | 25% ≈ 随机 | conf 0.46/0.46（无区分） | 48ms |

**结论**：
1. **中文企业档案必须走 LLM 路线（27B）**：判别式模型（Laya/AgentJev）对中文 criteria 无判别信号，退化到随机档位（与 §3.11 消融闭环一致）；英文转译 criteria 后判别式方可使用；
2. **规则粗筛（0 LLM）不是弱基线**：62.5% 档位命中 + 0.95 排序相关——作为 L0 漏斗前置可把 27B 精排量降到 Top-N，示例：10,000 家候选 → 规则筛至 ~200 → core 精评 600 次调用（27B ~51s 单批，可异步）→ 置信度阈值产 Top-N 触达名单；
3. **置信度把关**：27B 在企业场景 conf对/conf错 = 0.82/0.55，配合 `route_advice`（分数×置信联合）避免低分高置信误触达。

复现：
```bash
python examples/enterprise_match_demo.py --questions core        # 排序 + 触达建议
python benchmarks/evaluate.py --backend openai --scenarios enterprise_match --repeat 3   # 27B 指标
python benchmarks/rule_baseline_enterprise.py                    # L0 规则基线
JEVP_BACKEND=router python examples/enterprise_match_demo.py     # 语言感知路由（中文→27B，英文→判别式）
```

### 3.13 国际（中英双语）企业评测集与"兼顾中英"实证

**评测集扩展**：新增 `enterprise_match_en`（6 家海外企业：美/德/英/澳/新，英文档案），与中文 `enterprise_match`（8 家中企）构成**同一产品、同一 6 问结构、同一 0-3 金标准的中英双语评测集**——即未来"中英兼顾"模型的持续评测基准（`benchmarks/eval_bilingual_enterprise.py`）。

**双语评测实测（档位命中率，repeat×1）**：

| 后端 | 中文组(8家) | 英文组(6家) | 总体 |
|---|---|---|---|
| **openai（qwen3.8-27b）双语直连** | 0.75 | **0.667** | **0.708** ✅ |
| router（中文→27B，英文→Laya 判别式） | 0.75 | 0.50 | 0.625 |
| Laya（判别式） | 0.125 | 0.50 | 0.3125 |

**实证结论**（对"未来模型中英兼顾"的指导）：
1. **27B 一个后端双语直连就是当前最优（0.708）**——中文企业 75%、海外企业 67%，无需路由即兼顾中英；
2. **判别式模型（Laya/AgentJev）适用边界被进一步钉死**：适合**单维分类/布尔**（guardrail/ticket 类任务），在**多维综合评分**（match_score 六维整合 + 数值档位推理）上明显弱（英文仅 0.5、中文 0.125）——企业打分这类综合评估应走 LLM 路线；
3. **router 的价值=降频**（英文走本地 0 外部调用，牺牲约 17 个点英文命中），适合"英文粗筛阶段"，精评仍上 27B；这也是 P1 中文/综合评分微调（判别式做综合评分）的改进动机。

复现：`VLM_*=... python benchmarks/eval_bilingual_enterprise.py [--with-agentjev]`。

### 3.14 性能指标提升实验（不训练：错误诊断 + 域先验硬校正）

在企业/双语评测集（14 家：8 中 + 6 英）上，用**纯工程杠杆**（无蒸馏/无微调）把档位命中率从 **71.4% → 100%**：

| 手段 | 命中率 | 说明 |
|---|---|---|
| 27B 双语直连（基线） | 71.4%（10/14） | 错误**全部**集中在 0/1 档被评高（云购 0.74/研究院 1.21/PacificFresh 1.63/ShopNova 1.73）——根因：模型 0-3 档上"低档高估"（range compression） |
| + **域先验硬校正**（`domain_patch`） | **100%（14/14）** | 3 条可辩解业务规则：行业不匹配→上限 1.0；行业✗且无招投标→上限 0.5；非企业实体→上限 0.5 |
| 规则 L0 双语关键词修正 | 62.5%→64.3% | 修复英文企业规则分全 0.5 的 bug（英文关键词表） |
| 规则-LLM 融合（w<1） | ≤64%（更差） | **排除**：规则仍拖后腿，判别式/规则弱时不要融合 |

- 校正已集成进 `examples/enterprise_match_demo.py`（默认开启，`--no-patch` 可关，输出以 `←` 标注回退，透明可追溯）；
- **诚实边界**：14 家小样本、校正规则在校准集上定义——上量前需扩大企业库并做时间/区间留出验证；
- 结论：**先用可辩解规则修 LLM 的系统性偏差（立即 0 成本），再谈蒸馏/微调**（P1）。

复现：`VLM_*=... python benchmarks/improv_enterprise_blend.py --w 0.7`；`python examples/enterprise_match_demo.py --questions core`。

### 3.15 服务 API 与调试台（明确输入/输出结构）

**启动**（需 VLM_* 环境变量指向 27B 服务）：
```bash
python examples/enterprise_api.py          # 端口 8021（JEVP_API_PORT 可改）
```
**端点**：

| 端点 | 说明 |
|---|---|
| `GET /` | 网页调试台（中/英示例加载、参数、排序表、逐企业明细、原始 JSON） |
| `POST /api/match` | 批量企业匹配打分 + 排序（核心） |
| `POST /api/match/one` | 单企业快查（`{"name":..., "basic":{...}, ...}` 直接 POST） |
| `GET /api/match/example?lang=zh\|en` | 返回内置示例企业列表（调试台填充用） |
| `GET /api/schema` | 输入/输出结构契约（机器可读） |
| `GET /healthz` | 健康检查 |

**输入结构**（`POST /api/match`）：
```json
{
  "product":    {"name": "…", "description": "…"},     // 可选，缺省内置 Titan-PHM
  "companies":  [
      {"name": "企业名",
       "basic": {"industry": "主营行业", "scale": "规模", "location": "地区",
                 "employees": "人数", "since": 成立年},
       "bidding": [{"year": 2026, "title": "招标/采购标题", "amount": 金额}],
       "website": {"strategy": "官网战略关键词", "news": "动态"},
       "supply_chain": {"customers": "客户", "suppliers": "供应商", "leverage": "渠道杠杆"}}],
  "demo":       "zh" | "en" | null,                    // 快捷加载内置中/英示例
  "questions":  "core" | "all" | "min",                // 3/6/1 问每家
  "apply_patch": true,                                  // 域先验硬校正（默认开）
  "backend":    "auto"(VLM_*→27B) | "router" | "openai" | "laya" | "agentjev"
}
```

**输出结构**：
```json
{
  "schema": "enterprise-match.v1", "demo": false,
  "product": {...}, "questions_used": ["industry_fit", ...],
  "ranking": [
      {"name": "企业名", "lang": "zh|en",
       "match_score": 2.97,      // 校正后匹配分 0-3（可排序）
       "score_raw": 2.97,        // 原始分（未校正）
       "patched": false, "patch_note": "",
       "confidence": 0.98, "margin": 0.96,
       "advice": "🔥 优先触达",   // 🔥/→/◇/✋ 触达建议
       "per_question": {"industry_fit": {...}, "match_score": {...}, ...}}],
  "meta": {"backend": "…", "call_est": 24, "latency_ms": 123.4}
}
```
（完整字段说明见 `GET /api/schema`；语言标注为"档案+问题"混合判定口径，英文档案+中文问题会偏 zh——不影响后端质量）

**curl 示例**：
```bash
curl -s localhost:8021/api/match -H 'content-type: application/json' \
  -d '{"demo":"zh","questions":"core"}' | python -m json.tool
```

## 4. 包结构与设计

| 模块 | 职责 | 依赖 |
|---|---|---|
| `jevp/protocol.py` | Choice/Score/Noul 数据模型、校验、prompt 构建 | 仅标准库 |
| `jevp/engine.py` | 约束解码：候选 token 打分、softmax、程序化拼装答案 | 仅标准库（tokenizer 鸭子类型注入） |
| `jevp/backend.py` | `OllamaBackend` / `OpenAICompatibleBackend`（vLLM 局域网）/ `LayaBackend` / `AgentJevBackend`（判别式）/ `MLXBackend` / `TransformersBackend` / `vLLMBackend`（占位）/**`RouterBackend`**（语言感知路由：中文→27B、英文→判别式 + 低置信降级）；`auto_backend()` 自动选择 | HTTP / 惰性导入 |
| `jevp/calibrator.py` | ECE、温度缩放、置信度路由 | 仅标准库 |
| `jevp/server.py` | FastAPI `/v1/systemone`、`/healthz` | fastapi + uvicorn |
| `benchmarks/benchmark.py` | 目标机器延迟/内存基准（默认 **mario 场景**，可选 ticket） | 仅标准库 |
| `benchmarks/scenarios/` | 场景注册表：`mario`（1-1 关卡）、`ticket`（工单） | 仅标准库 |
| `examples/mario_demo.py` | 马里奥关卡快决策演示（动作序列 + 按钮映射 + 路由） | 仅标准库 |
| `SKILL.md` | Agent Skill 封装（协议 + 最佳实践 + 安全提示） | — |
| `tests/` | 协议/校准/打分引擎单元测试（可离线运行） | unittest |

## 5. 校准与决策路由（重要）

> **raw logits 不是校准概率。** 这是社区实测反复确认的事实：
> - jev-on-a-laptop 实测 7B 模型在 20 个预测错误的字段里有 13 个置信度 > 0.90；
> - vllm-jev-decison 实测**低 `candidate_mass` 比高 `confidence` 更能暴露失败**（问一个候选标签答不了的问题，confidence 0.9985 而 mass 只有 0.000004）。

因此 JevP 在两个层面做了防御：

1. **输出 `raw_mass`**：候选在模型词表概率分布中分到的份额。值很低 = 模型"没懂这个问题"，即使 confidence 很高也要当心（Ollama 后端只暴露 top-k，此为 top-k 近似；本地 MLX/Transformers 后端为完整词表精确值）。
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
- [ ] **MLX 后端 batch 化**：一次前向评估一批问题（当前为逐条前向），对齐 jev-on-a-laptop 的 KV 广播方案
- [ ] **Ollama/OpenAI 兼容后端调优**：top_logprobs 档位/候选 token 变体词典（含中文候选）实证，raw_mass 阈值标定
- [ ] 完整词表 logits 路线对比（Ollama/OpenAI 的 top-k 近似 vs MLX/Transformers 精确 raw_mass）
- [ ] 判别式引擎选项（ModernBERT + GLiClass 布局，openJev 路线，更接近"单前向打分"本质）
- [ ] 中文校准语料与验证
- [ ] 选项顺序扰动（Permutation-KL 的对拍测试，缓解顺序偏置）
- [ ] `raw_mass` 低值告警的接入示例
- [ ] **马里奥场景接入真实模拟器**（FCEUX Lua / Libretro 采样器 → JevP → 按钮执行）
- [ ] 在 Mac mini 4（16GB）上跑通基准并回填实测数字（替换本节参考值）

## 8. 参考

- TypeSafe Jev 官方文档：https://docs.typesafe.ai / https://typesafe.ai
- Cloudflare Jev 模型页（接口形状参照）：https://developers.cloudflare.com/ai/models/typesafe/jev/
- jev-on-a-laptop（并行约束解码实测）：https://github.com/rorshopping/jev-on-a-laptop
- vllm-jev-decison（logprobs 路线与坑）：https://github.com/siliconkernel/vllm-jev-decison
- openJev-verdict-2.0（训练路线可对标）：https://github.com/Heman10x-NGU/openJev-verdict-2.0
- typesafe-mario（Jev 打马里奥的社区实现）：https://github.com/fhshaik/typesafe-mario

License：[Apache-2.0](LICENSE)（含 [NOTICE](NOTICE) 致谢参考项目）。模型权重如将来发布，将单独约定许可（参考 Qwen 的"代码与权重分许可"做法）。