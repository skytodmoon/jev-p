# 企业匹配 v2：批处理、候选筛选和增量评分

v2 将产品、评价日期和企业档案一起传入模型，替代 API 原来的固定 PHM 场景问题。
旧的 `product/companies/questions/backend/apply_patch` 请求字段继续可用；响应 schema 改为
`enterprise-match.v2`。直接运行 benchmarks 的旧场景保持原有问题，历史评测与 v2 分开比较。

## 选择模式

| 参数 | 行为 | 覆盖含义 |
|---|---|---|
| `mode=full`，默认 | 对全部企业做 JEV 评价，分批处理并复用缓存 | 返回全部企业，包括失败行 |
| `mode=topk` | 混合召回、加权排序后，只对候选做 JEV 精评 | 返回候选中的 Top-K，可能遗漏其他好客户 |

`topk` 默认 `recall_limit=5000`、`candidate_limit=500`、`top_k=100`。
行业、企业正文、采购公告、新闻分别进行 BM25 检索，中文使用相邻双字词，英文使用词项。
采购证据通道权重为 2，其他通道为 1，通过 `weight/(60+rank)` 融合并截取召回池和候选。
战略中的同类产品词不会被直接当作采购证据。默认约 10% 精评预算保留给可复现的随机探索，
避免完全排除资料不完整的企业；探索会占用精评名额，仍不提供召回保证。

这些权重是启发式，需要用实际业务标签调优。检索相似度不作为采购概率或 JEV 分数。
核心管线不生成向量，也没有训练新的小模型或多任务评分头。

### 可选向量

产品和企业可携带 `embedding: [0.1, ...]`，必须来自**同一个嵌入模型及版本**，
维度一致、有限且非零。索引会归一化向量，以余弦相似度增加一个召回通道。
有 NumPy 时使用矩阵乘法，否则使用标准库；安装加速依赖：`pip install '.[retrieval]'`。
未提供企业向量的档案仍能参加词项召回和探索。向量不会传入 JEV 提示词。

## API 用法

使用现有自建推理服务启动；无需更换模型：

```bash
export JEVP_HTTP_CONCURRENCY=8
export JEVP_MODEL_REVISION=deployment-2026-10-20
uvicorn examples.enterprise_api:app --host 127.0.0.1 --port 8021 --workers 1
```

先准备请求文件 `companies.json`：`{"companies": [企业档案, ...]}`，然后构建一次索引：

```bash
curl -X POST http://127.0.0.1:8021/api/companies/index \
  -H 'Content-Type: application/json' --data-binary @companies.json
```

返回 `index_id` 后，不需要在后续每次产品查询时上传和预处理所有企业：

```json
{
  "product": {
    "name": "工业设备预测性维护平台",
    "description": "面向钢铁制造企业，降低设备非计划停机损失",
    "target_industries": ["钢铁", "制造"],
    "commercial_only": true
  },
  "index_id": "使用建索引接口返回的值",
  "mode": "topk",
  "recall_limit": 5000,
  "candidate_limit": 500,
  "top_k": 100,
  "questions": "core",
  "batch_size": 32,
  "as_of": "2026-10-20",
  "backend": "auto"
}
```

保存为 `match.json`。小批量可直接提交 `POST /api/match`，大批量使用：

```bash
curl -X POST http://127.0.0.1:8021/api/match/jobs \
  -H 'Content-Type: application/json' --data-binary @match.json
curl http://127.0.0.1:8021/api/match/jobs/JOB_ID
curl 'http://127.0.0.1:8021/api/match/jobs/JOB_ID/results?offset=0&limit=100'
```

`full` 模式保持每家完整评分；不要将未进入候选集的企业标为 0 分。
`GET /api/schema` 提供当前参数约束。企业来源 `companies/index_id/demo` 三选一。
`demo` 可以与自定义 `product` 组合。`POST /api/match/one` 同时支持旧的原始企业对象和
新的 `{"company": {...}, "product": {...}}`。

## 推理、缓存和错误

- `DecisionEngine.decide_many` 跨企业展开问题，按 `batch_size`（提示词数，1..512）和
  `max_batch_chars`（默认 64,000 字符）分批；字符预算不等于 tokenizer token 数或显存预算。
  超过字符预算的单条提示词报错，不由这个接口静默裁剪。MLX/Transformers 同时检查真实
  token 长度并拒绝超限；HTTP/direct 服务的上下文上限仍需在其服务端正确配置。
- HTTP 后端通过共享信号量限制并发；并发调用同一后端的作业也共享此上限。
  客户端并发让服务端有机会合批，不等同于一次神经网络前向。
  Transformers 可以跨企业做真实 batch；MLX、无批量协议的 direct 后端保留串行回退。
- 单条 HTTP 故障只影响对应企业；错误、无评分 token 证据或非法分数不进入缓存。
  响应 `errors` 单独列出失败项，`topk` 不会因截取结果而隐藏全部错误信息。
- SQLite 缓存默认容量 100,000，TTL 为 3,600 秒，目录为 `.cache/matching`，已加入忽略规则。
  可用 `JEVP_CACHE_DIR/JEVP_MATCH_CACHE_PATH/JEVP_CACHE_ENTRIES/JEVP_CACHE_TTL` 调整。
  键包含产品内容、企业档案内容、问题与规则版本、模型标识和配置、模型 revision、评价日期。
  同一服务地址和模型名背后替换权重时，必须更新 `JEVP_MODEL_REVISION` 或请求的 `model_revision`。
- `use_cache=false` 可进行冷评测。企业资料变化只重评变化项，失败后重提相同任务可复用成功结果。
  缓存复用以相同评分契约为前提，不会消除模型自身的随机性或推理误差。
- `min + apply_patch=true` 自动保留 `industry_fit`，实际为 2 问；关闭校正才是 1 问。
  仅在模型判断不符合该产品目标客户时封顶 1 分；高校/研究机构只在产品显式设定
  `commercial_only=true` 时按约束封顶。没有采购记录不再被直接判为负面事实。

### 响应中需要检查的字段

| 字段 | 解释 |
|---|---|
| `meta.total_companies/candidate_count` | 全库数量 / 本次参与评分数量 |
| `meta.ranking_scope` | `all_companies` 或 `retrieved_candidates` |
| `meta.all_companies_scored` | 全部企业成功获得分数才为 true |
| `meta.evaluated_companies/cache_hits` | 新评估数量 / 命中评分缓存数量 |
| `meta.call_est` | 兼容旧字段：新评估的“问题数”估计，并非 HTTP 请求或 GPU 前向实测次数 |
| `meta.failed_companies` | 失败数量 |
| `ranking[].input_index/company_id` | 输入位置和业务 ID；企业重名时仍能对齐 |
| `ranking[].score_source/cached` | JEV 来源及缓存状态 |

`score_pct` 为 0..3 量表的线性展示，不是匹配成功率；`confidence` 也未校准。

## 运行与容量边界

索引句柄和作业状态属于**单进程**。使用一个 worker，或者由网关按句柄固定路由。
默认最多 2 个索引、合计 200,000 份档案，索引从创建起保留 2 小时；
索引过期、容量淘汰或进程重启后返回 404，需重建。已排队任务持有自身的索引快照。
生产中的频繁全库更新、多进程共享应接持久企业库/检索服务。

异步任务默认 1 个执行线程、最多 4 个未完成任务、保留最多 16 个任务。
队列满返回 429；任务结果写 SQLite，分页上限 1,000 行。
完成任务被淘汰或进程重启后句柄失效。分页头最多附带 100 条错误，`error_count` 和
`errors_truncated` 会说明是否截断。全量任务仍需存储、排序和返回 O(N) 份结果。
首次建索引、缓存填充和数据上传开销不能从端到端延迟中忽略。

## 测试与压测

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python benchmarks/benchmark_matching.py --sizes 100,1000,10000,100000 --repeat 20
```

规模脚本使用明确标记的**离线合成数据和确定性 logprob fixture**，不访问模型服务。
报告包含建索引时间、全量耗时、复用索引的 topk 耗时、缓存命中、单企业增量失效、
与逐企业评分的一致性，以及合成标签的 Top-100 重合率和 NDCG。
脚本对一致性、缓存、增量失效和受控数据召回设置断言，失败时退出非零。
P95 只是指定重复次数下的样本分位数，不能代替生产延迟 SLO。

HTTP 测试使用本机回环服务，实际经过 HTTP 发送和解析，验证全局并发上限、结果顺序、
故障隔离；这些测试仍然不是实际模型测试。
实测记录见 [离线测试报告](MATCHING_BENCHMARK.md)。真实质量验收应使用人工标注、
跨行业和时间留出的企业数据，检查 Recall@100、NDCG@100、错误率、校准误差与 P95 延迟。
