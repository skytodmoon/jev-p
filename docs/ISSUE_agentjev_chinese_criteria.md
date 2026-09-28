# Issue 草稿：AgentJev-0.6B 对中文 criteria 返回近均匀分布（判别信号丢失）

> 给 https://github.com/malevrigns/agent-jev 上游的 issue 草稿。
> 提交方式：打开上方链接 → New issue → 粘贴本文（可整段贴入，含英文标题）。
> 若你在 GitHub 登录后不方便，也可复制到评论区。

---

**Title（英文 / 可选中文标题）**：
`[zh-cn] AgentJev returns near-uniform distributions for Chinese criteria (no discriminative signal) — ablation data inside`

---

## Summary

AgentJev-0.6B produces **near-uniform choice distributions (margin ≈ 0.07)** and hence
random argmax when the `criteria`/`instructions` are written in **Chinese**, even when
the `state` is structured English/JSON. The same task with **English-translated criteria**
argmaxes to the correct action family (accuracy 0.5–0.75 vs 0.0), while `guardrail`
(same Chinese criteria, but *state* carries a strong English signal) stays at 0.83
regardless of criteria language. This points to **training-data language** as the cause,
not the numeric/JSON state itself.

## Environment

- `malevrigns/agent-jev` main branch (clone 2026-09-28), AgentJev-0.6B checkpoint
  `aimeigaoshou/agent-jev` + `temperatures.json`
- Server: `python -m jev_service.server --device cpu` and `--device cuda:2` (RTX 5090);
  results identical across CPU/GPU (deterministic discriminative inference)
- Client: `agentjev_client.AgentJev` (`decide_choice`)

## Reproduction (minimal)

State (Mario JSON frame, English keys/values):

```json
{"frame": 0, "mario": {"tile_x": 14.0, "tile_y": 12.0, "vx": 1.8, "vy": 0.0,
 "status": "small", "on_ground": true},
 "terrain_scan": {"ground_ahead": 6, "gap_ahead": null,
                  "block_ahead": {"tile_x": 16, "type": "question_block"}, "stairs_ahead": null},
 "enemies": [], "pickups": [{"type": "coin", "tile_x": 16, "tile_y": 9}]}
```

`decide_choice(state, "马里奥下一步应该采取哪个宏观动作？", {
  "run": "保持/加速向前奔跑",
  "jump": "起跳：越过敌人、坑或顶出问号块",
  "crouch": "蹲下：闪避或安全通过低矮障碍",
  "wait": "原地等待：前方暂时安全，无需动作",
  "turn_back": "回头/减速：前进风险过高"})`

Observed (repeat×3, deterministic): distribution ≈ `{run:0.29, jump:0.24, crouch:0.25, wait:0.15, turn_back:0.08}`, **margin ≈ 0.045–0.07**; argmax lands on `run` although the play-correct answer for frame-0 (question block ahead) is `jump`.

## Ablation (criteria 中文 vs English; same state, same golden labels)

| Scenario | criteria 中文 | criteria English | margin 中/英 |
|---|---|---|---|
| mario (numeric JSON state, English keys) | **0.0** (12/12 wrong) | **0.50** (GPU) / **0.75** (CPU, repeat×3) | 0.071 / 0.079 |
| guardrail (English bash-command state) | 0.833 | 0.833 | 0.319 / 0.268 |

- `margin` = top1 − top2 of the choice distribution.
- mario: Chinese → uniform-ish (margin 0.07) → random argmax (0.0); English → argmax
  lands on the correct action-family (jump/run) but stays low-confidence (margin 0.08).
  (Repeat-size difference 0.5 vs 0.75 is a boundary-flip on 12 decisions; both ≫ 0.0.)
- guardrail: signal lives in the **English state** (commands like `rm -rf /home/dev`,
  `curl … -d @.env`); criteria language has almost no effect (0.833 both).
- Score/Boolean questions follow the same pattern (uniform-ish under Chinese criteria).

## Interpretation

The typed-decisions fine-tune is English-only; Chinese criteria pass through the
candidate head with **no discriminative gradient** (near-uniform). It is **not** the
numeric/JSON state (English criteria on the same state recovers direction), nor
ill-posed questions (English side is clearly better). This limits Chinese-production
users unless criteria are translated.

## Suggested fixes (any)

1. **Docs**: state the English-only limitation for `criteria`/`instructions` on the
   README (many Chinese users will hit this blind).
2. **Application-layer**: translate criteria to English before scoring (we did this —
   it works, see ablation).
3. **Training**: mix Chinese (and other languages) options/criteria into the
   typed-decisions fine-tune; also consider bilingual `question`/`criteria` during data
   augmentation.
4. Optionally expose a `lang` hint to the server (like Laya's router) so users can
   force English scoring for Chinese-authored criteria.

## Context / provenance

Tested as part of cross-comparing System One engines (JevP benchmark suite:
`benchmarks/evaluate_agentjev.py` and `benchmarks/ablation_criteria_lang.py`,
Apache-2.0, https://gitee.com/QD0103/jev-p ). Golden labels are rule-based and
defensible; full per-frame logs available on request.

---
*(中国) 中文摘要：`criteria` 用中文时，AgentJev 对 choice 返回接近均匀的分布（margin≈0.07，
argmax 随机化 → mario 0.0）。同一 state 用英文 criteria 即恢复方向（0.5–0.75）；而判别信号
本就在英文 state 的 guardrail（0.833）不受 criteria 语言影响。根因=微调语料为纯英文。
建议：README 标注限制 / 应用层英文化 criteria / 微调混合中文数据。*