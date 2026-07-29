# Noemb 变体对比汇总：bs1 vs bs4 vs bs4_lr1e-5

> 生成时间: 2026-08-03
> 基于日志原始数据逐条核对

---

## 1. 实验说明

### 1.1 三模型权重

| 标签 | 模型路径 | 说明 |
|------|----------|------|
| **noemb_bs1 (bs1, step500)** | `.../3emb_model_bs1/search_noemb_bs1/global_step_500/merged` | 对照基准, bs1 训练, step500 |
| **noemb_bs4 (bs4, step125)** | `.../3emb_model_bs2/search_no_emb_bs4/global_step_125/merged` | bs4 训练, step125 |
| **noemb_lr1e-5 (bs4, lr=1e-5, step125)** | `.../3emb_model_bs4/search_no_emb_bs4_lr1e-5/global_step_125/merged` | bs4 训练, lr=1e-5, step125 |

### 1.2 数据来源目录

| 模型 | Seed | 来源目录 |
|------|:----:|----------|
| **noemb_bs1** | 1 | `coldstart_test_search_seed1_rl_action_modified_seed1/test_noemb/` |
|  | 11 | `coldstart_test_search_seed1_rl_action_modified_seed11/test_noemb/` |
|  | 21 | `coldstart_test_search_seed1_rl_action_modified_seed21/test_noemb/` |
| **noemb_bs4** | 1 | `coldstart_noemb_bs4_test/test_noemb_bs4_seed1/` (_1 后缀) |
|  | 11 | `coldstart_noemb_bs4_test/test_noemb_bs4_seed11/` |
|  | 21 | `coldstart_noemb_bs4_test/test_noemb_bs4_seed21/` |
| **noemb_lr1e-5** | 1 | `coldstart_test_search_lr1e-5/test_noemb_seed1/` |
|  | 11 | `coldstart_test_search_lr1e-5/test_noemb_seed11/` |
|  | 21 | `coldstart_test_search_lr1e-5/test_noemb_seed21/` |

> **重要说明**:
> - `coldstart_noemb_bs4_test/test_noemb_bs4_seed1/` 目录中无后缀的 `.log` 文件来自旧实验 (rl_action/noemb-1, bs2 模型), 被错误复制至此。**本文档以 `_1` 后缀的日志为准** (seed=1, bs4 模型)。
> - `noemb_lr1e-5` 三 seed 均使用同一模型, 脚本 -1 和 -2 为数据拆分并行运行, 输出到同一目录。

### 1.3 实验设置

- 测试集: 7 个 datasource, 共计 1325 样本 (各 200 样本, bamboogle 125 样本)
- 推理 seed: 1 / 11 / 21
- 均为 noemb 变体, 无 embedding 训练

---

## 2. 逐 Seed 逐数据集成功率

### 2.1 Seed=1

| 数据集 | n | noemb_bs1 (bs1, step500) | noemb_bs4 (bs4, step125) | noemb_lr1e-5 (bs4, lr=1e-5) |
|--------|-----:|:------------------------:|:------------------------:|:----------------------------:|
| 2wikimultihopqa | 200 | 68/200 = **34.00%** | 75/200 = **37.50%** | 71/200 = **35.50%** |
| bamboogle | 125 | 35/125 = **28.00%** | 32/125 = **25.60%** | 31/125 = **24.80%** |
| hotpotqa | 200 | 71/200 = **35.50%** | 72/200 = **36.00%** | 72/200 = **36.00%** |
| musique | 200 | 24/200 = **12.00%** | 22/200 = **11.00%** | 26/200 = **13.00%** |
| nq | 200 | 74/200 = **37.00%** | 64/200 = **32.00%** | 74/200 = **37.00%** |
| popqa | 200 | 85/200 = **42.50%** | 87/200 = **43.50%** | 88/200 = **44.00%** |
| triviaqa | 200 | 122/200 = **61.00%** | 123/200 = **61.50%** | 120/200 = **60.00%** |
| **总成功率** | **1325** | **479 = 36.15%** | **475 = 35.85%** | **482 = 36.38%** |

### 2.2 Seed=11

| 数据集 | n | noemb_bs1 | noemb_bs4 | noemb_lr1e-5 |
|--------|-----:|:---------:|:---------:|:------------:|
| 2wikimultihopqa | 200 | 53/200 = **26.50%** | 67/200 = **33.50%** | 63/200 = **31.50%** |
| bamboogle | 125 | 38/125 = **30.40%** | 34/125 = **27.20%** | 40/125 = **32.00%** |
| hotpotqa | 200 | 65/200 = **32.50%** | 68/200 = **34.00%** | 60/200 = **30.00%** |
| musique | 200 | 31/200 = **15.50%** | 24/200 = **12.00%** | 21/200 = **10.50%** |
| nq | 200 | 61/200 = **30.50%** | 76/200 = **38.00%** | 70/200 = **35.00%** |
| popqa | 200 | 75/200 = **37.50%** | 77/200 = **38.50%** | 81/200 = **40.50%** |
| triviaqa | 200 | 103/200 = **51.50%** | 99/200 = **49.50%** | 111/200 = **55.50%** |
| **总成功率** | **1325** | **426 = 32.15%** | **445 = 33.58%** | **446 = 33.66%** |

### 2.3 Seed=21

| 数据集 | n | noemb_bs1 | noemb_bs4 | noemb_lr1e-5 |
|--------|-----:|:---------:|:---------:|:------------:|
| 2wikimultihopqa | 200 | 74/200 = **37.00%** | 79/200 = **39.50%** | 68/200 = **34.00%** |
| bamboogle | 125 | 35/125 = **28.00%** | 32/125 = **25.60%** | 36/125 = **28.80%** |
| hotpotqa | 200 | 75/200 = **37.50%** | 73/200 = **36.50%** | 73/200 = **36.50%** |
| musique | 200 | 22/200 = **11.00%** | 25/200 = **12.50%** | 23/200 = **11.50%** |
| nq | 200 | 82/200 = **41.00%** | 75/200 = **37.50%** | 78/200 = **39.00%** |
| popqa | 200 | 85/200 = **42.50%** | 84/200 = **42.00%** | 87/200 = **43.50%** |
| triviaqa | 200 | 117/200 = **58.50%** | 111/200 = **55.50%** | 118/200 = **59.00%** |
| **总成功率** | **1325** | **490 = 36.98%** | **479 = 36.15%** | **483 = 36.45%** |

---

## 3. 汇总对比

### 3.1 三 Seed 总成功率

| 模型 | Seed1 | Seed11 | Seed21 | 三 Seed 平均 | 波动范围 |
|------|:-----:|:------:|:------:|:------------:|:--------:|
| **noemb_bs1**(batch_size=1)| 36.15% | 32.15% | 36.98% | **35.09%** | 4.83% |
| **noemb_bs4** (batch_size=4) | 35.85% | 33.58% | 36.15% | **35.19%** | 2.57% |
| **noemb_lr1e-5** (batch_size=4, lr=1e-5) | 36.38% | 33.66% | 36.45% | **35.50%** | 2.79% |

**排名 (按三 Seed 平均):**
1. **noemb_lr1e-5**: 35.50% — 综合最优
2. **noemb_bs4**: 35.19% — 比 lr1e-5 低 0.31%
3. **noemb_bs1**: 35.09% — 比 lr1e-5 低 0.41%, 波动最大 (4.83%)

### 3.2 各数据集三 Seed 平均成功率

| 数据集 | noemb_bs1 | noemb_bs4 | noemb_lr1e-5 | 最优 |
|--------|:---------:|:---------:|:------------:|:----:|
| 2wikimultihopqa | 32.50% | **36.67%** | 33.67% | noemb_bs4 |
| bamboogle | **28.80%** | 25.87% | 28.53% | noemb_bs1 |
| hotpotqa | 35.17% | **35.50%** | 34.17% | noemb_bs4 |
| musique | **12.83%** | 12.50% | 11.67% | noemb_bs1 |
| nq | 36.17% | 35.83% | **36.83%** | noemb_lr1e-5 |
| popqa | 40.83% | 41.33% | **42.67%** | noemb_lr1e-5 |
| triviaqa | 57.00% | 55.50% | **58.17%** | noemb_lr1e-5 |

---

## 4. 关键结论

### 4.1 lr1e-5 效果分析

1. **noemb_lr1e-5 三 Seed 平均总成功率最高 (35.50%)**, 比 noemb_bs4 (35.19%) 提升 +0.31%, 比 noemb_bs1 (35.09%) 提升 +0.41%。
2. **lr1e-5 在 3 个数据集上占优**: nq (+0.66% vs bs1), popqa (+1.84%), triviaqa (+1.17%)。
3. **lr1e-5 在 2 个数据集上落后**: 2wikimultihopqa (-3.00% vs bs4), hotpotqa (-1.00% vs bs1)。
4. **Seed 波动**: lr1e-5 (2.79%) 与 bs4 (2.57%) 接近, 均显著优于 bs1 (4.83%)。

### 4.2 bs4 vs bs1

1. **noemb_bs4 (35.19%) 三 Seed 平均略高于 noemb_bs1 (35.09%)**, 提升 +0.10%, 差距很小。
2. **noemb_bs4 在 2wikimultihopqa 上显著领先** (36.67% vs 32.50%, +4.17%), 在 **hotpotqa 上持平** (35.50% vs 35.17%)。
3. **noemb_bs4 在 nq 上落后** (35.83% vs 36.17%), 在 **bamboogle 上落后** (25.87% vs 28.80%)。
4. **noemb_bs4 的稳定性显著优于 bs1**: 波动 2.57% vs 4.83%, 说明 bs4 训练对 seed 更不敏感。
5. 综合来看, **bs4 (step125) 与 bs1 (step500) 三 Seed 平均基本持平**, 但 bs4 稳定性更好。bs1 训练步数更多 (500 vs 125), 但更少的步数在 bs4 上已接近相同水平。

### 4.3 排名对比 (含 SFT 基准)

将 noemb 变体与之前分析的 SFT 和 emb_0.80 的三 Seed 平均数据对比:

| 模型 (三 Seed 平均) | 总成功率 | 排名 |
|------|:--------:|:----:|
| emb_0.80 | 35.85% | — |
| SFT | 35.52% | — |
| **noemb_lr1e-5** | **35.50%** | **1 (noemb 变体)** |
| noemb_bs4 | 35.19% | 2 |
| noemb_bs1 | 35.09% | 3 |

> noemb_lr1e-5 三 Seed 平均 (35.50%) 已接近 SFT (35.52%), 差距仅 0.02%, 基本持平。而 emb_0.80 (35.85%) 仍为最高。

---

## 5. 数据一致性验证

### 5.1 日志核查

| 实验 | 日志路径 | 验证结果 |
|------|----------|:--------:|
| noemb_bs1 seed1 | `coldstart_test_search_seed1_rl_action_modified_seed1/test_noemb/` | ✅ 与报告一致 |
| noemb_bs1 seed11 | `coldstart_test_search_seed1_rl_action_modified_seed11/test_noemb/` | ✅ 与报告一致 |
| noemb_bs1 seed21 | `coldstart_test_search_seed1_rl_action_modified_seed21/test_noemb/` | ✅ 与报告一致 |
| noemb_bs4 seed1 | `coldstart_noemb_bs4_test/test_noemb_bs4_seed1/*_1.log` | ✅ 与报告一致 |
| noemb_bs4 seed11 | `coldstart_noemb_bs4_test/test_noemb_bs4_seed11/*.log` | ✅ 与报告一致 |
| noemb_bs4 seed21 | `coldstart_noemb_bs4_test/test_noemb_bs4_seed21/*.log` | ✅ 与报告一致 |
| noemb_lr1e-5 seed1 | `coldstart_test_search_lr1e-5/test_noemb_seed1/*.log` | ✅ 与报告一致 |
| noemb_lr1e-5 seed11 | `coldstart_test_search_lr1e-5/test_noemb_seed11/*.log` | ✅ 与报告一致 |
| noemb_lr1e-5 seed21 | `coldstart_test_search_lr1e-5/test_noemb_seed21/*.log` | ✅ 与报告一致 |

### 5.2 已纠正的原有数据错误

在 `coldstart_test_search_lr1e-5/noemb_lr1e-5_三seed汇总结果.md` 中发现以下数据错误, 本文档已按日志原始数据更正:

| 数据点 | 原报告值 | 日志实际值 |
|--------|:--------:|:----------:|
| lr1e-5 seed11 nq | 69/200 = 34.50% | **70/200 = 35.00%** |
| lr1e-5 seed11 triviaqa | 110/200 = 55.00% | **111/200 = 55.50%** |
| lr1e-5 seed11 **总成功率** | 444/1325 = 33.51% | **446/1325 = 33.66%** |
| lr1e-5 seed21 nq | 77/200 = 38.50% | **78/200 = 39.00%** |
| lr1e-5 seed21 triviaqa | 114/200 = 57.00% | **118/200 = 59.00%** |
| lr1e-5 seed21 **总成功率** | 478/1325 = 36.08% | **483/1325 = 36.45%** |

### 5.3 分片说明

- lr1e-5 的 -1 和 -2 脚本为数据拆分并行运行: -1 跑前 4 个数据集 (nq/triviaqa/popqa/hotpotqa), -2 跑后 3 个数据集 (2wikimultihopqa/musique/bamboogle), 输出到同一目录, 合并后等价于一个完整实验。
- noemb_bs4 seed1 目录中的无后缀日志文件来自旧实验 (rl_action/noemb-1, 即 noemb_bs2 模型), 与 `_1` 后缀的 bs4 实验不同, 本文档以 `_1` 后缀为准。