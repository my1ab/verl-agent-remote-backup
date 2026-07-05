# 对话上下文摘要 — 2026-06-28

> 本文件按**对话轮次**顺序记录该会话中的关键上下文，用于迁移到新计算机后继承工作状态。

---

## 项目概览

| 项目 | 路径 |
|------|------|
| 工作区根目录 | `/diskpool/home/xuxz/verl-agent` |
| Search 冷启动数据生成 | `coldstart_genaration_search/coldstart_search.py` |
| Search 检索服务器 | `examples/search/retriever/retrieval_server.py` |
| 检索服务器原始版 | `examples/search/retriever/retrieval_server_original.py` |
| Search prompts | `coldstart_genaration_search/prompts_search.py` |
| 环境配置 | `agent_system/environments/env_package/search/` |

---

## 对话轮次记录

### 第 1 轮 — 解读 `retrieval_server.py` 与 `retrieval_server_original.py` 的区别

**用户请求**：解读两个文件的区别

**核心发现**：仅有一处不同，位于 `DenseRetriever.__init__` 中 FAISS 索引加载方式。

| 版本 | 代码 | 特点 |
|------|------|------|
| 原始版 | `faiss.read_index(index_path)` | 一次性加载全部到物理内存 |
| 修改版 | `faiss.read_index(index_path, faiss.IO_FLAG_MMAP \| faiss.IO_FLAG_READ_ONLY)` | 内存映射，按需读取，节省 RAM |

**修改版新增改动**：
- 增加了 `print(f'path = {self.index_path}')` 调试日志
- 注释说明："使用内存映射模式（只读，不占用物理内存），按需从磁盘读取"

**结论**：这是一次性能优化，通过 FAISS MMAP 机制加载大规模索引文件，降低物理内存占用、加快启动速度、支持更大索引。

---

### 第 2 轮 — `if config.faiss_gpu:` 是否触发的区别

**用户请求**：解释 `faiss_gpu` 参数在两种取值下的行为差异。

**回答要点**：

| 方面 | `faiss_gpu = False` | `faiss_gpu = True` |
|------|-------------------|-------------------|
| **索引位置** | 磁盘（MMAP 按需读取） | GPU 显存（全量驻留） |
| **CPU RAM 占用** | 极低（仅 page cache） | 中等（迁移时的临时缓冲区） |
| **GPU 显存占用** | 无 | 高（完整索引，用 float16 减半） |
| **搜索速度** | 受磁盘 IO 影响（可能慢） | **极快**（纯 GPU 计算） |
| **启动时间** | 极快（仅建映射） | 较慢（需要将索引传到 GPU） |
| **适用场景** | 索引极大、显存不足 | 索引能放进显存、追求低延迟 |

**注意事项**：当 `faiss_gpu = True` 时，`index_cpu_to_all_gpus()` 内部会全量读取索引数据再复制到 GPU，MMAP 仅节省加载阶段的临时内存。

---

### 第 3 轮 — 将注释添加到 `retrieval_server.py`

**用户请求**：将上述分析以注释形式写入 `retrieval_server.py`

**实际操作**：在 `if config.faiss_gpu:` 上方添加了多行注释，同时在 `co.useFloat16` 和 `co.shard` 行尾添加了行内注释。

**修改后代码片段**（对应 `DenseRetriever.__init__` 中第 200-209 行）：

```python
        # 使用内存映射模式（只读，不占用物理内存），按需从磁盘读取
        # 按需从磁盘读取，不占用物理内存（依赖 OS 的 page cache） 避免oom
        self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        # if config.faiss_gpu 是否触发决定了索引的存储位置和搜索性能：
        # - faiss_gpu = False: 索引留在磁盘（MMAP 按需读取），CPU RAM 占用极低，搜索受磁盘 IO 影响
        # - faiss_gpu = True:  将整个索引从磁盘全量读取并复制到 GPU 显存（此时 MMAP 仅节省加载阶段内存），搜索极快
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True  # 使用 float16 减小 GPU 显存占用
            co.shard = True       # 在多 GPU 间分片存储索引
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
```

**涉及文件**：`examples/search/retriever/retrieval_server.py`

---

### 第 4 轮 — 生成上下文摘要文件

**用户请求**：将当前对话压缩后的上下文输出到一个 md 文件，用于新计算机继承。

**实际操作**：创建了本文件 `context_summary.md`。

**用户后续反馈**：需要按对话轮次顺序记录。因此本文件从第 5 轮开始改为当前格式——**按轮次记录**。

---

## 关键知识点速查

### `retrieval_server.py` 文件结构

- **工具函数**: `load_corpus()`, `read_jsonl()`, `load_docs()`, `load_model()`, `pooling()`
- **Encoder 类**: 封装模型加载与文本编码，支持 e5/bge/T5/DPR 等模型
- **BaseRetriever**: 抽象基类，定义 `search()` / `batch_search()` 接口
- **BM25Retriever**: 基于 pyserini 的 BM25 稀疏检索
- **DenseRetriever**: 基于 FAISS + 向量编码的稠密检索
- **Config**: 配置类
- **FastAPI 服务**: POST `/retrieve` 端点

### `coldstart_search.py` 核心功能

为 Search-R1 任务生成冷启动轨迹数据。Agent 与搜索引擎交互来回答问题。

**关键配置**（`__main__` 入口）：
```python
USE_LOCAL_MODEL = False       # True=本地模型, False=DeepSeek API
DS_MODEL = 1                  # 1=Flash, 2=Pro
EFFORT = 0                    # 0=无 thinking, 1=high, 2=max
MAX_TURNS = 10                # 每问最大搜索轮数
HIS_LEN = 5                   # 历史窗口 (-1=全部历史)
SEED = -1                     # <0 → sequential, >=0 → seed 控制随机
SHOW_TURN = True
```

**数据路径**：
```python
SEARCH_DATA_DIR = os.path.expanduser('~/data/searchR1_processed_direct')
#   test.parquet  → range(0, 500)
#   train.parquet → range(1500, end)
#   sft          → train.parquet, range(600, end)
```

**检索服务器**：
```python
SEARCH_URL = 'http://127.0.0.1:8000/retrieve'
# 启动方式: bash examples/search/retriever/retrieval_launch.sh
```

**SearchTaskSampler 行为**：
- `sequential` 模式：顺序取，不重复
- `exclude` 模式：随机不重复
- 默认模式：随机可重复

**轨迹格式**：
- `messages`: 完整对话历史（system + user + assistant）
- `seperated_list`: 按轮次拆分的部分消息（用于训练）
  - 每个元素: `{"messages": [task_idx/turn信息, system_msg, user_msg, assistant_msg]}`
- 每轮 assistant 消息含: `action_type`, `action_content`, `think`, `won`, `reward`

**Action 提取**：正则提取 `<think>...</think>`, `<search>...</search>`, `<answer>...</answer>`

**Search 环境 API**：
```python
env.reset(kwargs) → (obs_list, info_list)     # kwargs 需含 ground_truth, question, data_source
env.step(actions) → (obs_list, reward_list, done_list, info_list)
# obs: search 动作返回搜索结果, answer 动作返回 ""
# reward: done=True 时有意义 (1.0=正确)
# done: answer 找到或 max_turns 耗尽
# info: 含 "won" 键
```

### 启动流程

```bash
# 1. 启动检索服务器
bash examples/search/retriever/retrieval_launch.sh

# 2. 运行冷启动数据生成
cd /diskpool/home/xuxz/verl-agent
conda activate base
python coldstart_genaration_search/coldstart_search.py
```

### 输出文件

保存在 `coldstart_genaration_search/` 目录下：
- `search_coldstart_test.json` — 全部轨迹
- `search_coldstart_test_success.json` — 成功轨迹
- `search_coldstart_test_seperated.json` — 按轮次拆分的训练数据
- `search_coldstart_test.log` — 运行日志

### 环境依赖

- Python: conda base
- PyTorch + Transformers (HuggingFace)
- FAISS (GPU 版)
- FastAPI + uvicorn
- pyserini (BM25)
- datasets
- omegaconf
- openai (DeepSeek API)
- pandas / numpy / tqdm
