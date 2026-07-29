"""
================================================================
Search Coldstart — 多线程 + 本地知识库缓存 加速版
================================================================
相比原版 coldstart_search_local_3epoch.py 的改动:

1. 搜索缓存 (本地知识库)
   - 相同 query 不会重复调用检索服务器
   - 内存 + 磁盘两级缓存，跨 session 持久化
   - 线程安全，多线程共享

2. 多线程流水线
   - ThreadPoolExecutor 并行处理多个 task
   - 模型推理通过 threading.Lock 串行化（单 GPU 限制）
   - 搜索 I/O 可与模型推理重叠

3. 连接池复用
   - requests.Session 复用 HTTP Keep-Alive 连接

4. 模型推理加锁保护
   - 确保单 GPU 场景下不会多线程冲突

使用方法:
    python coldstart_search_local_3epoch_multithread.py
================================================================
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import json
import time
import re
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent'))

from agent_system.environments.env_package.search import build_search_envs
print("import build_search_envs success")

from coldstart_genaration_search.prompts_search import (
    SYSTEM_PROMPT_SEARCH_PARA,
    USER_PROMPT_NO_HIS_PARA,
    USER_PROMPT_HIS_PARA,
)
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 多线程 + 搜索缓存支持
# ============================================================
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval_search.coldstart_test_search.search_cache import SearchCache

# 全局搜索缓存（所有线程共享，自动持久化到磁盘）
GLOBAL_SEARCH_CACHE = SearchCache(
    cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_cache_db"),
    use_memory=True,
)

# 模型推理锁（单 GPU 同一时间只能跑一次推理，用可重入锁防止意外递归）
MODEL_INFER_LOCK = threading.Lock()

# ============================================================
# Global Configuration
# ============================================================
local_model = None
local_tokenizer = None

MAX_CONTEXT_LENGTH = 32768
BASE_MODEL_PATH = '/diskpool/home/xuxz/ms-swift/model/Qwen2.5-1.5B-Instruct'
SEARCH_URL = 'http://127.0.0.1:8000/retrieve'
SEARCH_TOPK = 3
SEARCH_TIMEOUT = 60
SEARCH_LOG_REQUESTS = False

SEARCH_DATA_DIR = os.path.expanduser('~/data/searchR1_processed_direct')


# ============================================================
# Model Loading
# ============================================================
def load_local_model(tokenizer_path=None, model_path=None, show=1):
    global local_model, local_tokenizer
    if model_path is not None:
        print(f"\n{'='*60}")
        print(f"Loading tokenizer and model from checkpoint: {tokenizer_path}")
        print(f"{'='*60}")

        print("Loading tokenizer...")
        try:
            local_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True
            )
            print(f"Tokenizer loaded: {local_tokenizer.__class__.__name__}")

            if hasattr(local_tokenizer, 'model_max_length'):
                local_tokenizer.model_max_length = MAX_CONTEXT_LENGTH
                print(f"Set tokenizer model_max_length to: {MAX_CONTEXT_LENGTH}")

            local_tokenizer.truncation_side = "right"
            print(f"Set tokenizer truncation_side to: {local_tokenizer.truncation_side}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            import traceback
            traceback.print_exc()
            raise

        if show:
            print(f"\nLoading model weights from {model_path}")
        local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='bfloat16',
            device_map='auto',
            attn_implementation='flash_attention_2',
            local_files_only=True,
            use_safetensors=True
        )
        local_model.eval()

        if show:
            print(f"\n{'='*60}")
            print(f"Model loaded successfully!")
            print(f"Model type: {local_model.__class__.__name__}")
            print(f"Model device: {local_model.device}")
            print(f"Model dtype: {local_model.dtype}")
            print(f"Number of parameters: {sum(p.numel() for p in local_model.parameters()):,}")
            print(f"Max context length: {MAX_CONTEXT_LENGTH}")
            print(f"{'='*60}")

        print("load_local_model success")
        print(f"{'='*60}")
    return local_model, local_tokenizer


def local_model_infer(messages, max_new_tokens=4096, show=0):
    """Run inference with the local model (线程安全，带锁保护)。"""
    with MODEL_INFER_LOCK:
        model, tokenizer = load_local_model()
        text = tokenizer.apply_chat_template(messages, tokenize=False)

        max_input_length = MAX_CONTEXT_LENGTH - max_new_tokens

        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=False,
            padding=False
        ).to(model.device)

        input_length = inputs['input_ids'].shape[1]

        if input_length > max_input_length:
            if show:
                print(f"WARNING: Input truncated to max length!")
                print(f"Original input tokens: {input_length}, truncating to {max_input_length}")
            inputs = {
                'input_ids': inputs['input_ids'][:, :max_input_length],
                'attention_mask': inputs['attention_mask'][:, :max_input_length]
            }

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
                use_cache=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        actual_input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if response.startswith(actual_input_text):
            response = response[len(actual_input_text):].strip()
        return response


from openai import OpenAI, APIConnectionError, APITimeoutError

DEEPSEEK_RETRY_MAX = 5
DEEPSEEK_RETRY_INTERVAL = 1


def deepseek(messages, ds_model=1, effort=0, show=0, turn=None):
    """Use DeepSeek API for inference."""
    client = OpenAI(api_key="sk-b718f52386c34ffeb714f684d225f688", base_url="https://api.deepseek.com")
    model_name = "deepseek-v4-flash" if ds_model == 1 else "deepseek-v4-pro"
    turn_tag = f"[turn {turn}] "

    last_exception = None
    for attempt in range(1, DEEPSEEK_RETRY_MAX + 1):
        try:
            if effort == 0:
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    extra_body={"thinking": {"type": "disabled"}},
                    messages=messages,
                    stream=False,
                )
            else:
                reasoning_effort = "high" if effort == 1 else "max"
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    extra_body={"thinking": {"type": "enabled"}},
                    reasoning_effort=reasoning_effort,
                    messages=messages,
                    stream=False,
                )

            if show:
                print(f'{turn_tag}model: {model_name}, effort: {"disabled" if effort == 0 else reasoning_effort}')
            return response.choices[0].message.content

        except (APIConnectionError, APITimeoutError) as e:
            last_exception = e
            if attempt < DEEPSEEK_RETRY_MAX:
                print(f"  {turn_tag}[Retry {attempt}/{DEEPSEEK_RETRY_MAX}] DeepSeek API 连接失败，{DEEPSEEK_RETRY_INTERVAL}s 后重试...")
                time.sleep(DEEPSEEK_RETRY_INTERVAL)
            else:
                print(f"  {turn_tag}[Retry {attempt}/{DEEPSEEK_RETRY_MAX}] 已达最大重试次数，放弃。")
                raise last_exception
        except Exception as e:
            raise e


# ============================================================
# Model Test
# ============================================================
def test_model(ds_model=1, effort=0, use_local_model=False):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "introduce yourself."},
    ]

    if use_local_model:
        print("=" * 60)
        print("Testing Local model connectivity...")
        print("=" * 60)
        try:
            response = local_model_infer(messages, show=1)
            print(f"\nResponse: {response}")
            print(f"\n{'='*60}")
            print("Local model test PASSED!")
            print(f"{'='*60}")
            return True
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Local model test FAILED: {e}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("=" * 60)
        print("Testing DeepSeek API connectivity...")
        print("=" * 60)
        try:
            response = deepseek(messages, ds_model=ds_model, effort=effort, show=1)
            print(f"\nResponse: {response}")
            print(f"\n{'='*60}")
            print("DeepSeek API test PASSED!")
            print(f"{'='*60}")
            return True
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"DeepSeek API test FAILED: {e}")
            print(f"{'='*60}")
            return False


# ============================================================
# Prompt Helpers (不变)
# ============================================================
def format_parallel_observations(obs_list, total_envs):
    parts = []
    for i in range(total_envs):
        obs = obs_list[i] if i < len(obs_list) else ""
        parts.append(f"<observation_{i+1}>\n{obs}\n</observation_{i+1}>")
    return "\n".join(parts)


def format_parallel_history(history_list, turn, his_len, total_envs):
    lines = []
    for env_idx in range(total_envs):
        env_lines = [f"In Environment {env_idx+1}"]
        if his_len < 0:
            start = 0
        else:
            start = max(0, turn - his_len)
        for t_idx in range(start, turn):
            action = history_list[env_idx]['Action'][t_idx]
            obs = history_list[env_idx]['Observation'][t_idx]
            obs_trimmed = obs[:2000] if obs and len(obs) > 2000 else (obs or "")
            env_lines.append(f"  Action {t_idx+1}: {action}")
            env_lines.append(f"  Observation {t_idx+1}: {obs_trimmed}")
        lines.append("\n".join(env_lines))
    return "\n\n".join(lines)


def format_parallel_last_history(history_list, turn, total_envs):
    lines = []
    for env_idx in range(total_envs):
        action = history_list[env_idx]['Action'][-1] if history_list[env_idx]['Action'] else "null"
        obs = history_list[env_idx]['Observation'][-1] if history_list[env_idx]['Observation'] else ""
        obs_trimmed = obs[:2000] if obs and len(obs) > 2000 else (obs or "")
        lines.append(f"In Environment {env_idx+1}")
        lines.append(f"  Action {turn}: {action}")
        lines.append(f"  Observation {turn}: {obs_trimmed}")
    return "\n".join(lines)


def get_search_system_message_para(num_para, total_envs):
    return {"role": "system", "content": SYSTEM_PROMPT_SEARCH_PARA.format(
        num_parallel=num_para, total_envs=total_envs
    )}


def get_search_user_message_para(question, initial_observations, total_envs, num_para,
                                 history_info=None, last_history=None):
    init_obs_str = format_parallel_observations(initial_observations, total_envs)
    if history_info is not None and last_history is not None:
        content = USER_PROMPT_HIS_PARA.format(
            task_description=question,
            initial_observation=init_obs_str,
            history_info=history_info,
            last_history=last_history,
            total_envs=total_envs,
            num_parallel=num_para,
        )
    else:
        content = USER_PROMPT_NO_HIS_PARA.format(
            question=question,
            observations=init_obs_str,
            total_envs=total_envs,
            num_parallel=num_para,
        )
    return {"role": "user", "content": content}


# ============================================================
# Action Extraction
# ============================================================
def extract_parallel_search_action(text, total_envs=1):
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None

    actions = {}
    parallel_pattern = r'<parallel>(.*?)</parallel>'
    parallel_match = re.search(parallel_pattern, text, re.DOTALL)
    if parallel_match:
        parallel_content = parallel_match.group(1)
        env_pattern = r'<env_(\d+)>(.*?)</env_\d+>'
        for env_idx_str, raw_action in re.findall(env_pattern, parallel_content, re.DOTALL):
            env_idx = int(env_idx_str) - 1
            raw_action = raw_action.strip()
            inner_search = re.search(r'<search>(.*?)</search>', raw_action, re.DOTALL)
            inner_answer = re.search(r'<answer>(.*?)</answer>', raw_action, re.DOTALL)
            if inner_search:
                actions[env_idx] = {
                    "type": "search",
                    "content": inner_search.group(1).strip(),
                    "raw": raw_action,
                }
            elif inner_answer:
                actions[env_idx] = {
                    "type": "answer",
                    "content": inner_answer.group(1).strip(),
                    "raw": raw_action,
                }
            else:
                actions[env_idx] = {
                    "type": None,
                    "content": raw_action,
                    "raw": raw_action,
                }

    for i in range(total_envs):
        if i not in actions:
            actions[i] = {"type": None, "content": None, "raw": "null"}

    all_null = all(a["type"] is None for a in actions.values())

    return {
        'think': think_content,
        'actions': actions,
        'all_null': all_null,
    }


# ============================================================
# Unique filename helper
# ============================================================
def get_unique_filename(file_path):
    if not os.path.exists(file_path):
        return file_path
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    counter = 1
    while True:
        new_file_name = f"{name}_{counter}{ext}"
        new_file_path = os.path.join(dir_name, new_file_name) if dir_name else new_file_name
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1


# ============================================================
# 缓存辅助函数 — 包装 env.step，插入搜索缓存
# ============================================================
def cached_env_step(env, env_actions, total_envs, step_actions_dict):
    """包装 env.step()，利用搜索缓存避免重复请求。

    对已经缓存过的 search query，直接用缓存结果返回，
    未缓存的走 env.step，然后将结果写入缓存。
    """
    # 检查是否有已缓存的 search query
    all_cached = True
    cached_results = {}
    need_real_step = False

    for env_idx in range(total_envs):
        action_info = step_actions_dict.get(env_idx, {})
        if action_info.get("type") == "search":
            query = action_info.get("content", "")
            cached = GLOBAL_SEARCH_CACHE.get(query)
            if cached is not None:
                cached_results[env_idx] = cached
            else:
                all_cached = False
                need_real_step = True
        else:
            # answer 或 null action 不需要缓存
            pass

    # 如果全部已缓存，直接组装返回
    if all_cached and cached_results:
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        for env_idx in range(total_envs):
            action_info = step_actions_dict.get(env_idx, {})
            if action_info.get("type") == "search" and env_idx in cached_results:
                obs_list.append(cached_results[env_idx]["obs"])
                reward_list.append(cached_results[env_idx]["reward"])
                done_list.append(False)
                info_list.append({})
            elif action_info.get("type") == "answer":
                obs_list.append("")
                reward_list.append(0.0)
                done_list.append(True)
                info_list.append({"won": False})
            else:
                obs_list.append("")
                reward_list.append(None)
                done_list.append(False)
                info_list.append({})
        return obs_list, reward_list, done_list, info_list

    # 部分或全部未缓存 → 走真实 env.step
    obs_list, reward_list, done_list, info_list = env.step(env_actions)

    # 将 search 结果写入缓存
    for env_idx in range(total_envs):
        action_info = step_actions_dict.get(env_idx, {})
        if action_info.get("type") == "search":
            query = action_info.get("content", "")
            if env_idx < len(obs_list) and obs_list[env_idx]:
                GLOBAL_SEARCH_CACHE.set(query, {
                    "obs": obs_list[env_idx],
                    "reward": reward_list[env_idx] if env_idx < len(reward_list) else None,
                })

    return obs_list, reward_list, done_list, info_list


# ============================================================
# Trajectory Generation (单任务，供多线程调用)
# ============================================================
def get_single_trajectory(env, question, ground_truth, data_source, task_idx=0,
                          max_turns=10, show_turn=False, his_len=5,
                          use_local_model=True, ds_model=1, effort=0,
                          group_n=1, env_num=1, num_para=1,
                          _thread_log=None):
    """同原版 get_single_trajectory，但 use_local_model=True 时模型推理已加锁保护。"""
    total_envs = group_n * env_num
    num_para = min(num_para, total_envs)

    messages = []
    success_flag = 0
    seperated_list = []

    kwargs = [{
        "ground_truth": ground_truth,
        "question": question,
        "data_source": data_source
    }] * total_envs
    obs_list, info_list = env.reset(kwargs)
    current_obs_list = list(obs_list)
    initial_obs_list = list(obs_list)

    history_list = []
    for _ in range(total_envs):
        history_list.append({'Action': [], 'Observation': []})

    messages.append(get_search_system_message_para(num_para, total_envs))

    null_count = 0

    for turn in range(max_turns):
        if turn == 0:
            user_msg = get_search_user_message_para(
                question, initial_obs_list, total_envs, num_para,
                history_info=None, last_history=None
            )
        else:
            history_info = format_parallel_history(history_list, turn, his_len, total_envs)
            last_history = format_parallel_last_history(history_list, turn, total_envs)
            user_msg = get_search_user_message_para(
                question, initial_obs_list, total_envs, num_para,
                history_info=history_info, last_history=last_history
            )

        messages.append(user_msg)

        partial_messages = [
            {"task_idx": task_idx, "turn": turn},
            messages[0],
            messages[-1],
        ]

        # ── 模型推理（local_model_infer 内部已加 MODEL_INFER_LOCK） ──
        if use_local_model:
            assistant_response = local_model_infer(messages=partial_messages[1:])
        else:
            assistant_response = deepseek(messages=partial_messages[1:],
                                          ds_model=ds_model, effort=effort,
                                          turn=turn)

        partial_messages.append({"role": "assistant", "content": assistant_response})
        seperated_list.append({"messages": partial_messages.copy()})

        para_result = extract_parallel_search_action(assistant_response, total_envs)
        step_actions_dict = {}
        for env_idx in range(total_envs):
            a = para_result['actions'][env_idx]
            step_actions_dict[env_idx] = a["raw"] if a["type"] is not None else "null"
        all_null = para_result['all_null']

        env_actions = []
        for env_idx in range(total_envs):
            raw = step_actions_dict.get(env_idx, "null")
            env_actions.append("" if (raw == "null" or raw is None) else raw)

        # ── 走缓存感知的 env.step ────────────────────────────────
        if not all_null:
            obs_list, reward_list, done_list, info_list = cached_env_step(
                env, env_actions, total_envs, step_actions_dict
            )
            null_count = 0
        else:
            null_count += 1
            obs_list = [""] * total_envs
            reward_list = [None] * total_envs
            done_list = [False] * total_envs
            info_list = [{}] * total_envs

        new_obs_list = list(obs_list)[:total_envs]
        new_reward_list = list(reward_list)[:total_envs]
        new_done_list = list(done_list)[:total_envs]
        new_info_list = list(info_list)[:total_envs]

        for env_idx in range(total_envs):
            action_raw = step_actions_dict.get(env_idx, "null")
            action_content = action_raw
            inner_s = re.search(r'<search>(.*?)</search>', action_raw, re.DOTALL)
            inner_a = re.search(r'<answer>(.*?)</answer>', action_raw, re.DOTALL)
            if inner_s:
                action_content = inner_s.group(1).strip()
            elif inner_a:
                action_content = inner_a.group(1).strip()
            history_list[env_idx]['Action'].append(action_content if action_content else 'null')
            history_list[env_idx]['Observation'].append(new_obs_list[env_idx])

        current_obs_list = new_obs_list

        assistant_msg = {
            "task_idx": task_idx,
            "role": "assistant",
            "content": assistant_response,
            "rewards": json.dumps(new_reward_list),
            "get_actions": step_actions_dict,
            "think": para_result['think'],
        }
        if show_turn:
            assistant_msg["turn"] = turn + 1
        messages.append(assistant_msg)

        if all_null and null_count >= 2:
            status_msg = f"Task {task_idx} exit(all null) at turn {turn + 1}"
            return messages, success_flag, status_msg, seperated_list

        any_done = any(new_done_list)
        any_success = any(r is not None and r >= 1.0 for r in new_reward_list)

        if any_done:
            completed_idx = [i for i, d in enumerate(new_done_list[:total_envs]) if d]
            if any_success:
                success_flag = 1
                status_msg = f"Task {task_idx} SUCCESS at turn {turn + 1} in environments {completed_idx}"
            else:
                status_msg = f"Task {task_idx} FAILED at turn {turn + 1} in environments {completed_idx}"
            break
    else:
        status_msg = f"Task {task_idx} out of max turn"

    return messages, success_flag, status_msg, seperated_list


# ============================================================
# SearchTaskSampler (不变)
# ============================================================
class SearchTaskSampler:
    _SPLIT_CONFIG = {
        "test":  {"file": "test.parquet",  "range": (0, None)},
        "train": {"file": "train.parquet", "range": (1500, None)},
        "sft":   {"file": "train.parquet", "range": (0, None)},
    }

    def __init__(self, data_dir=None, split="test", split_range=None, seed=42, sequential=True, exclude=False, batch_size=1):
        data_dir = data_dir or SEARCH_DATA_DIR
        split_cfg = self._SPLIT_CONFIG.get(split)
        if split_cfg is None:
            raise ValueError(f"Unknown split '{split}'. Valid: {list(self._SPLIT_CONFIG.keys())}")

        data_path = os.path.join(data_dir, split_cfg["file"])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")

        print(f"SearchTaskSampler loading: {data_path}")
        self._df = pd.read_parquet(data_path)
        self._total_rows = len(self._df)
        print(f"  Total rows: {self._total_rows}")

        if split_range is not None:
            start, end = split_range
        else:
            start, end = split_cfg["range"]
        if end is None:
            end = self._total_rows
        end = min(end, self._total_rows)
        start = min(start, end)
        self._goal_idxs = list(range(start, end))
        print(f"  Split '{split}': goal_idxs = [{start}, {end})  ({len(self._goal_idxs)} tasks)")

        if seed < 0:
            self._sequential = True
            self._exclude = False
            self._seed = seed
            self._rng = None
            mode = "sequential (seed<0 forced)"
        else:
            self._seed = seed
            self._rng = np.random.RandomState(seed)
            self._sequential = sequential
            self._exclude = exclude and not sequential
            mode = "sequential" if self._sequential else (f"exclude (seed={seed})" if self._exclude else f"random (seed={seed})")

        self._batch_size = batch_size
        self._goal_idx_counter = 0
        self._used_goal_idxs = set()
        self._is_exhausted = False
        print(f"Sampling mode: {mode}, Seed: {seed}")

    @property
    def goal_start(self): return self._goal_idxs[0] if self._goal_idxs else 0

    @property
    def goal_end(self): return self._goal_idxs[-1] + 1 if self._goal_idxs else 0

    @property
    def total_tasks(self): return len(self._goal_idxs)

    @property
    def seed(self): return self._seed

    @property
    def remaining(self):
        if self._sequential:
            return len(self._goal_idxs) - self._goal_idx_counter
        elif self._exclude:
            return len(self._goal_idxs) - len(self._used_goal_idxs)
        else:
            return len(self._goal_idxs)

    def has_next(self): return self.remaining > 0

    def reset(self):
        self._goal_idx_counter = 0
        self._used_goal_idxs = set()
        self._is_exhausted = False

    def sample(self, n=None):
        if n is None:
            n = self._batch_size
        if not self.has_next():
            raise RuntimeError("No tasks remaining in sampler")

        if self._sequential:
            avail = self._goal_idxs[self._goal_idx_counter:]
            selected = avail[:n]
            self._goal_idx_counter += len(selected)
        elif self._exclude:
            avail = [g for g in self._goal_idxs if g not in self._used_goal_idxs]
            k = min(n, len(avail))
            selected = list(self._rng.choice(avail, size=k, replace=False))
            self._used_goal_idxs.update(selected)
        else:
            k = min(n, len(self._goal_idxs))
            selected = list(self._rng.choice(self._goal_idxs, size=k, replace=False))

        tasks = []
        for idx in selected:
            row = self._df.iloc[idx]
            env_kwargs = row.get("env_kwargs", {})
            if isinstance(env_kwargs, str):
                env_kwargs = json.loads(env_kwargs)
            question = env_kwargs.get("question", row.get("question", ""))
            ground_truth = env_kwargs.get("ground_truth", row.get("ground_truth", ""))
            data_source = env_kwargs.get("data_source", row.get("data_source", "unknown"))
            tasks.append({
                "question": question,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "task_idx": int(idx),
            })
        return tasks

    def __len__(self): return self.total_tasks

    def __repr__(self):
        return (f"SearchTaskSampler(total={self.total_tasks}, "
                f"remaining={self.remaining}, "
                f"sequential={self._sequential}, "
                f"exclude={self._exclude})")


# ============================================================
# 多线程执行单个 task 的包装函数
# ============================================================
def _run_single_task(task, env, max_turns, show_turn, his_len,
                     use_local_model, ds_model, effort,
                     group_n, env_num, num_para):
    """单个 task 的执行包装，供线程池调用。"""
    try:
        trajectory, success_flag, status_msg, seperated_list = get_single_trajectory(
            env=env,
            question=task["question"],
            ground_truth=task["ground_truth"],
            data_source=task["data_source"],
            task_idx=task["task_idx"],
            max_turns=max_turns,
            show_turn=show_turn,
            his_len=his_len,
            use_local_model=use_local_model,
            ds_model=ds_model,
            effort=effort,
            group_n=group_n,
            env_num=env_num,
            num_para=num_para,
        )
        return {
            "task_idx": task["task_idx"],
            "trajectory": trajectory,
            "success_flag": success_flag,
            "status_msg": status_msg,
            "seperated_list": seperated_list,
            "error": None,
        }
    except Exception as e:
        import traceback
        error_msg = f"Error evaluating sample {task['task_idx']}: {e}"
        print(f"\n  {error_msg}")
        traceback.print_exc()
        return {
            "task_idx": task["task_idx"],
            "trajectory": None,
            "success_flag": 0,
            "status_msg": error_msg,
            "seperated_list": [],
            "error": str(e),
        }


# ============================================================
# Main Evaluation — 多线程版
# ============================================================
def evaluate_coldstart_data_multithread(
    output_file, sampler, max_turns=10,
    show_turn=False, his_len=5,
    save_traj=1, use_local_model=True,
    ds_model=1, effort=0,
    start_idx=None, end_idx=None,
    group_n=1, env_num=1, num_para=1,
    max_workers=4,               # ← 新增：线程池大小
):
    """
    多线程版本的 evaluate_coldstart_data。
    相比原版，使用 ThreadPoolExecutor 并行处理多个 task。

    Args:
        max_workers: 线程池大小。
                     当 use_local_model=True（单 GPU），推荐 2~4。
                     当 use_local_model=False（API），推荐 8~16。
    """
    log_file = output_file.replace('.json', '.log')
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

    print(f'Start time: {start_time_str}')
    print(f'log_file: {log_file}')

    total_envs = group_n * env_num
    config_lines = [
        f'Evaluating search coldstart (MULTITHREAD): {sampler.total_tasks} samples',
        f'max_turns: {max_turns}, his_len: {his_len}',
        f'use_local_model: {use_local_model}',
        f'sampler: {sampler}',
        f'search_url: {SEARCH_URL}, topk: {SEARCH_TOPK}',
        f'group_n: {group_n}, env_num: {env_num}, total_envs: {total_envs}, num_para: {num_para}',
        f'max_workers: {max_workers}',
        f'search_cache: {GLOBAL_SEARCH_CACHE._cache_dir}',
    ]
    with open(log_file, 'w') as f:
        for line in config_lines:
            f.write(line + '\n')
            print(line)

    # Build environment
    print(f'building environment')
    env_config = OmegaConf.create({
        'max_steps': max_turns,
        'search': {
            'search_url': SEARCH_URL,
            'topk': SEARCH_TOPK,
            'timeout': SEARCH_TIMEOUT,
            'log_requests': SEARCH_LOG_REQUESTS,
        }
    })
    env_seed = max(sampler.seed, 0)
    env = build_search_envs(
        seed=env_seed,
        env_num=env_num,
        group_n=group_n,
        is_train=False,
        env_config=env_config,
    )
    print(f'Environment built successfully')

    # 确定处理范围
    goal_slice = sampler._goal_idxs
    if start_idx is not None and end_idx is not None:
        if end_idx < start_idx:
            raise ValueError(f"end_idx ({end_idx}) must be >= start_idx ({start_idx})")
        goal_slice = goal_slice[start_idx:end_idx + 1]
        print(f"Processing idx range: [{start_idx}, {end_idx}]  ({len(goal_slice)} tasks)")

    all_trajectories = []
    success_trajectories = []
    seperated_trajectories = []
    success_indices = []
    success_count = 0
    total_count = 0

    # ── 预收集所有 task ─────────────────────────────────────
    tasks_to_run = []
    sampler._goal_idx_counter = start_idx if start_idx is not None else 0
    for absolute_idx in goal_slice:
        tasks = sampler.sample(n=1)
        if not tasks:
            break
        task = tasks[0]
        if not task["question"]:
            print(f"WARNING: Skipping idx {absolute_idx} - empty question")
            continue
        tasks_to_run.append(task)

    print(f"\nTotal tasks to process: {len(tasks_to_run)}")
    print(f"ThreadPool workers: {max_workers}")
    print(f"Search cache: {len(GLOBAL_SEARCH_CACHE)} entries")
    print(f"{'='*60}\n")

    # ── 多线程并行执行 ──────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(
                _run_single_task,
                task, env, max_turns, show_turn, his_len,
                use_local_model, ds_model, effort,
                group_n, env_num, num_para,
            ): task
            for task in tasks_to_run
        }

        # 收集结果（按完成顺序）
        with tqdm(total=len(tasks_to_run), desc="Multi-thread evaluating") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                task_idx = result["task_idx"]

                # 写 log
                with open(log_file, 'a') as f:
                    f.write(result["status_msg"] + '\n')

                if result["error"] is not None:
                    pbar.update(1)
                    continue

                all_trajectories.append({
                    "task_idx": task_idx,
                    "trajectory": result["trajectory"],
                })
                seperated_trajectories.append(result["seperated_list"])
                total_count += 1

                if result["success_flag"] == 1:
                    success_trajectories.append({
                        "task_idx": task_idx,
                        "trajectory": result["trajectory"],
                    })
                    success_indices.append(task_idx)
                    success_count += 1

                pbar.update(1)

    # ── 统计 ─────────────────────────────────────────────────
    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"\n{'='*60}")
    print(f"Total: {total_count}, Success: {success_count}, Rate: {success_rate:.2%}")
    GLOBAL_SEARCH_CACHE.print_stats()
    print(f"{'='*60}")

    # ── 保存 ─────────────────────────────────────────────────
    if save_traj:
        with open(output_file, 'w') as f:
            json.dump(all_trajectories, f, indent=4)
        print(f"Trajectories saved to {output_file}")

        if success_trajectories:
            success_output = output_file.replace('.json', '_success.json')
            with open(success_output, 'w') as f:
                json.dump(success_trajectories, f, indent=4)
            print(f"Success trajectories saved to {success_output}")

        if seperated_trajectories:
            seperated_output = output_file.replace('.json', '_seperated.json')
            with open(seperated_output, 'w') as f:
                json.dump(seperated_trajectories, f, indent=4)
            print(f"Seperated data saved to {seperated_output}")
            print(f"Seperated count: {len(seperated_trajectories)}")

    with open(log_file, 'a') as f:
        f.write(f"\nTotal: {total_count}, Success: {success_count}, Rate: {success_rate:.2%}\n")
        f.write(f"Success indices: {success_indices}\n")
        f.write(f"Seperated count: {len(seperated_trajectories)}\n")
        f.write(f"Search cache stats: {GLOBAL_SEARCH_CACHE.stats}\n")
        if save_traj:
            f.write(f"Trajectories saved to {output_file}\n")

    env.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time:.1f}s')
    with open(log_file, 'a') as f:
        f.write(f"Elapsed time: {elapsed_time:.1f}s\n")

    return all_trajectories


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    """
    ================================================================
    多线程 + 搜索缓存加速版入口
    ================================================================
    """
    ckpt_path = "/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260705-053350/checkpoint-4500"
    load_local_model(tokenizer_path=ckpt_path, model_path=ckpt_path, show=1)
    test_model(ds_model=1, effort=1, use_local_model=True)
    print(f'[DEBUG] test done')

    # ── 超参数 ──────────────────────────────────────────
    his_len = 8
    group_n = 5
    env_num = 1
    num_para = group_n
    import argparse
    parser = argparse.ArgumentParser(description="Search Coldstart Data Generation")
    parser.add_argument("--seed", type=int, required=True,
                        help="随机种子; <0 → sequential, >=0 → 随机采样")
    args = parser.parse_args()
    seed = args.seed
    show_turn = True
    max_turns = 25
    save_traj = 1

    # ── Sampler ─────────────────────────────────────────
    sampler = SearchTaskSampler(
        data_dir=SEARCH_DATA_DIR,
        split="test",
        split_range=None,
        seed=seed,
    )

    # ── 分片配置 ─────────────────────────────────────────
    slice_ranges = [
        (0, 99),
        (100, 199),
        (200, 299),
        (300, 399),
        (400, 499),
    ]

    print(f"\n{'='*80}")
    print(f"Multithread mode enabled!")
    print(f"Sampler: {sampler}")
    print(f"Parallel config: group_n={group_n}, env_num={env_num}, total_envs={group_n * env_num}, num_para={num_para}")
    print(f"Slice ranges: {slice_ranges}")
    print(f"{'='*80}\n")

    sampler.reset()

    OUTPUT_BASE_DIR = '/diskpool/home/xuxz/verl-agent/coldstart_test_search/result_test_3epoch_multithread'
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    range1 = slice_ranges

    # ═══════════════════════════════════════════════════════
    # 关键参数：max_workers
    #   - use_local_model=True (单GPU) → 推荐 2~4
    #     因为 GPU 推理是串行的，过多线程只会增加锁竞争
    #   - use_local_model=False (API)  → 推荐 8~16
    #     因为 API 调用是纯 I/O 等待，可以大量并发
    # ═══════════════════════════════════════════════════════
    MAX_WORKERS = 3  # 单 GPU 推荐 2~4

    for chunk_start, chunk_end in range1:
        sampler._goal_idx_counter = chunk_start

        output_file = get_unique_filename(
            os.path.join(OUTPUT_BASE_DIR, f'search_coldstart_{chunk_start}_{chunk_end}.json')
        )
        print(f"\n{'─'*60}")
        print(f"Processing chunk [{chunk_start}, {chunk_end}] → {os.path.basename(output_file)}")
        print(f"ThreadPool workers: {MAX_WORKERS}")
        print(f"Cached search queries: {len(GLOBAL_SEARCH_CACHE)}")
        print(f"{'─'*60}")

        evaluate_coldstart_data_multithread(
            output_file=output_file,
            sampler=sampler,
            max_turns=max_turns,
            show_turn=show_turn,
            his_len=his_len,
            save_traj=save_traj,
            use_local_model=True,
            ds_model=1,
            effort=1,
            start_idx=chunk_start,
            end_idx=chunk_end,
            group_n=group_n,
            env_num=env_num,
            num_para=num_para,
            max_workers=MAX_WORKERS,
        )

    print(f"\n{'='*80}")
    print("All chunks completed!")
    print(f"Total cached search queries: {len(GLOBAL_SEARCH_CACHE)}")
    GLOBAL_SEARCH_CACHE.print_stats()
    print(f"{'='*80}")
