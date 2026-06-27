"""
Search Task Coldstart Data Generation Script
Based on: coldstart_test_search/coldstart_para_his_test_1.5B_hislen8_epoch3.5_v2.py

Generates coldstart trajectories for the Search-R1 task.
The agent interacts with a search engine to answer questions.
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
    SYSTEM_MESSAGE_SEARCH,
    SEARCH_PROMPT_NO_HIS,
    SEARCH_PROMPT_HIS,
)
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Configuration
# ============================================================
MAX_CONTEXT_LENGTH = 32768
local_model = None
local_tokenizer = None

# Model paths (adjust as needed)
BASE_MODEL_PATH = '/diskpool/home/xuxz/ms-swift/model/Qwen2.5-1.5B-Instruct'

# Search environment config
SEARCH_MAX_STEPS = 10  # max search turns per question
SEARCH_URL = 'http://127.0.0.1:8000/retrieve'
SEARCH_TOPK = 3
SEARCH_TIMEOUT = 60
SEARCH_LOG_REQUESTS = False

# Data paths
SEARCH_DATA_PATH = os.path.expanduser('~/data/searchR1_processed_direct/test.parquet')

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
    """Run inference with the local model."""
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


def deepseek(messages, ds_model=1, effort=0, show=0):
    """Use DeepSeek API for inference."""
    from openai import OpenAI
    client = OpenAI(api_key="sk-d588ac9454c84f4186db19750c4c8a11", base_url="https://api.deepseek.com")

    model_name = "deepseek-v4-flash" if ds_model == 1 else "deepseek-v4-pro"

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
        print(f'model: {model_name}, effort: {"disabled" if effort == 0 else reasoning_effort}')
    return response.choices[0].message.content


# ============================================================
# Prompt Helpers
# ============================================================
def get_search_system_message():
    return {"role": "system", "content": SYSTEM_MESSAGE_SEARCH}


def get_search_user_message(question, history_text=None, step_count=0):
    if history_text:
        content = SEARCH_PROMPT_HIS.format(
            question=question,
            step_count=step_count,
            history=history_text,
        )
    else:
        content = SEARCH_PROMPT_NO_HIS.format(question=question)
    return {"role": "user", "content": content}


# ============================================================
# Action Extraction
# ============================================================
def extract_search_action(text):
    """Extract <search> or <answer> action from model output."""
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None

    search_pattern = r'<search>(.*?)</search>'
    search_match = re.search(search_pattern, text, re.DOTALL)
    search_content = search_match.group(1).strip() if search_match else None

    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else None

    action_type = None
    action_content = None
    if search_content:
        action_type = 'search'
        action_content = search_content
    elif answer_content:
        action_type = 'answer'
        action_content = answer_content

    return {
        'think': think_content,
        'action_type': action_type,
        'action_content': action_content,
        'raw_action': f'<{action_type}>{action_content}</{action_type}>' if action_type else None
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
# Trajectory Generation
# ============================================================
def get_single_trajectory(env, question, ground_truth, data_source, task_idx=0,
                          max_turns=10, show_turn=False, his_len=5,
                          use_local_model=True, ds_model=1, effort=0):
    """
    Generate a single trajectory for one search question.

    ═══════════════════════════════════════════════════════════════
    SEARCH ENV API (compared to WebShop):

    env.reset(kwargs: List[Dict]) -> (obs_list, info_list)
      - kwargs MUST contain: "ground_truth", "question", "data_source"
      - obs_list: the questions themselves (strings)
      - info_list: metadata dicts

    env.step(actions: List[str]) -> (obs_list, reward_list, done_list, info_list)
      - ALWAYS returns exactly 4 lists (WebShop can return 4-6)
      - obs_list: search results (str) if search action, "" if answer action
      - reward_list: float (only meaningful when done=True, 1.0=correct)
      - done_list: bool (True if <answer> found or max_turns reached)
      - info_list: dict with key "won" = bool(done and reward >= 1.0)
    ═══════════════════════════════════════════════════════════════

    Args:
        env: SearchMultiProcessEnv instance (group_n=1, env_num=1)
        question: The question string
        ground_truth: Ground truth answer(s)
        data_source: Data source name
        task_idx: Index for tracking
        max_turns: Maximum number of search/answer turns
        show_turn: Whether to print turn info
        his_len: History window length (-1 for full history)
        use_local_model: Use local model vs deepseek API

    Returns:
        (messages, success_flag, status_msg, seperated_list)
    """
    messages = []
    success_flag = 0
    seperated_list = []

    # ── Reset environment ────────────────────────────────────────
    # SearchMultiProcessEnv.reset() takes a list of kwargs dicts,
    # each containing: ground_truth, question, data_source.
    # It returns (obs_list, info_list) where obs = questions.
    kwargs = [{
        "ground_truth": ground_truth,
        "question": question,
        "data_source": data_source
    }]
    obs_list, info_list = env.reset(kwargs)
    current_obs = obs_list[0]

    # History storage
    history_list = {
        'Action': [],
        'Observation': [],
    }

    # System message
    messages.append(get_search_system_message())

    null_count = 0

    for turn in range(max_turns):
        # ── Build prompt with history ────────────────────────────
        # NOTE: history format uses <search>/<information> tags to match
        # the prompt description in SEARCH_PROMPT_HIS AND the actual
        # env output format (SearchEnv wraps results in <information>).
        if turn == 0:
            user_msg = get_search_user_message(question, history_text=None)
        else:
            # Build history context
            if his_len < 0:
                history_lines = []
                for t_idx in range(turn):
                    action = history_list['Action'][t_idx]
                    obs = history_list['Observation'][t_idx]
                    history_lines.append(f"  Step {t_idx+1}:")
                    history_lines.append(f"    <search>{action}</search>")
                    if obs:
                        # Trim obs to avoid context overflow
                        obs_trimmed = obs[:2000] if len(obs) > 2000 else obs
                        history_lines.append(f"    <information>{obs_trimmed}</information>")
                    history_lines.append("")
                history_text = "\n".join(history_lines)
            else:
                history_lines = []
                start_idx = max(0, turn - his_len)
                for t_idx in range(start_idx, turn):
                    action = history_list['Action'][t_idx]
                    obs = history_list['Observation'][t_idx]
                    history_lines.append(f"  Step {t_idx+1}:")
                    history_lines.append(f"    <search>{action}</search>")
                    if obs:
                        obs_trimmed = obs[:2000] if len(obs) > 2000 else obs
                        history_lines.append(f"    <information>{obs_trimmed}</information>")
                    history_lines.append("")
                history_text = "\n".join(history_lines)

            # Most recent step appended separately
            last_action = history_list['Action'][-1]
            last_obs = history_list['Observation'][-1]
            last_obs_trimmed = last_obs[:2000] if last_obs and len(last_obs) > 2000 else (last_obs or '(empty)')
            last_step_text = (
                f"\n  Step {turn} (most recent):\n"
                f"    <search>{last_action}</search>\n"
                f"    <information>{last_obs_trimmed}</information>\n"
            )

            user_msg = get_search_user_message(
                question, history_text + last_step_text, step_count=turn
            )

        messages.append(user_msg)

        # ── Prepare partial messages (for training data) ─────────
        partial_messages = [
            {"task_idx": task_idx, "turn": turn},
            messages[0],  # system message
            messages[-1],  # current turn user message
        ]

        # ── Model inference ──────────────────────────────────────
        if use_local_model:
            assistant_response = local_model_infer(messages=partial_messages[1:])
        else:
            assistant_response = deepseek(messages=partial_messages[1:],
                                          ds_model=ds_model, effort=effort)

        partial_messages.append({"role": "assistant", "content": assistant_response})
        seperated_list.append({"messages": partial_messages.copy()})

        # ── Extract action from response ─────────────────────────
        result = extract_search_action(assistant_response)
        action_type = result['action_type']
        action_content = result['action_content']

        # ── Step the environment ─────────────────────────────────
        # SearchMultiProcessEnv.step() ALWAYS returns 4 values:
        #   (obs_list, reward_list, done_list, info_list)
        if action_type == 'search':
            env_action = f"<search>{action_content}</search>"
            obs_list, reward_list, done_list, info_list = env.step([env_action])

        elif action_type == 'answer':
            env_action = f"<answer>{action_content}</answer>"
            obs_list, reward_list, done_list, info_list = env.step([env_action])

        else:
            # No valid action — treat as null
            null_count += 1
            history_list['Action'].append('null')
            history_list['Observation'].append(current_obs)

            # Still record the assistant message
            assistant_msg = {
                "role": "assistant",
                "content": assistant_response,
                "action_type": None,
                "action_content": None,
                "think": result['think'],
            }
            if show_turn:
                assistant_msg["turn"] = turn + 1
            messages.append(assistant_msg)

            # Check null termination
            if null_count >= 2:
                status_msg = f"Task {task_idx} exit(all null) at turn {turn + 1}"
                if show_turn:
                    print(status_msg)
                return messages, success_flag, status_msg, seperated_list
            continue

        # ── Unpack step results ──────────────────────────────────
        new_obs = obs_list[0] if obs_list else ''
        new_reward = reward_list[0] if reward_list else 0.0
        new_done = done_list[0] if done_list else False
        new_info = info_list[0] if info_list else {}
        won = new_info.get("won", False)

        # Record in history (store raw content WITHOUT XML tags to avoid
        # double-wrapping when rendering history via <search>{content}</search>)
        history_list['Action'].append(action_content if action_content else 'null')
        history_list['Observation'].append(new_obs)
        current_obs = new_obs

        # Build assistant message for full trajectory
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "action_type": action_type,
            "action_content": action_content,
            "think": result['think'],
            "won": won,
            "reward": new_reward,
        }
        if show_turn:
            assistant_msg["turn"] = turn + 1
        messages.append(assistant_msg)

        # ── Termination conditions ───────────────────────────────
        if action_type == 'answer':
            if won or new_reward >= 1.0:
                success_flag = 1
                status_msg = f"Task {task_idx} SUCCESS at turn {turn + 1}"
            else:
                status_msg = f"Task {task_idx} answered but WRONG at turn {turn + 1}"
            if show_turn:
                print(status_msg)
            break

        if new_done:
            # done=True without answer = max_turns exhausted during search
            status_msg = f"Task {task_idx} max_turns exhausted at turn {turn + 1}"
            if show_turn:
                print(status_msg)
            break

    else:
        status_msg = f"Task {task_idx} out of max turn"
        if show_turn:
            print(status_msg)

    return messages, success_flag, status_msg, seperated_list


# ============================================================
# Data Loading
# ============================================================
def load_search_data(data_path, max_samples=None, start_index=0):
    """Load search data from parquet file.

    Expected columns: question, ground_truth, data_source, env_kwargs
    """
    if not os.path.exists(data_path):
        # Try alternative paths
        alt_path = os.path.join(os.path.dirname(data_path), 'test.parquet')
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Search data not found at {data_path} or {alt_path}")

    print(f"Loading search data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples")

    if max_samples is not None:
        df = df.iloc[start_index:start_index + max_samples]
    else:
        df = df.iloc[start_index:]

    print(f"Using {len(df)} samples (start_index={start_index})")
    return df


# ============================================================
# Main Evaluation
# ============================================================
def evaluate_coldstart_data(output_file, max_samples=100, max_turns=10,
                            show_turn=False, his_len=5, seed=42,
                            start_index=0, save_traj=1,
                            use_local_model=True,
                            data_path=None,
                            ds_model=1, effort=0):
    """
    Generate coldstart trajectories for search task.
    """
    log_file = output_file.replace('.json', '.log')

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

    print(f'Start time: {start_time_str}')
    print(f'log_file: {log_file}')

    # Log configuration
    config_lines = [
        f'Evaluating search coldstart: {max_samples} samples',
        f'max_turns: {max_turns}, his_len: {his_len}, seed: {seed}',
        f'start_index: {start_index}',
        f'use_local_model: {use_local_model}',
        f'search_url: {SEARCH_URL}, topk: {SEARCH_TOPK}',
    ]
    with open(log_file, 'w') as f:
        for line in config_lines:
            f.write(line + '\n')
            print(line)

    # Load data
    data_path = data_path or SEARCH_DATA_PATH
    df = load_search_data(data_path, max_samples=max_samples, start_index=start_index)

    # Build environment
    env_config = OmegaConf.create({
        'max_steps': max_turns,
        'search': {
            'search_url': SEARCH_URL,
            'topk': SEARCH_TOPK,
            'timeout': SEARCH_TIMEOUT,
            'log_requests': SEARCH_LOG_REQUESTS,
        }
    })

    env = build_search_envs(
        seed=seed,
        env_num=1,
        group_n=1,
        is_train=False,
        env_config=env_config,
    )

    print(f'Environment built successfully')

    # Track results
    all_trajectories = []
    success_trajectories = []
    seperated_trajectories = []
    success_indices = []
    success_count = 0
    total_count = 0

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), desc="Generating search coldstart", total=len(df))):
        # Extract question and ground truth
        env_kwargs = row.get('env_kwargs', {})
        if isinstance(env_kwargs, str):
            env_kwargs = json.loads(env_kwargs)

        question = env_kwargs.get('question', row.get('question', ''))
        ground_truth = env_kwargs.get('ground_truth', row.get('ground_truth', ''))
        data_source = env_kwargs.get('data_source', row.get('data_source', 'unknown'))

        if not question:
            print(f"WARNING: Skipping row {idx} - no question found")
            continue

        absolute_idx = start_index + idx

        try:
            trajectory, success_flag, status_msg, seperated_list = get_single_trajectory(
                env=env,
                question=question,
                ground_truth=ground_truth,
                data_source=data_source,
                task_idx=absolute_idx,
                max_turns=max_turns,
                show_turn=show_turn,
                his_len=his_len,
                use_local_model=use_local_model,
                ds_model=ds_model,
                effort=effort,
            )

            all_trajectories.append(trajectory)
            seperated_trajectories.append(seperated_list)
            total_count += 1

            with open(log_file, 'a') as f:
                f.write(status_msg + '\n')

            if success_flag == 1:
                success_trajectories.append(trajectory)
                success_indices.append(absolute_idx)
                success_count += 1
                print(f"\n  *** SUCCESS: {absolute_idx} | {status_msg}")
            elif show_turn:
                print(f"\n  {status_msg}")

        except Exception as e:
            error_msg = f"Error evaluating sample {absolute_idx}: {e}"
            print(f"\n  {error_msg}")
            with open(log_file, 'a') as f:
                f.write(error_msg + '\n')
            import traceback
            traceback.print_exc()
            continue

    # Calculate success rate
    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"\n{'='*60}")
    print(f"Total: {total_count}, Success: {success_count}, Rate: {success_rate:.2%}")
    print(f"{'='*60}")

    # Save results
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

    # Final log
    with open(log_file, 'a') as f:
        f.write(f"\nTotal: {total_count}, Success: {success_count}, Rate: {success_rate:.2%}\n")
        f.write(f"Success indices: {success_indices}\n")
        if save_traj:
            f.write(f"Output saved to {output_file}\n")

    env.close()

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    print(f'Start time: {start_time_str}')
    print(f'End time: {end_time_str}')
    print(f'Elapsed time: {elapsed_time:.1f}s')

    with open(log_file, 'a') as f:
        f.write(f"Start time: {start_time_str}\n")
        f.write(f"End time: {end_time_str}\n")
        f.write(f"Elapsed time: {elapsed_time:.1f}s\n")

    return all_trajectories


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    """
    ================================================================
    Search Coldstart Data Generation
    ================================================================
    REQUIREMENTS:
    1. A running search/retrieval server at SEARCH_URL (default: http://127.0.0.1:8000/retrieve)
       - Start with: bash examples/search/retriever/retrieval_launch.sh
    2. Preprocessed search data at SEARCH_DATA_PATH (default: ~/data/searchR1_processed_direct/test.parquet)
       - Generate with: python examples/data_preprocess/preprocess_search_r1_dataset.py
    3. (Optional) Local model checkpoint for local inference
       - Or use DeepSeek API (use_local_model=False)
    ================================================================
    """
    OUTPUT_BASE_DIR = '/diskpool/home/xuxz/verl-agent/coldstart_genaration_search'
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # ================================================================
    # Configuration - adjust these as needed
    # ================================================================
    USE_LOCAL_MODEL = False       # Set to True to use local model
    DS_MODEL = 1                  # 1=Flash, 2=Pro (only if use_local_model=False)
    EFFORT = 0                    # 0=disabled thinking, 1=high, 2=max
    MAX_TURNS = 10                # Max search/answer turns per question
    HIS_LEN = 5                   # History window (-1 for full history)
    SEED = 42
    SHOW_TURN = True              # Print per-turn status

    # Local model checkpoint path (only if USE_LOCAL_MODEL=True)
    # CHECKPOINT_PATH = '/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5/...'
    # load_local_model(tokenizer_path=CHECKPOINT_PATH, model_path=CHECKPOINT_PATH, show=1)

    # ================================================================
    # Generate coldstart data in chunks
    # ================================================================
    # Each chunk generates trajectories for a range of test samples.
    # Adjust start_index and max_samples to process different chunks.

    CHUNKS = [
        # (start_index, max_samples, description)
        (0,   100, "test_samples_0_99"),
        (100, 100, "test_samples_100_199"),
        (200, 100, "test_samples_200_299"),
    ]

    for start_idx, n_samples, desc in CHUNKS:
        OUTPUT_FILE = get_unique_filename(
            os.path.join(OUTPUT_BASE_DIR, f'search_coldstart_{desc}.json')
        )
        print(f"\n{'='*80}")
        print(f"Generating chunk: {desc} (start={start_idx}, samples={n_samples})")
        print(f"Output: {OUTPUT_FILE}")
        print(f"{'='*80}\n")

        evaluate_coldstart_data(
            output_file=OUTPUT_FILE,
            max_samples=n_samples,
            max_turns=MAX_TURNS,
            show_turn=SHOW_TURN,
            his_len=HIS_LEN,
            seed=SEED,
            start_index=start_idx,
            save_traj=1,
            use_local_model=USE_LOCAL_MODEL,
            ds_model=DS_MODEL,
            effort=EFFORT,
            data_path=SEARCH_DATA_PATH,
        )

    print(f"\n{'='*80}")
    print("All chunks completed!")
    print(f"{'='*80}")
