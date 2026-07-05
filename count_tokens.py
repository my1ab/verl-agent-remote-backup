"""
统计 coldstart_genaration_search/result_search/search_coldstart_30_39_3_seperated.json
中每个 messages 的 token 数量。结果同时输出到终端和 txt 文件。

数据格式说明:
  data = [task1, task2, ...]          # 10 个任务
  task = [turn1, turn2, ...]          # 每个任务多个回合
  turn = {"messages": [...]}          # 每个回合的 messages 列表
  messages[0] = {"task_idx":.., "turn":..}   # 元信息
  messages[1..3] = {"role":.., "content":..} # system/user/assistant

用法: python count_tokens.py <文件路径>
"""

import json
import os
import sys

try:
    import tiktoken
except ImportError:
    print("请先安装 tiktoken: pip install tiktoken")
    sys.exit(1)


def log(lines, out_file=None):
    for line in lines:
        print(line)
        if out_file:
            out_file.write(line + "\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python count_tokens.py <文件路径>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.path.dirname(__file__), file_path)
    file_path = os.path.normpath(file_path)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(os.path.dirname(__file__), f"count_tokens_{base_name}.txt")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        sys.exit(1)

    print(f"正在读取: {file_path}\n")
    encoding = tiktoken.get_encoding("cl100k_base")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as out_f:
        total_tokens_all = 0
        total_messages_all = 0
        task_stats = []

        # data: list of tasks, each task is a list of turns
        for task_idx, task in enumerate(data):
            task_tokens = 0
            task_msgs = 0
            # 提前从第一个 turn 中获取原始 task_idx
            first_task_idx = "?"
            if task and len(task) > 0 and "messages" in task[0] and len(task[0]["messages"]) > 0:
                first_task_idx = task[0]["messages"][0].get("task_idx", "?")

            log([f"{'='*60}", f"Task {task_idx} (task_idx={first_task_idx})", f"{'='*60}"], out_f)

            # task: list of turns, each turn is {"messages": [...]}
            for turn_entry in task:
                msgs = turn_entry["messages"]
                # msgs[0] = {"task_idx":.., "turn":..} (元信息)
                # msgs[1..3] = {"role":.., "content":..}

                # 跳过 msgs[0] 元信息，只统计 role/content 消息
                for msg in msgs:
                    if "role" not in msg:
                        continue
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    msg_tokens = len(encoding.encode(role + "\n" + content))
                    task_tokens += msg_tokens
                    task_msgs += 1

                    turn_info = msg.get("turn", "")
                    extra = f" [turn={turn_info}]" if turn_info and role == "assistant" else ""
                    preview = content[:60].replace("\n", " ")
                    log([f"  [{role:>9}]  tokens={msg_tokens:>6}  | {preview}...{extra}"], out_f)

            total_tokens_all += task_tokens
            total_messages_all += task_msgs
            task_stats.append((task_idx, first_task_idx, task_tokens, task_msgs))
            log([f"  >> Task {task_idx} (task_idx={first_task_idx}) 总计: {task_tokens} tokens, {task_msgs} messages\n"], out_f)

        # ===== 汇总 =====
        log([f"\n{'='*60}", f"汇总统计", f"{'='*60}"], out_f)
        log([f"输入文件:              {file_path}"], out_f)
        log([f"总任务数:              {len(task_stats)}"], out_f)
        log([f"总 messages 数:        {total_messages_all}"], out_f)
        log([f"总 tokens:             {total_tokens_all}"], out_f)
        if task_stats:
            log([f"平均 tokens/任务:       {total_tokens_all / len(task_stats):.1f}"], out_f)
        if total_messages_all > 0:
            log([f"平均 tokens/message:    {total_tokens_all / total_messages_all:.1f}"], out_f)

        log([f"\n各任务 token 数排序 (从高到低):"], out_f)
        sorted_stats = sorted(task_stats, key=lambda x: x[2], reverse=True)
        for task_idx, orig_idx, tokens, msgs in sorted_stats:
            log([f"  Task {task_idx} (task_idx={orig_idx}): {tokens} tokens, {msgs} messages"], out_f)

        # ===== 按 role 统计 =====
        log([f"\n按 role 统计 tokens:"], out_f)
        role_tokens = {}
        role_counts = {}
        for task in data:
            for turn_entry in task:
                for msg in turn_entry["messages"]:
                    if "role" not in msg:
                        continue
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    t = len(encoding.encode(role + "\n" + content))
                    role_tokens[role] = role_tokens.get(role, 0) + t
                    role_counts[role] = role_counts.get(role, 0) + 1

        for role in sorted(role_tokens.keys()):
            log([f"  {role:>12}: {role_tokens[role]:>8} tokens ({role_counts[role]}条)"], out_f)

        log([f"\n结果已保存到: {output_path}"], out_f)


if __name__ == "__main__":
    main()
