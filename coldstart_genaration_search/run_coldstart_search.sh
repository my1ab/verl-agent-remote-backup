#!/bin/bash
# Coldstart Search Task - Data Generation Script
# 启动 coldstart_search.py 以生成 Search 任务的冷启动轨迹数据

set -e

# ================================================================
# 环境选择: remote(服务器) / local(本地)
# ================================================================
USE_REMOTE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote|-r)
            USE_REMOTE=1
            shift
            ;;
        --local|-l)
            USE_REMOTE=0
            shift
            ;;
        *)
            echo "Usage: $0 [--remote/-r | --local/-l]"
            exit 1
            ;;
    esac
done

if [ $USE_REMOTE -eq 1 ]; then
    BASE_PATH="/diskpool/home/xuxz/verl-agent"
else
    BASE_PATH="/home/dpepo/verl-agent"
fi

# ================================================================
# 环境变量
# ================================================================
export PYTHONPATH=$BASE_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1

# ================================================================
# GPU 设置
# 使用 DeepSeek API (默认) 时不需要 GPU, 注释掉即可。
# 使用本地模型时 (USE_LOCAL_MODEL=True) 需设置 GPU。
# ================================================================
# export CUDA_VISIBLE_DEVICES=0

# ================================================================
# 日志文件
# ================================================================
LOG_FILE="coldstart_search_gen.log"
echo "=========================================="
echo "Search Coldstart Data Generation"
echo "Base:  $BASE_PATH"
echo "Log:   $LOG_FILE"
echo "GPU:   ${CUDA_VISIBLE_DEVICES:-'(none, using API)'}"
echo "=========================================="

# ================================================================
# 启动生成脚本
# ================================================================
# 前置依赖:
#   1. 检索服务: bash examples/search/retriever/retrieval_launch.sh
#   2. 预处理数据: python examples/data_preprocess/preprocess_search_r1_dataset.py
# ================================================================

nohup python3 $BASE_PATH/coldstart_genaration_search/coldstart_search.py \
    > $LOG_FILE 2>&1 &

echo "Started. PID: $!"
echo "Log: $LOG_FILE"
echo ""
echo "监控日志: tail -f $LOG_FILE"

# 自动开始查看日志
tail -f $LOG_FILE

echo ""
echo "Generation ended."
echo "Log: $LOG_FILE"
