#!/bin/bash
# Coldstart WebShop Test Script

set -e

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
            exit 1
            ;;
    esac
done

if [ $USE_REMOTE -eq 1 ]; then
    BASE_PATH="/diskpool/home/xuxz/verl-agent"
else
    BASE_PATH="/home/dpepo/verl-agent"
fi

# mkdir -p $BASE_PATH/coldstart_test/output
export PYTHONPATH=$BASE_PATH:$PYTHONPATH

# 设定gpu编号
export CUDA_VISIBLE_DEVICES=7
echo "GPU: $CUDA_VISIBLE_DEVICES"

# python3 $BASE_PATH/coldstart_test/coldstart_para_his_test.py 
# nohup python3 $BASE_PATH/coldstart_test/coldstart_para_his_test.py  \
#     > Webshop_generation.log 2>&1 &
# nohup python3 $BASE_PATH/coldstart_test/coldstart_para_his_test_7B.py
#     > Webshop_generation.log 2>&1 &
# tail -f -n 10 Webshop_generation.log
python3 $BASE_PATH/coldstart_test/coldstart_para_his_test_7B.py

# tail -f -n 10 Parallel-WebShop-1.5B-Epoch5_new.log

# echo "Test started. PID: $!"
echo "Test ended. PID: $!"
echo "GPU: $CUDA_VISIBLE_DEVICES"
# echo "Log: $BASE_PATH/coldstart_test/coldstart_test.log"
# tail -f -n 10 $BASE_PATH/coldstart_test/coldstart_test.log