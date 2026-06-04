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
export PYTHONUNBUFFERED=1

# 设定gpu编号
LOG_FILE="Webshop_test_e3.5_hislen8_v2.log"
export CUDA_VISIBLE_DEVICES=5
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"

# python3 $BASE_PATH/coldstart_test/coldstart_para_his_test.py 
nohup python3 $BASE_PATH/coldstart_test_new/coldstart_para_his_test_1.5B_hislen8_epoch3.5_v2.py \
    > $LOG_FILE 2>&1 &
echo "Test started. PID: $!"
tail -f $LOG_FILE

echo "Test ended."
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"