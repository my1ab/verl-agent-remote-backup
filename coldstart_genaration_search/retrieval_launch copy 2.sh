
# 数据存储根目录 faiss向量索引文件 语料库文件 语料库文件路径 文本名称+路径
save_path=$HOME/data/searchR1
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

export CUDA_VISIBLE_DEVICES=0

echo "GPU: $CUDA_VISIBLE_DEVICES"
# LOG_FILE="retrieval_server.log"
LOG_FILE="coldstart_genaration_search/retrieval_server.log"

# 使用 setsid 将服务器进程放入新会话组，彻底脱离终端控制
# Ctrl+C 等终端信号不会传递给它，即使启动它的 shell 退出也不受影响
setsid python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --port 8000 \
  > $LOG_FILE 2>&1 &

PID=$!
echo "Server started. PID: $PID"
echo "logging in $LOG_FILE"

# 等待服务器启动（检查端口）
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if ss -tlnp | grep -q ":8000"; then
        echo "Server is ready on port 8000 (attempt $i)"
        break
    fi
    sleep 1
done

echo "Server is running in background. Use 'kill $PID' to stop it."
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"

# 不再 tail -f 阻塞，避免 Ctrl+C 影响后台进程