
# 数据存储根目录 faiss向量索引文件 语料库文件 语料库文件路径 文本名称+路径
save_path=$HOME/data/searchR1
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

export CUDA_VISIBLE_DEVICES=5
echo "GPU: $CUDA_VISIBLE_DEVICES"
LOG_FILE="retrieval_server.log"

# 启动http服务器 加载索引+语料库 接收查询文本并返回最相似的文本片段
nohup python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --port 8000 \
  > $LOG_FILE 2>&1 &

echo "Server started. PID: $!"
# disown - 将该后台进程从 shell 作业表中移除，脚本退出时不再向它发信号
# disown
tail -f $LOG_FILE

echo "Server ended."
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"