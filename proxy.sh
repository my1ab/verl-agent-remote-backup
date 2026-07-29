# export HTTP_PROXY=http://127.0.0.1:7890
# export HTTPS_PROXY=http://127.0.0.1:7890
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
# export NO_PROXY=localhost,127.0.0.1
# export no_proxy=localhost,127.0.0.1

export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1

# hf镜像快速下载  也可使用modelscope
# export HF_ENDPOINT=https://hf-mirror.com
# unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy

# echo "HTTPS_PROXY=$HTTPS_PROXY"
#echo "=== 环境变量检查 ===" && echo "HTTP_PROXY=$HTTP_PROXY" && echo "HTTPS_PROXY=$HTTPS_PROXY" && echo "NO_PROXY=$NO_PROXY" && echo "=== 代理连通性测试 ==="