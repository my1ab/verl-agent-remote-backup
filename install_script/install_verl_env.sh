#!/bin/bash
# ============================================================
# verl_train 环境安装脚本 (uv 版)
# 基于 pip_list_verl_train.log 整理，按依赖层级分批安装
#
# 用法:
#   conda create -n verl_train python=3.10 -y
#   conda activate verl_train
#   pip install uv   # 首次需要安装 uv
#   bash install_verl_env.sh
#
# 说明:
#   使用 uv 替代 pip，速度更快，依赖解析更可靠。xuyao
#   所有 pip install 替换为 uv pip install。
# ============================================================
set -euo pipefail

echo "=========================================="
echo "  verl_train 环境安装 (uv 版)"
echo "  安装策略：只安装顶层包，依赖自动解析"
echo "=========================================="

# ============================================================
# Step 1: 基础工具
# ============================================================
echo ""
echo "===== [1/7] 基础工具 ====="
uv pip install --upgrade pip setuptools wheel

# ============================================================
# Step 2: PyTorch（GPU 版）
# ============================================================
echo ""
echo "===== [2/7] PyTorch (CUDA 12.4) ====="
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ============================================================
# Step 3: 核心 ML 框架
# ============================================================
echo ""
echo "===== [3/7] 核心 ML 框架 ====="
uv pip install \
    transformers==4.51.1 \
    datasets==4.8.5 \
    accelerate==1.13.0 \
    peft==0.17.0 \
    deepspeed \
    vllm==0.8.5 \
    flash-attn==2.7.4.post1 \
    xformers==0.0.29.post2 \
    sentencepiece \
    tiktoken \
    einops

# ============================================================
# Step 4: 分布式框架
# ============================================================
echo ""
echo "===== [4/7] 分布式框架 ====="
uv pip install \
    "ray[default]>=2.41.0,<=2.50.0" \
    hydra-core \
    omegaconf

# ============================================================
# Step 5: verl 及其直接依赖
# ============================================================
echo ""
echo "===== [5/7] verl 及其依赖 ====="
uv pip install \
    wandb \
    tensorboardX \
    pyarrow>=19.0.0 \
    torchdata \
    tensordict \
    codetiming \
    dill \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    pylatexenc \
    pybind11 \
    packaging>=20.0 \
    qwen-vl-utils[decord] \
    opencv-python-headless \
    diskcache \
    termcolor \
    prettytable \
    nest-asyncio

# ============================================================
# Step 6: 强化学习环境
# ============================================================
echo ""
echo "===== [6/7] 强化学习环境 ====="
uv pip install \
    gym==0.24.0 \
    gymnasium==0.29.1 \
    stable_baselines3 \
    torchrl \
    textworld \
    alfworld==0.4.2

# 下载 spaCy 模型（alfworld 需要）
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download en_core_web_lg

# ============================================================
# Step 7: 安装 verl（本地项目）
# ============================================================
echo ""
echo "===== [7/7] 安装 verl (本地) ====="
uv pip install -e /diskpool/home/xuxz/Code-for-DPEPO

echo ""
echo "=========================================="
echo "  安装完成！"
echo "  运行 uv pip list 验证"
echo "=========================================="
    cloudpickle==3.1.2 \
    msgpack==1.1.2 \
    py4j==0.10.9.9 \
    qwen-vl-utils[decord]==0.0.14 \
    decord==0.6.0 \
    av==17.0.1 \
    ffmpy==1.0.0 \
    pydub==0.25.1 \
    pylatexenc==2.10 \
    gdown==6.0.0 \
    beautifulsoup4==4.11.1 \
    soupsieve==2.8.3 \
    lxml \
    diskcache==5.6.3 \
    termcolor==3.3.0 \
    prettytable==3.17.0 \
    wcwidth==0.7.0

# ============================================================
# 第八步：Web 服务与 API
# ============================================================
echo "===== Step 8: Web services ====="
uv pip install \
    fastapi==0.136.1 \
    starlette==0.52.1 \
    uvicorn==0.46.0 \
    uvloop==0.22.1 \
    httptools==0.7.1 \
    watchfiles==1.1.1 \
    websockets==11.0.3 \
    gunicorn==26.0.0 \
    python-multipart==0.0.27 \
    python-dotenv==1.2.2 \
    email-validator==2.3.0 \
    dnspython==2.8.0 \
    orjson==3.11.8 \
    ujson==5.12.0 \
    httpx==0.28.1 \
    httpcore==1.0.9 \
    h2==4.3.0 \
    hpack==4.1.0 \
    hyperframe==6.1.0 \
    fastapi-cli==0.0.24 \
    typer==0.9.4 \
    shellingham==1.5.4 \
    rich==15.0.0 \
    markdown-it-py==4.0.0 \
    mdurl==0.1.2 \
    Pygments==2.20.0 \
    rich-toolkit==0.19.7 \
    prometheus-fastapi-instrumentator==7.1.0 \
    sentry-sdk==2.58.0 \
    fastapi-cloud-cli==0.17.1 \
    gradio==4.26.0 \
    gradio_client==0.15.1 \
    fsspec \
    aiofiles \
    altair==5.5.0 \
    narwhals==2.20.0 \
    semantic-version==2.10.0 \
    orjson \
    ruff==0.15.12

# ============================================================
# 第九步：spaCy 与 NLP
# ============================================================
echo "===== Step 9: spaCy & NLP ====="
uv pip install \
    spacy==3.8.14 \
    thinc==8.2.5 \
    blis==1.3.3 \
    cymem==2.0.13 \
    murmurhash==1.0.15 \
    preshed==3.0.13 \
    srsly==2.5.3 \
    wasabi==1.1.3 \
    catalogue==2.0.10 \
    confection==0.1.5 \
    spacy-legacy==3.0.12 \
    spacy-loggers==1.0.5 \
    weasel==0.3.4 \
    langcodes==3.5.1 \
    nltk==3.9.4 \
    cleantext==1.1.4 \
    pycountry==26.2.16 \
    airportsdata==20260315

# 下载 spaCy 模型
uv run python -m spacy download en_core_web_sm-3.8.0 --direct
uv run python -m spacy download en_core_web_lg-3.8.0 --direct

# ============================================================
# 第十步：强化学习环境
# ============================================================
echo "===== Step 10: RL environments ====="
uv pip install \
    gym==0.24.0 \
    gym-notices==0.1.0 \
    gymnasium==0.29.1 \
    Farama-Notifications==0.0.6 \
    stable_baselines3==2.6.0 \
    torchrl==0.12.0 \
    tensorboardX \
    cloudpickle \
    msgpack \
    py4j \
    textworld==1.7.0 \
    jericho==3.3.1 \
    fast_downward_textworld==20.6.4 \
    scienceworld==1.2.3 \
    alfworld==0.4.2 \
    nmslib==2.1.2 \
    pyserini==0.17.0 \
    pyjnius==1.7.0 \
    faiss==1.9.0 \
    rank-bm25==0.2.2 \
    thefuzz==0.19.0 \
    RapidFuzz==3.14.5 \
    Levenshtein==0.27.3 \
    python-Levenshtein==0.27.3

# ============================================================
# 第十一步：MLflow 与实验跟踪
# ============================================================
echo "===== Step 11: MLflow & tracking ====="
uv pip install \
    mlflow==3.13.0 \
    mlflow-skinny==3.13.0 \
    mlflow-tracing==3.13.0 \
    alembic==1.18.4 \
    Mako==1.3.12 \
    greenlet==3.5.1 \
    SQLAlchemy==2.0.50 \
    graphene==3.4.3 \
    graphql-core==3.2.11 \
    graphql-relay==3.2.0 \
    sqlparse==0.5.5 \
    prometheus-flask-exporter \
    Flask==2.1.2 \
    Werkzeug==2.1.0 \
    itsdangerous==2.2.0 \
    flask-cors==6.0.2 \
    gunicorn \
    docker==7.1.0 \
    protobuf==6.33.6 \
    proto-plus==1.28.0 \
    googleapis-common-protos==1.74.0 \
    google-api-core==2.31.0 \
    google-auth==2.53.0 \
    pyasn1==0.6.3 \
    pyasn1_modules==0.4.2 \
    cachetools==7.1.0 \
    opentelemetry-api==1.42.1 \
    opentelemetry-sdk==1.42.1 \
    opentelemetry-exporter-otlp==1.26.0 \
    opentelemetry-exporter-otlp-proto-common==1.26.0 \
    opentelemetry-exporter-otlp-proto-grpc==1.26.0 \
    opentelemetry-exporter-otlp-proto-http==1.26.0 \
    opentelemetry-proto==1.42.1 \
    opentelemetry-semantic-conventions==0.63b1 \
    opentelemetry-semantic-conventions-ai==0.4.13 \
    opentelemetry-exporter-prometheus==0.63b1 \
    opencensus==0.11.4 \
    opencensus-context==0.1.3 \
    databricks-sdk==0.114.0 \
    prometheus_client \
    sentry-sdk \
    tomlkit==0.12.0

# ============================================================
# 第十二步：其他工具与杂项
# ============================================================
echo "===== Step 12: Other utilities ====="
uv pip install \
    pytest==8.3.5 \
    pluggy==1.6.0 \
    iniconfig==2.3.0 \
    exceptiongroup==1.3.1 \
    pyzmq==27.1.0 \
    cffi==2.0.0 \
    pycparser==3.0 \
    cryptography==47.0.0 \
    pyOpenSSL==26.1.0 \
    Brotli==1.2.0 \
    Deprecated==1.3.1 \
    wrapt==2.1.2 \
    hf_xet \
    certifi \
    idna \
    urllib3 \
    chardet \
    nest-asyncio==1.6.0 \
    prompt_toolkit==3.0.52 \
    wcwidth \
    sortedcontainers==2.4.0 \
    trio==0.33.0 \
    outcome==1.3.0.post0 \
    wsproto==1.3.2 \
    trio-websocket==0.12.2 \
    selenium==4.2.0 \
    PySocks==1.7.1 \
    python-json-logger==4.1.0 \
    mementos==1.3.1 \
    pyvers==0.2.2 \
    rignore==0.7.6 \
    annotated-doc==0.0.4 \
    colorful==0.5.8 \
    commonmark==0.9.1 \
    env==0.1.0 \
    hashids==1.3.1 \
    huey==3.0.1 \
    lightgbm==4.6.0 \
    llvmlite==0.44.0 \
    numba==0.61.2 \
    modelscope==1.37.0 \
    more-itertools==11.0.2 \
    nvidia-* \
    onnxruntime==1.25.1 \
    opencv-python-headless==4.13.0.92 \
    pandas \
    pyarrow \
    requests-mock==1.12.1 \
    scikit-learn \
    skops==0.14.0 \
    spacy \
    stable-baselines3 \
    tensorboardX \
    tensordict \
    torchrl \
    train==0.0.5 \
    transformers \
    typer-slim==0.24.0 \
    tzdata==2026.2 \
    verl \
    virtualenv \
    vllm \
    wandb \
    xformers \
    xxhash \
    yarl \
    zipp

# ============================================================
# 第十三步：安装 verl（本地项目）
# ============================================================
echo "===== Step 13: Install verl (local) ====="
# 假设 verl 源码在 /diskpool/home/xuxz/Code-for-DPEPO
uv pip install -e /diskpool/home/xuxz/Code-for-DPEPO

echo "===== Installation complete! ====="
echo "Run 'uv pip list' to verify."
