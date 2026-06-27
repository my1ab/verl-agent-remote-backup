#!/bin/bash
# ============================================================
# verl_train 环境完整安装脚本
# 基于 pip_list_verl_train.log 整理，按依赖层级分批安装
# ============================================================
set -euo pipefail

# 用法: bash install_verl_env.sh
# 建议在 conda 环境创建后运行

# ============================================================
# 第〇步：基础环境准备（conda create 时已完成）
# ============================================================
# conda create -n verl_train python=3.10 -y
# conda activate verl_train

# ============================================================
# 第一步：核心基础库（无依赖或极少依赖）
# ============================================================
echo "===== Step 1: Core base libraries ====="
pip install \
    setuptools==68.0.0 \
    wheel==0.47.0 \
    pip==26.1.2 \
    numpy==1.26.4 \
    ninja==1.13.0 \
    pybind11==3.0.4 \
    Cython==3.2.4 \
    packaging==26.0 \
    typing_extensions==4.15.0 \
    filelock==3.25.2 \
    six==1.17.0 \
    psutil==7.2.2 \
    tqdm==4.67.3 \
    pyyaml==6.0.2 \
    click==8.3.3 \
    Jinja2==3.1.6 \
    MarkupSafe==2.1.5 \
    colorama==0.4.6 \
    certifi==2026.4.22 \
    charset-normalizer==2.0.12 \
    idna==3.13 \
    urllib3==1.26.20 \
    requests==2.33.1 \
    pyparsing==3.3.2 \
    python-dateutil==2.9.0.post0 \
    pytz==2026.1.post1 \
    zipp==3.23.1 \
    importlib_metadata==8.0.0 \
    importlib_resources==6.5.2 \
    platformdirs==4.10.0 \
    distlib==0.4.1 \
    sniffio==1.3.1 \
    anyio==4.13.0 \
    h11==0.16.0

# ============================================================
# 第二步：科学计算与 ML 框架
# ============================================================
echo "===== Step 2: Scientific computing & ML frameworks ====="
pip install \
    scipy==1.17.1 \
    scikit-learn==1.8.0 \
    joblib==1.5.3 \
    threadpoolctl==3.6.0 \
    pandas==2.2.3 \
    matplotlib==3.10.9 \
    contourpy==1.3.3 \
    cycler==0.12.1 \
    fonttools==4.62.1 \
    kiwisolver==1.5.0 \
    pillow==10.4.0 \
    sympy==1.13.1 \
    mpmath==1.3.0 \
    networkx==3.6.1

# ============================================================
# 第三步：PyTorch 及相关（GPU 版本）
# ============================================================
echo "===== Step 3: PyTorch & GPU libs ====="
# 注意：torch 2.6.0+cu124 建议从官方源安装
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install \
    xformers==0.0.29.post2 \
    triton==3.2.0 \
    flash-attn==2.7.4.post1 \
    nvidia-nccl-cu12==2.21.5 \
    nvidia-cublas-cu12==12.4.5.8 \
    nvidia-cuda-runtime-cu12==12.4.127 \
    nvidia-cuda-nvrtc-cu12==12.4.127 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cufft-cu12==11.2.1.3 \
    nvidia-curand-cu12==10.3.5.147 \
    nvidia-cusolver-cu12==11.6.1.9 \
    nvidia-cusparse-cu12==12.3.1.170 \
    nvidia-cusparselt-cu12==0.6.2 \
    nvidia-nvjitlink-cu12==12.4.127 \
    nvidia-nvtx-cu12==12.4.127 \
    nvidia-cuda-cupti-cu12==12.4.127

# ============================================================
# 第四步：HuggingFace 生态
# ============================================================
echo "===== Step 4: HuggingFace ecosystem ====="
pip install \
    transformers==4.51.1 \
    tokenizers==0.21.4 \
    safetensors==0.7.0 \
    accelerate==1.13.0 \
    datasets==4.8.5 \
    huggingface_hub==0.36.2 \
    peft==0.17.0 \
    einops==0.8.2 \
    regex==2026.4.4 \
    sentencepiece==0.2.1 \
    tiktoken==0.12.0 \
    fsspec==2026.2.0 \
    dill==0.4.1 \
    multiprocess==0.70.19 \
    xxhash==3.7.0 \
    aiohttp==3.13.5 \
    aiohappyeyeballs==2.6.1 \
    aiosignal==1.4.0 \
    frozenlist==1.8.0 \
    multidict==6.7.1 \
    yarl==1.23.0 \
    propcache==0.4.1 \
    hf-xet==1.4.3

# ============================================================
# 第五步：vLLM 及相关推理库
# ============================================================
echo "===== Step 5: vLLM & inference ====="
pip install \
    vllm==0.8.5 \
    outlines==0.1.11 \
    outlines_core==0.1.26 \
    interegular==0.3.3 \
    lm-format-enforcer==0.10.12 \
    llguidance==0.7.30 \
    partial-json-parser==0.2.1.1.post7 \
    lark==1.2.2 \
    msgspec==0.21.1 \
    xgrammar==0.1.18 \
    compressed-tensors==0.9.3 \
    gguf==0.18.0 \
    fastar==0.11.0 \
    depyf==0.18.0 \
    py-cpuinfo==9.0.0 \
    openai==2.33.0 \
    jiter==0.14.0 \
    pydantic==2.13.3 \
    pydantic_core==2.46.3 \
    pydantic-settings==2.14.0 \
    pydantic-extra-types==2.11.1 \
    annotated-types==0.7.0 \
    typing-inspection==0.4.2

# ============================================================
# 第六步：Ray（分布式框架）
# ============================================================
echo "===== Step 6: Ray ====="
pip install \
    ray[default]==2.47.1 \
    py-spy==0.4.2 \
    smart-open==6.4.0 \
    virtualenv==21.4.2 \
    aiofiles==23.2.1 \
    aiohttp-cors==0.8.1 \
    prometheus_client==0.25.0 \
    redis \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    referencing==0.37.0 \
    rpds-py==0.30.0

# ============================================================
# 第七步：verl 及其直接依赖
# ============================================================
echo "===== Step 7: verl ====="
pip install \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    antlr4-python3-runtime==4.9.3 \
    codetiming==1.4.0 \
    wandb==0.27.1 \
    pyarrow==24.0.0 \
    tensorboardX==2.6.5 \
    torchdata==0.11.0 \
    tensordict==0.12.4 \
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
pip install \
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
pip install \
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
python -m spacy download en_core_web_sm-3.8.0 --direct
python -m spacy download en_core_web_lg-3.8.0 --direct

# ============================================================
# 第十步：强化学习环境
# ============================================================
echo "===== Step 10: RL environments ====="
pip install \
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
pip install \
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
pip install \
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
pip install -e /diskpool/home/xuxz/Code-for-DPEPO

echo "===== Installation complete! ====="
echo "Run 'pip list' to verify."
