# 翻译大模型私有化部署 · 服务实现

本目录是架构设计文档 [`翻译大模型私有化部署-架构设计文档.md`](../翻译大模型私有化部署-架构设计文档.md) 的落地实现。

## 合规红线（反需求 R1）

- 服务端在**本地加载开源模型权重**（来自本地缓存或磁盘）；
- 仅支持**进程内嵌**推理引擎（`vllm` / `sglang` / `llamacpp`），无任何网络跳转；
- 代码中**不存在任何 HTTP 客户端**（无本地代理、无云端转发）；
- 全程**可离线运行**（可开启 `HF_HUB_OFFLINE=1`）。

## 引擎与运行环境

| 引擎 | 操作系统 | 加速 | 权重格式 | 吞吐 | 适用 |
| --- | --- | --- | --- | --- | --- |
| `vllm`（默认） | Linux x86_64 | NVIDIA CUDA | HF / AWQ / FP8 | **高** | 生产在线服务 |
| `sglang` | Linux x86_64 | NVIDIA CUDA | HF / AWQ / FP8 | **高** | 高前缀复用场景 |
| `llamacpp` | **Linux / macOS / Windows** | CPU 或 **Metal** / CUDA | **GGUF** | 中低 | 开发机、边缘、无 NVIDIA GPU |

硬性要求：

| 项 | 要求 |
| --- | --- |
| Python | 3.10 ~ 3.12（推荐 3.12） |
| GPU（vllm/sglang） | NVIDIA 显卡 + 匹配驱动（CUDA 12.x），**无 GPU 无法运行** |
| 显存（vllm/sglang） | 1.5B：AWQ 4bit ≈ 2~3 GB；FP16 ≈ 4 GB |
| macOS（vllm/sglang） | ❌ 不可运行（无预编译 wheel） |
| macOS（llamacpp） | ✅ 可运行（CPU 或 Metal 加速） |

> 在 macOS 上执行 `pip install vllm` 会回退到源码编译并失败（日志会出现
> `VLLM_TARGET_DEVICE automatically set to cpu due to macOS`）。macOS 请使用 `llamacpp`。

## 目录结构

```
app/
  config.py                 # 配置（pydantic）
  main.py                   # FastAPI 入口
  api/                      # HTTP 接口（业务 + OpenAI 兼容）
  engine/                   # 推理引擎抽象 + vllm / sglang / llamacpp 实现 + 工厂
  translation/              # Prompt / 术语 / 后处理 / 编排
  infra/                    # 缓存 / 指标 / 日志 / 限流 / 微批
configs/                    # model.yaml / serve.yaml / prompts / glossary
scripts/                    # 下载模型 / 启动 / 压测
docker/                     # 应用镜像 + compose（单容器内嵌引擎）
```

## 快速开始 A：Linux + NVIDIA GPU（vllm / sglang）

```bash
cd deploy/translation-serving
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm                 # 或 pip install "sglang[all]"

python scripts/download_model.py # 默认落到 HF 缓存 ~/.cache/huggingface/hub
python app/main.py               # 或 bash scripts/start_gateway.sh
```

`configs/model.yaml` 默认 `engine: vllm`、`model_path: Qwen/Qwen2.5-1.5B-Instruct`（HF 仓库 ID，
从默认缓存解析），通常无需修改。

## 快速开始 B：macOS / 无 GPU（llama.cpp + GGUF）

```bash
cd deploy/translation-serving
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install llama-cpp-python          # macOS 会启用 Metal；CPU 亦可

# 1) 下载 GGUF 量化权重（只取需要的量化档，节省空间）
python scripts/download_model.py \
    --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    --include "*q4_k_m.gguf"

# 2) 将 configs/model.yaml 改为：
#      engine: llamacpp
#      model_path: <上一步打印的快照目录>/qwen2.5-1.5b-instruct-q4_k_m.gguf
#      n_gpu_layers: -1     # macOS Metal 全部层加速；纯 CPU 用 0
#    或直接用环境变量覆盖：
export TS_ENGINE=llamacpp
export TS_MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/<rev>/qwen2.5-1.5b-instruct-q4_k_m.gguf"
export TS_N_GPU_LAYERS=-1

# 3) 启动
python app/main.py
```

调用：

```bash
curl -s http://127.0.0.1:8080/v1/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world.","source_lang":"en","target_lang":"zh"}'
```

## 引擎切换

修改 [`configs/model.yaml`](configs/model.yaml) 的 `engine` 字段（或环境变量 `TS_ENGINE`）：

| 取值 | 说明 |
| --- | --- |
| `vllm` | vLLM 进程内嵌（默认，成熟稳定，Linux+CUDA） |
| `sglang` | SGLang 进程内嵌（RadixAttention，前缀复用吞吐更优，Linux+CUDA） |
| `llamacpp` | llama.cpp 进程内嵌（GGUF，跨平台，CPU / Metal / CUDA） |

三者均通过 [`app/engine/factory.py`](app/engine/factory.py) 装配，业务代码零改动。

## 默认路径说明

| 项 | 默认值 |
| --- | --- |
| `download_model.py` 下载位置 | `~/.cache/huggingface/hub`（HF 官方缓存） |
| `model_path`（vllm/sglang） | HF 仓库 ID，从默认缓存解析 |
| `model_path`（llamacpp） | 需填本地 `.gguf` **文件**路径 |
| 缓存自定义 | `HF_HOME=/data/hf` → `/data/hf/hub`；或 `HF_HUB_CACHE=/data/hub` |

> **不同引擎的权重产物格式不同**：vLLM / SGLang 使用 HF **目录**（或 `-AWQ` / FP8 量化仓库），
> llama.cpp 使用**单个 `.gguf` 文件**。切换 `engine` 时请同步更换 `model_path`
> （详见架构文档 §2.2.6「模型产物差异」）。

## 容器部署（Linux GPU 主机）

```bash
docker compose -f docker/docker-compose.yml up
```

容器内通过 `HF_HOME=/root/.cache/huggingface` 复用宿主缓存，并设置
`HF_HUB_OFFLINE=1` 强制离线；宿主机缓存目录可用 `HF_CACHE` 覆盖。
使用 SGLang 时，将 [`docker/Dockerfile.gateway`](docker/Dockerfile.gateway) 基础镜像替换为
`lmsysorg/sglang` 并设置 `TS_ENGINE=sglang`；使用 llama.cpp 时基础镜像可换成
`python:3.11-slim` 并 `pip install llama-cpp-python`，同时设置 `TS_ENGINE=llamacpp`。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/translate` | 业务翻译（支持单/多段文本） |
| POST | `/v1/translate/stream` | SSE 流式翻译 |
| POST | `/v1/chat/completions` | OpenAI 兼容（含流式） |
| GET | `/healthz` | 健康检查（含引擎/缓存/术语状态） |
| GET | `/metrics` | Prometheus 指标 |
