"""配置层：读取 configs/*.yaml 与环境变量，产出强类型 Settings。

设计要点：
- 使用 pydantic v2 做校验，避免手写解析；
- 不依赖 pydantic-settings，降低镜像依赖面；
- 支持少量关键环境变量覆盖，便于容器化部署（12-factor）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# 仅支持进程内嵌的推理引擎；服务端在本地加载模型权重，无任何外部 HTTP 客户端。
#   vllm / sglang  —— Linux + NVIDIA GPU(CUDA)
#   llamacpp       —— 跨平台（Linux/macOS/Windows，CPU 或 Metal/CUDA），GGUF 权重
EngineName = Literal["vllm", "sglang", "llamacpp"]


def _expand(value: str) -> str:
    """展开 ~ 与环境变量，便于直接填写缓存目录路径。"""
    return os.path.expandvars(os.path.expanduser(value))


class ModelConfig(BaseModel):
    """模型与推理引擎配置（对应 configs/model.yaml）。"""

    engine: EngineName = "vllm"
    # vLLM/SGLang：填 HF 仓库 ID（默认缓存解析）或本地目录；llama.cpp：填本地 .gguf 文件路径
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # 仅 vllm/sglang 生效；llama.cpp 的量化已内嵌于 GGUF，此项被忽略
    quantization: Optional[str] = None
    tp_size: int = 1
    gpu_mem_util: float = 0.90
    max_model_len: int = 4096
    prefix_caching: bool = True
    trust_remote_code: bool = True

    # ---- llama.cpp 专用（engine=llamacpp 时生效）----
    # 卸载到 GPU 的层数：0=纯 CPU；-1=全部层（Metal/CUDA）；N=前 N 层
    n_gpu_layers: int = 0
    # 线程数；null 表示自动
    n_threads: Optional[int] = None
    # 聊天模板覆盖；一般留空由 GGUF 内嵌模板决定，必要时填 "qwen"
    chat_format: Optional[str] = None

    # 生成默认参数
    default_temperature: float = 0.0
    default_max_tokens: int = 1024

    @field_validator("model_path")
    @classmethod
    def _expand_model_path(cls, value: str) -> str:
        expanded = _expand(value)
        # 仅对本地路径展开；HF 仓库 ID（如 "Qwen/Qwen2.5-1.5B-Instruct"）保持原样
        if expanded.startswith(("/", "~", ".")) or os.path.sep in expanded:
            return expanded
        return value


class CacheConfig(BaseModel):
    enabled: bool = True
    l1_size: int = 4096
    l2_redis: Optional[str] = None
    ttl: int = 86400


class RateLimitConfig(BaseModel):
    enabled: bool = True
    qps: float = 200.0
    burst: int = 400


class BatchConfig(BaseModel):
    enabled: bool = False
    max_batch_size: int = 16
    max_wait_ms: int = 10


class ServeConfig(BaseModel):
    """服务层配置（对应 configs/serve.yaml）。"""

    host: str = "0.0.0.0"
    port: int = 8080
    max_concurrency: int = 64
    log_level: str = "INFO"
    config_dir: str = "configs"
    glossary_path: Optional[str] = "configs/glossary.yaml"

    cache: CacheConfig = Field(default_factory=CacheConfig)
    ratelimit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    batching: BatchConfig = Field(default_factory=BatchConfig)

    @field_validator("glossary_path")
    @classmethod
    def _expand_glossary_path(cls, value: Optional[str]) -> Optional[str]:
        return _expand(value) if value else value


class Settings(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)

    @property
    def config_dir(self) -> Path:
        return Path(self.serve.config_dir)

    @property
    def prompts_dir(self) -> Path:
        return self.config_dir / "prompts"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_env_overrides(model_data: dict, serve_data: dict) -> None:
    """允许通过环境变量覆盖少量关键项（容器友好）。"""
    if v := os.getenv("TS_ENGINE"):
        model_data["engine"] = v
    if v := os.getenv("TS_MODEL_PATH"):
        model_data["model_path"] = v
    if v := os.getenv("TS_QUANTIZATION"):
        model_data["quantization"] = v or None
    if v := os.getenv("TS_TP_SIZE"):
        model_data["tp_size"] = int(v)
    if v := os.getenv("TS_N_GPU_LAYERS"):
        model_data["n_gpu_layers"] = int(v)
    if v := os.getenv("TS_HOST"):
        serve_data["host"] = v
    if v := os.getenv("TS_PORT"):
        serve_data["port"] = int(v)
    if v := os.getenv("TS_LOG_LEVEL"):
        serve_data["log_level"] = v
    if v := os.getenv("TS_CONFIG_DIR"):
        serve_data["config_dir"] = v
    if v := os.getenv("TS_L2_REDIS"):
        serve_data.setdefault("cache", {})["l2_redis"] = v or None


def load_settings(config_dir: str | os.PathLike | None = None) -> Settings:
    """从 config_dir 加载 model.yaml + serve.yaml，并应用环境变量覆盖。"""
    base = Path(config_dir or os.getenv("TS_CONFIG_DIR", "configs"))
    model_data = _read_yaml(base / "model.yaml")
    serve_data = _read_yaml(base / "serve.yaml")
    # config_dir 以实际入参为准，避免 yaml 内外不一致
    serve_data["config_dir"] = str(base)
    _apply_env_overrides(model_data, serve_data)
    return Settings(model=ModelConfig(**model_data), serve=ServeConfig(**serve_data))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置。"""
    return load_settings()


def reset_settings_cache() -> None:
    """清除配置缓存。"""
    get_settings.cache_clear()
