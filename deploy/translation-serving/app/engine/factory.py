"""引擎工厂。

仅装配"进程内嵌"的真实推理引擎（vLLM / SGLang / llama.cpp）：服务端在本地加载开源
模型权重，不存在任何把请求转发到外部（云厂商）的 HTTP 客户端，从架构上满足反需求 R1。
"""

from __future__ import annotations

from ..config import ModelConfig
from .base import EngineError, InferenceEngine
from .llamacpp_engine import LlamaCppEngine
from .sglang_engine import SGLangEngine
from .vllm_engine import VLLMEngine


def build_engine(cfg: ModelConfig) -> InferenceEngine:
    """按配置构建推理引擎。"""
    if cfg.engine == "vllm":
        return VLLMEngine(cfg)
    if cfg.engine == "sglang":
        return SGLangEngine(cfg)
    if cfg.engine == "llamacpp":
        return LlamaCppEngine(cfg)
    raise EngineError(f"未知引擎: {cfg.engine}")
