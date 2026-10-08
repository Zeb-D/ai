"""推理引擎抽象层。

- 业务代码只依赖 InferenceEngine，不直接 import vllm / sglang；
- 通过 factory.build_engine() 按配置动态装配；
- vllm / sglang 采用"延迟导入"，保证未安装推理引擎时本包仍可导入。
"""

from .base import EngineError, GenRequest, GenResult, InferenceEngine, render_chat
from .factory import build_engine

__all__ = [
    "InferenceEngine",
    "GenRequest",
    "GenResult",
    "EngineError",
    "render_chat",
    "build_engine",
]
