"""llama.cpp 进程内推理引擎（GGUF 权重）。

定位与特点：
- **跨平台**：Linux / macOS(Apple Silicon, Metal) / Windows 均可运行，是唯一
  "无需 NVIDIA GPU 也能跑"的引擎，适合开发机、边缘设备与低配部署；
- **进程内嵌**：`llama_cpp.Llama` 直接加载本地 `.gguf`，不经过任何 HTTP 客户端
  （满足反需求 R1）；
- **量化内嵌**：GGUF 文件自带量化（Q4_K_M / Q8_0 等），`quantization` 配置项对其无效；
- **能力边界**：无 PagedAttention / Continuous Batching，吞吐低于 vLLM/SGLang，
  适合低并发或离线场景。
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from ..config import ModelConfig
from .base import EngineError, GenRequest, GenResult, InferenceEngine

logger = logging.getLogger(__name__)


class LlamaCppEngine(InferenceEngine):
    name = "llamacpp"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.llm = None

    def load(self) -> None:
        try:
            from llama_cpp import Llama  # 延迟导入
        except ImportError as exc:
            raise EngineError(
                "未安装 llama-cpp-python。请执行 pip install llama-cpp-python，"
                "或将 engine 配置为 'vllm' / 'sglang'。"
            ) from exc

        kwargs: dict = dict(
            model_path=self.cfg.model_path,  # 必须是本地 .gguf 文件
            n_ctx=self.cfg.max_model_len,
            n_gpu_layers=self.cfg.n_gpu_layers,
            verbose=False,
        )
        if self.cfg.n_threads:
            kwargs["n_threads"] = self.cfg.n_threads
        if self.cfg.chat_format:
            kwargs["chat_format"] = self.cfg.chat_format

        logger.info(
            "加载 llama.cpp GGUF: %s (n_gpu_layers=%s)", self.cfg.model_path, self.cfg.n_gpu_layers
        )
        self.llm = Llama(**kwargs)

    def _params(self, req: GenRequest) -> dict:
        params: dict = dict(
            messages=req.messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
        )
        if req.stop:
            params["stop"] = req.stop
        if req.guided_json is not None:
            # llama.cpp 支持 response_format=json_object（或 GBNF grammar）实现结构化输出
            params["response_format"] = {"type": "json_object"}
        return params

    @staticmethod
    def _to_result(resp: dict) -> GenResult:
        choice = resp["choices"][0]
        usage = resp.get("usage") or {}
        return GenResult(
            text=choice["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def generate(self, req: GenRequest) -> GenResult:
        if self.llm is None:
            raise EngineError("引擎尚未 load()")
        return self._to_result(self.llm.create_chat_completion(**self._params(req)))

    def generate_batch(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        # llama.cpp 单实例串行推理（无 Continuous Batching），逐个执行。
        return [self.generate(r) for r in reqs]

    async def stream(self, req: GenRequest) -> AsyncIterator[str]:
        if self.llm is None:
            raise EngineError("引擎尚未 load()")
        # NOTE: 迭代同步生成器；高并发建议配合 MicroBatcher / 线程池，避免阻塞事件循环。
        stream = self.llm.create_chat_completion(stream=True, **self._params(req))
        for chunk in stream:
            choices = chunk.get("choices") or [{}]
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta

    def health(self) -> dict:
        info = {
            "engine": self.name,
            "loaded": self.llm is not None,
            "model": self.cfg.model_path,
            "n_gpu_layers": self.cfg.n_gpu_layers,
        }
        if self.llm is not None:
            try:
                info["n_ctx"] = self.llm.n_ctx()
            except Exception:  # noqa: BLE001 - 属性缺失不影响健康判定
                pass
        return info
