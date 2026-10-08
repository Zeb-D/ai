"""SGLang 进程内推理引擎（Pattern A 可选引擎）。

相对 vLLM 的优势：RadixAttention 对"固定 System Prompt + 多变 User 输入"命中率更高，
正好匹配翻译流量形态；并支持真正的流式生成。
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, Sequence

from ..config import ModelConfig
from .base import EngineError, GenRequest, GenResult, InferenceEngine, render_chat

logger = logging.getLogger(__name__)


class SGLangEngine(InferenceEngine):
    name = "sglang"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.engine = None
        self._tokenizer = None

    def load(self) -> None:
        try:
            import sglang as sgl  # 延迟导入
        except ImportError as exc:  # pragma: no cover - 依赖 GPU 环境
            raise EngineError(
                "未安装 sglang。请安装 sglang 推理引擎，或将 engine 配置为 'vllm'。"
            ) from exc

        kwargs: dict = dict(
            model_path=self.cfg.model_path,
            tp_size=self.cfg.tp_size,
            mem_fraction_static=self.cfg.gpu_mem_util,
            context_length=self.cfg.max_model_len,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if self.cfg.quantization:
            kwargs["quantization"] = self.cfg.quantization

        logger.info("加载 SGLang 模型: %s (quant=%s, tp=%d)",
                    self.cfg.model_path, self.cfg.quantization, self.cfg.tp_size)
        self.engine = sgl.Engine(**kwargs)
        self._tokenizer = self._try_load_tokenizer()

    def _try_load_tokenizer(self):
        """优先用 transformers 加载 tokenizer 以渲染 chat 模板。"""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                self.cfg.model_path, trust_remote_code=self.cfg.trust_remote_code
            )
        except Exception:  # noqa: BLE001
            logger.warning("SGLang 引擎未能加载 tokenizer，将使用兜底模板渲染。")
            return None

    def _sampling_params(self, req: GenRequest) -> dict:
        params: dict = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_new_tokens": req.max_tokens,
        }
        if req.stop:
            params["stop"] = req.stop
        if req.guided_json is not None:
            params["json_schema"] = req.guided_json
        return params

    def _render(self, req: GenRequest) -> str:
        # SGLang 接受 messages 列表亦可，这里统一渲染为字符串以复用兜底模板逻辑。
        return render_chat(req.messages, self._tokenizer)

    def _to_result(self, out: dict) -> GenResult:
        meta = out.get("meta_info", {}) or {}
        return GenResult(
            text=out.get("text", ""),
            prompt_tokens=int(meta.get("prompt_tokens", 0)),
            completion_tokens=int(meta.get("completion_tokens", 0)),
        )

    def generate(self, req: GenRequest) -> GenResult:
        if self.engine is None:
            raise EngineError("引擎尚未 load()")
        out = self.engine.generate(prompt=self._render(req),
                                   sampling_params=self._sampling_params(req))
        return self._to_result(out)

    def generate_batch(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        if self.engine is None:
            raise EngineError("引擎尚未 load()")
        return [self.generate(r) for r in reqs]

    async def stream(self, req: GenRequest) -> AsyncIterator[str]:
        if self.engine is None:
            raise EngineError("引擎尚未 load()")
        # SGLang 原生支持 stream=True，返回增量 dict 的生成器。
        # NOTE: 此处直接迭代同步生成器；高并发场景建议通过线程/队列桥接以彻底避免阻塞事件循环。
        generator = self.engine.generate(
            prompt=self._render(req),
            sampling_params=self._sampling_params(req),
            stream=True,
        )
        for chunk in generator:
            delta = chunk.get("text", "")
            if delta:
                yield delta

    def health(self) -> dict:
        return {"engine": self.name, "loaded": self.engine is not None,
                "model": self.cfg.model_path}
