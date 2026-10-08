"""vLLM 进程内推理引擎（Pattern A 默认引擎）。

要点：
- 权重从本地磁盘加载（model_path），不访问任何外部服务；
- 采用离线 LLM 接口做 generate / generate_batch，吞吐最优；
- 开启 prefix caching，翻译固定 System Prompt 收益大。
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from ..config import ModelConfig
from .base import EngineError, GenRequest, GenResult, InferenceEngine, render_chat

logger = logging.getLogger(__name__)


class VLLMEngine(InferenceEngine):
    name = "vllm"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.llm = None
        self._tokenizer = None

    def load(self) -> None:
        try:
            from vllm import LLM  # 延迟导入：未安装 vllm 时仍可导入本模块
        except ImportError as exc:  # pragma: no cover - 依赖 GPU 环境
            raise EngineError(
                "未安装 vllm。请安装 vllm 推理引擎，或将 engine 配置为 'sglang'。"
            ) from exc

        kwargs: dict = dict(
            model=self.cfg.model_path,
            tensor_parallel_size=self.cfg.tp_size,
            gpu_memory_utilization=self.cfg.gpu_mem_util,
            max_model_len=self.cfg.max_model_len,
            trust_remote_code=self.cfg.trust_remote_code,
            enable_prefix_caching=self.cfg.prefix_caching,
        )
        if self.cfg.quantization:
            kwargs["quantization"] = self.cfg.quantization

        logger.info("加载 vLLM 模型: %s (quant=%s, tp=%d)",
                    self.cfg.model_path, self.cfg.quantization, self.cfg.tp_size)
        self.llm = LLM(**kwargs)
        self._tokenizer = self.llm.get_tokenizer()

    def _sampling_params(self, req: GenRequest):
        from vllm import SamplingParams

        kwargs: dict = dict(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
        )
        if req.stop:
            kwargs["stop"] = req.stop
        if req.guided_json is not None:
            kwargs["guided_json"] = req.guided_json
        return SamplingParams(**kwargs)

    def _render(self, req: GenRequest) -> str:
        return render_chat(req.messages, self._tokenizer)

    def generate(self, req: GenRequest) -> GenResult:
        if self.llm is None:
            raise EngineError("引擎尚未 load()")
        prompt = self._render(req)
        out = self.llm.generate([prompt], self._sampling_params(req))[0]
        return GenResult(
            text=out.outputs[0].text,
            prompt_tokens=len(out.prompt_token_ids),
            completion_tokens=len(out.outputs[0].token_ids),
        )

    def generate_batch(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        if self.llm is None:
            raise EngineError("引擎尚未 load()")
        if not reqs:
            return []
        prompts = [self._render(r) for r in reqs]
        params = [self._sampling_params(r) for r in reqs]
        outs = self.llm.generate(prompts, params)
        return [
            GenResult(
                text=o.outputs[0].text,
                prompt_tokens=len(o.prompt_token_ids),
                completion_tokens=len(o.outputs[0].token_ids),
            )
            for o in outs
        ]

    async def stream(self, req: GenRequest) -> AsyncIterator[str]:
        """流式输出。

        NOTE: vLLM 的离线 LLM 接口不支持 token 级流式；真正的逐 token 流式需使用
        AsyncLLMEngine。此处以"整段生成后分块下发"的方式提供兼容接口，保证协议一致。
        """
        result = self.generate(req)
        text = result.text
        # 按较小粒度切分，改善首字节观感；非严格 token 级。
        step = 16
        for i in range(0, len(text), step):
            yield text[i : i + step]

    def health(self) -> dict:
        status = {"engine": self.name, "loaded": self.llm is not None,
                  "model": self.cfg.model_path}
        try:  # 尽力采集显存信息，失败不影响健康判定
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                status["gpu_mem_used_ratio"] = round(1 - free / total, 4)
        except Exception:  # noqa: BLE001
            pass
        return status
