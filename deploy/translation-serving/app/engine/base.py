"""推理引擎统一抽象。

InferenceEngine 屏蔽 vLLM / SGLang 之间的差异，是"模型层可配置化"的关键：
同一份业务代码可在不同框架 / 不同模型间切换，而业务层无需感知具体实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Sequence

# 兜底 Chat 模板（Qwen 系列风格）。仅在 tokenizer 不可用时使用。
_QWEN_FALLBACK_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
    "{% endfor %}<|im_start|>assistant\n"
)


@dataclass
class GenRequest:
    """一次生成请求（OpenAI messages 风格）。"""

    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[list[str]] = None
    guided_json: Optional[dict] = None


@dataclass
class GenResult:
    """一次生成结果。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class EngineError(RuntimeError):
    """引擎加载失败 / 不可用 / 超时等错误的基类。"""


def render_chat(messages: list[dict], tokenizer: Any | None = None) -> str:
    """把 messages 渲染为模型输入字符串。

    优先使用 tokenizer 自带的 chat_template，保证与模型训练时格式一致；
    若 tokenizer 不可用，则退化为 Qwen 风格模板。
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        return apply(messages, tokenize=False, add_generation_prompt=True)

    normalized = [
        {"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages
    ]
    try:  # 用 jinja2 渲染兜底模板（运行环境已依赖 jinja2）
        from jinja2 import Template

        return Template(_QWEN_FALLBACK_TEMPLATE).render(messages=normalized)
    except Exception:  # noqa: BLE001 - 最终兜底，绝不再抛
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in normalized]
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)


class InferenceEngine(ABC):
    """推理引擎抽象基类。"""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """加载模型权重到 GPU（进程启动时调用一次）。"""

    @abstractmethod
    def generate(self, req: GenRequest) -> GenResult:
        """单请求生成（同步）。"""

    def generate_batch(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        """批量生成。默认逐个实现，子类应覆盖为真正的批处理。"""
        return [self.generate(r) for r in reqs]

    @abstractmethod
    def stream(self, req: GenRequest) -> AsyncIterator[str]:
        """流式生成，返回增量文本片段的异步迭代器。"""

    @abstractmethod
    def health(self) -> dict:
        """健康检查，返回状态字典（含队列深度、显存等）。"""

    def close(self) -> None:
        """释放资源（可选）。"""
        return None
