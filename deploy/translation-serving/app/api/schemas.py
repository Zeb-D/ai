"""API 请求 / 响应数据模型（Pydantic v2）。"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 业务接口
class TranslateRequest(BaseModel):
    text: Union[str, list[str]] = Field(..., description="待翻译文本，支持单条或多条")
    source_lang: str = Field("en", description="源语言代码，如 en / zh")
    target_lang: str = Field("zh", description="目标语言代码")
    domain: str = Field("general", description="领域，用于术语过滤与 Prompt 提示")
    use_glossary: bool = Field(True, description="是否启用术语约束")


class GlossaryHit(BaseModel):
    src: str
    tgt: str


class TranslationItem(BaseModel):
    translation: str
    cached: bool = False
    glossary_hits: list[GlossaryHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TranslationMeta(BaseModel):
    model: str
    engine: str
    request_id: str
    duration_ms: int


class TranslateResponse(BaseModel):
    translations: list[str]
    items: list[TranslationItem]
    meta: TranslationMeta


# ---------------------------------------------------------------- OpenAI 兼容
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    response_format: Optional[dict] = None


class ErrorResponse(BaseModel):
    detail: str
