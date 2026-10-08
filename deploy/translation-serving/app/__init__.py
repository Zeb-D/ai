"""翻译大模型私有化部署 · 应用层包。

分层（单向依赖）:
    api -> translation -> engine -> (vllm / sglang)
                    \\-> infra (cache / metrics / batching / ratelimit / logging)
"""

__version__ = "1.0.0"
