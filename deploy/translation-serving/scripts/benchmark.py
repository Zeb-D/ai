#!/usr/bin/env python3
"""简单压测脚本：并发请求 /v1/translate，统计吞吐与延迟分位数。

用法：
    python scripts/benchmark.py --base-url http://127.0.0.1:8080 \\
        --requests 200 --concurrency 16 --text "Hello world."
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round((pct / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="翻译服务压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--text", default="Hello world.")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    url = args.base_url.rstrip("/") + "/v1/translate"
    payload = {
        "text": args.text,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=args.timeout) as client:

        async def worker() -> None:
            nonlocal errors
            async with semaphore:
                start = time.perf_counter()
                try:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"请求失败: {exc}")
                    return
                latencies.append(time.perf_counter() - start)

        started = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(args.requests)])
        elapsed = time.perf_counter() - started

    latencies.sort()
    ok = len(latencies)
    print("=" * 48)
    print(f"总请求: {args.requests}  成功: {ok}  失败: {errors}")
    print(f"总耗时: {elapsed:.2f}s  吞吐: {ok / elapsed:.2f} req/s")
    if latencies:
        print(f"平均延迟: {statistics.mean(latencies) * 1000:.1f} ms")
        print(f"P50: {percentile(latencies, 50) * 1000:.1f} ms")
        print(f"P95: {percentile(latencies, 95) * 1000:.1f} ms")
        print(f"P99: {percentile(latencies, 99) * 1000:.1f} ms")
    print("=" * 48)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
