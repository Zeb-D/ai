#!/usr/bin/env python3
"""从 Hugging Face 预下载开源模型权重，供离线私有化部署。

默认下载到 Hugging Face 官方缓存目录（~/.cache/huggingface/hub），
与 huggingface_hub / transformers 的默认行为一致；如需指定目录再传 --out。
仅执行"下载"动作，不涉及任何推理代理。下载完成后即可断网运行。

用法（可直接无参数运行）：
    python scripts/download_model.py                     # 下载到 HF 缓存目录
    python scripts/download_model.py --dry-run           # 仅预览，不下载
    python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct-AWQ
    python scripts/download_model.py --out ~/models/Qwen2.5-1.5B-Instruct
    python scripts/download_model.py --include "*.safetensors" "*.json" "*.py"

自定义缓存位置（与官方一致）：
    export HF_HOME=/data/hf        # 缓存变为 /data/hf/hub
    export HF_HUB_CACHE=/data/hub  # 直接指定 hub 缓存目录
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载开源翻译模型权重")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"HF 仓库 ID（默认 {DEFAULT_MODEL}）"
    )
    parser.add_argument("--revision", default=None, help="可选：指定分支/commit")
    parser.add_argument(
        "--out",
        default=None,
        help="本地输出目录（支持 ~ 展开）；不传则使用 HF 缓存目录 ~/.cache/huggingface/hub",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="仅下载匹配文件，如 *.safetensors *.json *.py",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF Token（私有仓需要）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印解析后的参数，不执行下载")
    return parser.parse_args()


def default_cache_dir() -> Path:
    """解析 Hugging Face 默认缓存目录（与 huggingface_hub 规则一致）。"""
    if cache := os.getenv("HF_HUB_CACHE"):
        return Path(os.path.expanduser(cache))
    if hf_home := os.getenv("HF_HOME"):
        return Path(os.path.expanduser(hf_home)) / "hub"
    if xdg := os.getenv("XDG_CACHE_HOME"):
        return Path(os.path.expanduser(xdg)) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_out(args: argparse.Namespace) -> Optional[Path]:
    """返回自定义输出目录；None 表示使用 HF 默认缓存。"""
    if args.out:
        return Path(os.path.expandvars(os.path.expanduser(args.out))).resolve()
    return None


def main() -> int:
    args = parse_args()
    out_dir = resolve_out(args)
    target = out_dir if out_dir is not None else default_cache_dir()

    print(f"模型仓库 : {args.model}")
    if args.revision:
        print(f"版本     : {args.revision}")
    print(f"下载位置 : {target}" + ("" if out_dir is not None else "（HF 默认缓存）"))
    if args.include:
        print(f"包含模式 : {args.include}")

    if args.dry_run:
        print("dry-run：仅预览，未执行下载。")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少依赖：请先安装 huggingface_hub（pip install huggingface_hub）", file=sys.stderr)
        return 1

    kwargs: dict = dict(
        repo_id=args.model,
        revision=args.revision,
        allow_patterns=args.include,
        token=args.token,
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(out_dir)

    print("开始下载 ...")
    path = snapshot_download(**kwargs)
    print(f"完成，快照目录：{path}")
    print("提示：model.yaml 的 model_path 默认为仓库 ID，会自动从缓存解析，通常无需修改。")
    print(f"      如需固定为本地路径，可将 model_path 设为：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
