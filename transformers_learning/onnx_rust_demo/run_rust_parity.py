#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Rust(ort) 加载**同一份** ONNX 模型并推理，与 Python 侧结果做等价性对照。

这就是大纲"动机清单"里 **跨框架 / 跨语言部署** 那条的可复现用例：
同一组 `encoder_model.onnx` / `decoder_model.onnx`，**不经过 Python 运行时**也能跑出相同译文。

流程（每个句子）：
    HF AutoTokenizer 分词 -> 写 ids.txt / mask.txt
        -> 调用 Rust 可执行文件（贪心解码）
        -> 读回 target ids -> decode
        -> 与 Python(onnxruntime) 贪心结果逐句比对

依赖：transformers + onnxruntime（用于对照）+ 已 `cargo build --release` 的本项目。

用法::

    python run_rust_parity.py
    python run_rust_parity.py --text "I love you." --text "Hello world"
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR.parent / "onnx_model"
DEFAULT_BIN = SCRIPT_DIR / "target" / "release" / "onnx_rust_demo"


def find_ort_dylib() -> str | None:
    """定位 libonnxruntime（优先 venv 里的 onnxruntime 包）。"""
    env = os.environ.get("ORT_DYLIB_PATH")
    if env and Path(env).exists():
        return env
    roots = [sys.prefix, os.path.dirname(os.path.dirname(sys.executable))]
    patterns = ["lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.dylib",
                "lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.so*"]
    for root in roots:
        for pat in patterns:
            hits = sorted(glob.glob(str(Path(root) / pat)))
            if hits:
                return hits[-1]
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="run_rust_parity.py",
                                description="Rust(ort) 加载同一 ONNX 的等价性验证")
    p.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="ONNX 模型目录")
    p.add_argument("--bin", default=str(DEFAULT_BIN), help="Rust 可执行文件路径")
    p.add_argument("--text", action="append", default=None, help="待翻译英文，可重复")
    p.add_argument("--max-length", type=int, default=60, help="贪心解码最大步数")
    p.add_argument("--no-python-ref", action="store_true",
                   help="跳过 Python(onnxruntime) 对照（只跑 Rust）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_dir = Path(args.model)
    bin_path = Path(args.bin)
    texts = args.text or ["I love you.", "The cat is sleeping on the sofa.", "Machine translation is fun."]

    if not model_dir.exists():
        print(f"找不到 ONNX 目录：{model_dir}（先运行 export_onnx_from_hf.py）")
        return 1
    if not bin_path.exists():
        print(f"找不到 Rust 可执行文件：{bin_path}\n请先：cd {SCRIPT_DIR} && cargo build --release")
        return 1

    dylib = find_ort_dylib()
    if not dylib:
        print("找不到 libonnxruntime（可设置 ORT_DYLIB_PATH 指向它）")
        return 1

    # ---- 读取生成元信息 ----
    import json

    meta = json.loads((model_dir / "onnx_meta.json").read_text(encoding="utf-8"))
    start_id, eos_id = int(meta["decoder_start_token_id"]), int(meta["eos_token_id"])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))

    py_engine = None
    if not args.no_python_ref:
        sys.path.insert(0, str(SCRIPT_DIR.parent.parent))  # 仓库根
        from transformers_learning.hf_onnx_inference import load_model

        py_engine = load_model(str(model_dir))

    print(f"[info] rust bin = {bin_path}")
    print(f"[info] ort dylib = {dylib}")
    print(f"[info] start_id={start_id} eos_id={eos_id} max_new={args.max_length}")
    print("-" * 78)

    rows, all_ok = [], True
    env = {**os.environ, "ORT_DYLIB_PATH": dylib}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for text in texts:
            enc = tok([text])
            ids = list(enc["input_ids"][0])
            mask = list(enc["attention_mask"][0])

            ids_f, mask_f, out_f = tmp / "ids.txt", tmp / "mask.txt", tmp / "out.txt"
            ids_f.write_text(" ".join(map(str, ids)))
            mask_f.write_text(" ".join(map(str, mask)))
            out_f.write_text("")

            t0 = time.time()
            proc = subprocess.run(
                [str(bin_path), str(model_dir), str(ids_f), str(mask_f), str(out_f),
                 str(start_id), str(eos_id), str(args.max_length)],
                env=env, capture_output=True, text=True,
            )
            rust_secs = time.time() - t0
            if proc.returncode != 0:
                print(f"  [rust] 失败：{proc.stderr.strip()[-500:]}")
                return 1

            rust_ids = [int(x) for x in out_f.read_text().split()]
            rust_out = tok.decode(rust_ids, skip_special_tokens=True)

            py_out, py_secs = None, None
            if py_engine is not None:
                t0 = time.time()
                py_out = py_engine.translate([text], max_length=args.max_length,
                                             num_beams=1)[0]
                py_secs = time.time() - t0

            ok = (py_out is None) or (rust_out == py_out)
            all_ok = all_ok and ok
            rows.append((text, rust_out, py_out, rust_secs, py_secs, ok))

    for text, rust_out, py_out, rs, ps, ok in rows:
        print(f"EN   : {text}")
        print(f"RUST : {rust_out}    ({rs*1000:.0f} ms, 含进程启动)")
        if py_out is not None:
            print(f"PY   : {py_out}    ({ps*1000:.0f} ms)")
        print(f"一致 : {'✅ 是' if ok else '❌ 否'}")
        print("-" * 78)

    if args.no_python_ref:
        print("（跳过了 Python 对照，仅展示 Rust 输出）")
        return 0

    same = sum(1 for r in rows if r[5])
    print(f"Rust 与 Python(onnxruntime) 贪心结果一致：{same}/{len(rows)}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
