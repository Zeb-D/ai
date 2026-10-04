#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ONNX 可复现实验集合（对应大纲"动机清单"里可验证的几条）。

子命令：

    python onnx_experiments.py opt      # 图优化级别 ORT_DISABLE_ALL/BASIC/EXTENDED/ALL 对比
    python onnx_experiments.py quant    # int8 动态量化：体积 / 延迟 / 与 fp32 的输出一致性

设计原则：
- 所有实验都在**同一台机器、同一份权重、同一批句子**上跑，结果可直接复现；
- 量化实验会**诚实地报告译文是否发生变化**（不假设一定无损）。
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "onnx_model"
INT8_DIR = SCRIPT_DIR / "onnx_model_int8"
SENTENCES = [
    "I love you.",
    "The cat is sleeping on the sofa.",
    "Machine translation is fun.",
]


# ---------------------------------------------------------------- 公共工具
def _engine(model_dir: Path, level=None, providers=None):
    """构造 OnnxSeq2Seq，并按需覆盖 session（优化级别 / provider）。"""
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from hf_onnx_inference import OnnxSeq2Seq

    import onnxruntime as ort

    eng = OnnxSeq2Seq(str(model_dir))
    if level is None and providers is None:
        return eng

    so = ort.SessionOptions()
    if level is not None:
        so.graph_optimization_level = level
    prov = providers or [eng.provider]
    t0 = time.time()
    eng.enc = ort.InferenceSession(str(model_dir / "encoder_model.onnx"), so, providers=prov)
    eng.dec = ort.InferenceSession(str(model_dir / "decoder_model.onnx"), so, providers=prov)
    return eng, time.time() - t0


def _bench(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        s = time.time()
        fn()
        ts.append(time.time() - s)
    return min(ts), statistics.mean(ts)


# ---------------------------------------------------------------- 实验 1：图优化级别
def exp_opt(model_dir: Path) -> int:
    import onnxruntime as ort

    eng = _engine(model_dir)
    print(f"模型：{model_dir}")
    print(f"provider：{eng.provider}    句子数：{len(SENTENCES)}    num_beams=1")
    print()
    print("| 图优化级别 | session 构建 | 首次推理 | min | avg |")
    print("| --- | --- | --- | --- | --- |")

    levels = [
        ("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("ORT_ENABLE_BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("ORT_ENABLE_EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("ORT_ENABLE_ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]
    results = {}
    for name, lvl in levels:
        e, build_s = _engine(model_dir, level=lvl)
        t0 = time.time()
        out0 = e.translate(SENTENCES, num_beams=1)
        first = time.time() - t0
        mn, avg = _bench(lambda: e.translate(SENTENCES, num_beams=1))
        results[name] = out0
        print(f"| {name} | {build_s:.2f}s | {first:.3f}s | {mn:.3f}s | {avg:.3f}s |")

    ref = results["ORT_DISABLE_ALL"]
    print()
    for name, out in results.items():
        same = sum(1 for a, b in zip(out, ref) if a == b)
        print(f"  {name}: 与 DISABLE_ALL 译文一致 {same}/{len(ref)}")

    # 保留基本优化即可，避免默认 ALL 带来的 build 成本
    return 0


# ---------------------------------------------------------------- 实验 2：int8 动态量化
def exp_quant(model_dir: Path, out_dir: Path) -> int:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    # 复制除 .onnx 外的所有文件（tokenizer / config / onnx_meta.json）
    for f in model_dir.iterdir():
        if f.is_file() and f.suffix != ".onnx":
            shutil.copy2(f, out_dir / f.name)

    print(f"源目录：{model_dir}  ->  量化目录：{out_dir}")
    print()
    for name in ("encoder_model.onnx", "decoder_model.onnx"):
        src, dst = model_dir / name, out_dir / name
        t0 = time.time()
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        print(f"  quantize_dynamic({name}): {time.time() - t0:.1f}s  "
              f"{src.stat().st_size/1e6:.1f}MB -> {dst.stat().st_size/1e6:.1f}MB")

    fp32 = _engine(model_dir)
    int8 = _engine(out_dir)

    def total(d: Path) -> float:
        return sum(f.stat().st_size for f in d.glob("*.onnx")) / 1e6

    print()
    print(f"仅 .onnx 体积：fp32 = {total(model_dir):.1f}MB   int8 = {total(out_dir):.1f}MB "
          f"（压缩 {total(out_dir)/total(model_dir)*100:.0f}%）")

    t0 = time.time()
    ref = fp32.translate(SENTENCES, num_beams=1)
    fp32_time = time.time() - t0
    t0 = time.time()
    got = int8.translate(SENTENCES, num_beams=1)
    int8_time = time.time() - t0

    print()
    print("| 句子 | fp32 | int8 | 一致 |")
    print("| --- | --- | --- | --- |")
    same = 0
    for s, a, b in zip(SENTENCES, ref, got):
        ok = a == b
        same += ok
        print(f"| {s} | {a} | {b} | {'✅' if ok else '❌'} |")

    print()
    print(f"输出一致率：{same}/{len(SENTENCES)}")
    print(f"3 句总耗时：fp32 = {fp32_time:.3f}s   int8 = {int8_time:.3f}s")
    if same < len(SENTENCES):
        print("⚠️  int8 动态量化**改变了输出**——说明该模型对量化敏感，需按业务评估是否可接受。")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="onnx_experiments.py")
    p.add_argument("cmd", choices=["opt", "quant"])
    p.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--out", default=str(INT8_DIR), help="quant 的输出目录")
    args = p.parse_args(argv)

    if args.cmd == "opt":
        return exp_opt(Path(args.model))
    return exp_quant(Path(args.model), Path(args.out))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
