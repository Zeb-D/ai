#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 onnxruntime 加载 export_onnx.py 导出的 ONNX 模型，做真实的英译中推理，
并与 PyTorch 原模型逐 token 对比，确保 ONNX 结果没问题。

依赖 export_onnx.py 导出的两个拆分模型（自回归解码用它们，编码器只跑一次）：
    transformer_encoder.onnx   src(int64, B,S) -> memory(B,S,D)
    transformer_decoder.onnx   tgt(int64, B,T), memory(B,S,D), src_mask(bool,B,1,S) -> log_probs(B,T,V)

用法（在 transformers_learning 目录下执行）：
    python translate_onnx.py                                  # 取 dev.json 前 3 句做校验+翻译
    python translate_onnx.py --text "The cat is sleeping on the sofa."
    python translate_onnx.py --from-dev 10 --decode beam --no-compare
    python translate_onnx.py --onnx-dir data/train/exp/weights/onnx --show-steps
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# tools/tokenizer_utils.py 里写死了 ./tokenizer/xxx.model，先把工作目录切到脚本目录
os.chdir(SCRIPT_DIR)

import config  # noqa: E402
from tools.tokenizer_utils import chinese_tokenizer_load, english_tokenizer_load  # noqa: E402

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("translate_onnx")


# ---------------------------------------------------------------------------
# ONNX 侧：加载模型 + 解码
# ---------------------------------------------------------------------------
def load_sessions(onnx_dir: str, use_coreml: bool = False):
    import onnxruntime as ort

    enc_path = os.path.join(onnx_dir, "transformer_encoder.onnx")
    dec_path = os.path.join(onnx_dir, "transformer_decoder.onnx")
    for path in (enc_path, dec_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到 {path}，请先执行: python export_onnx.py --ckpt <权重路径>"
            )
    # 默认只用 CPU：CoreML/GPU 可能把算子降精度，导致与 PyTorch 的逐 token 一致性被破坏
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if use_coreml and "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    enc = ort.InferenceSession(enc_path, providers=providers)
    dec = ort.InferenceSession(dec_path, providers=providers)
    LOGGER.info("onnxruntime %s, providers=%s", ort.__version__, [p for p in providers])
    return enc, dec


def _encode_source(enc, src_ids, padding_idx: int):
    """编码一次，返回 (memory, src_mask)，自回归解码时复用。"""
    src = np.asarray([src_ids], dtype=np.int64)          # (1, S)
    src_mask = (src != padding_idx)[:, None, :]          # (1, 1, S)
    memory = enc.run(None, {"src": src})[0]              # (1, S, D)
    return memory, src_mask


def _decode_step(dec, tgt_batch: np.ndarray, memory: np.ndarray, src_mask: np.ndarray) -> np.ndarray:
    """一次前向，返回每条序列最后一个位置的 log_probs，shape (batch, vocab)。"""
    return dec.run(None, {"tgt": tgt_batch, "memory": memory, "src_mask": src_mask})[0][:, -1, :]


def onnx_greedy_decode(enc, dec, src_ids, max_len, padding_idx, bos_idx, eos_idx, show_steps=False):
    memory, src_mask = _encode_source(enc, src_ids, padding_idx)
    tgt = np.array([[bos_idx]], dtype=np.int64)
    for step in range(max_len - 1):
        log_probs = _decode_step(dec, tgt, memory, src_mask)
        nxt = int(log_probs[0].argmax())
        if show_steps:
            LOGGER.info("  step %2d: token=%d, log_prob=%.4f", step + 1, nxt, float(log_probs[0][nxt]))
        tgt = np.concatenate([tgt, np.array([[nxt]], dtype=np.int64)], axis=1)
        if nxt == eos_idx:
            break
    return tgt[0].tolist()


def onnx_beam_search(enc, dec, src_ids, max_len, beam_size, padding_idx, bos_idx, eos_idx):
    """束搜索：每步把所有 beam 作为一个 batch 一起喂给 decoder（顺便验证动态 batch）。"""
    memory, src_mask = _encode_source(enc, src_ids, padding_idx)
    beams = [([bos_idx], 0.0)]
    finished = []
    for _ in range(max_len - 1):
        if not beams:
            break
        tgt_batch = np.array([ids for ids, _ in beams], dtype=np.int64)
        mem_batch = np.repeat(memory, len(beams), axis=0)
        mask_batch = np.repeat(src_mask, len(beams), axis=0)
        last_log_probs = _decode_step(dec, tgt_batch, mem_batch, mask_batch)

        candidates = []
        for i, (ids, score) in enumerate(beams):
            top_ids = np.argpartition(-last_log_probs[i], beam_size)[:beam_size]
            for tid in top_ids:
                candidates.append((ids + [int(tid)], score + float(last_log_probs[i][tid])))
        candidates.sort(key=lambda item: item[1], reverse=True)

        beams = []
        for ids, score in candidates:
            if ids[-1] == eos_idx:
                finished.append((ids, score))  # 这里存累计 log 概率，最后统一做长度归一化
            else:
                beams.append((ids, score))
            if len(beams) >= beam_size:
                break

    # 按平均 log 概率（长度归一化）挑最优：避免打分偏好短句或长句
    pool = finished or beams
    pool.sort(key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)
    return pool[0][0]


# ---------------------------------------------------------------------------
# PyTorch 侧：同一个句子的贪心解码（作为对照基准）
# ---------------------------------------------------------------------------
def torch_reference_decode(src_ids, ckpt_path, max_len, padding_idx, bos_idx, eos_idx):
    """加载 .pth，用原模型（未打补丁）做贪心解码，返回 token id 列表。"""
    import torch

    from model.tf_model import make_model

    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.to(config.device)
    state_dict = torch.load(ckpt_path, map_location=config.device, weights_only=True)
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()})
    model.eval()

    with torch.no_grad():
        src = torch.LongTensor([src_ids])
        src_mask = (src != padding_idx).unsqueeze(-2)
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), bos_idx, dtype=torch.long)
        for _ in range(max_len - 1):
            size = ys.size(1)
            tgt_mask = (ys != padding_idx).unsqueeze(-2) & torch.tril(torch.ones(size, size, dtype=torch.bool))
            out = model.decode(memory, src_mask, ys, tgt_mask)
            nxt = int(model.generator(out[:, -1]).argmax(dim=-1).item())
            ys = torch.cat([ys, torch.full((1, 1), nxt, dtype=torch.long)], dim=1)
            if nxt == eos_idx:
                break
    return ys[0].tolist()


# ---------------------------------------------------------------------------
# 文本 <-> id
# ---------------------------------------------------------------------------
def text_to_ids(sent: str, en_tok, bos_idx: int, eos_idx: int):
    return [bos_idx] + en_tok.EncodeAsIds(sent) + [eos_idx]


def ids_to_text(ids, chn_tok, padding_idx: int, bos_idx: int, eos_idx: int):
    cleaned = []
    for tid in ids:
        if tid == eos_idx:
            break
        if tid in (padding_idx, bos_idx):
            continue
        cleaned.append(tid)
    return chn_tok.decode_ids(cleaned)


def load_dev_sentences(count: int):
    path = os.path.join(SCRIPT_DIR, config.dev_data_path.lstrip("./"))
    if not os.path.isfile(path):
        LOGGER.warning("找不到 %s，改用内置示例句", path)
        return [["I love you.", "我爱你。"]][:count]
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data[:count]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="translate_onnx.py",
        description="用 onnxruntime 跑 ONNX 翻译模型，并与 PyTorch 原模型对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-dir", default=None, help="ONNX 目录，默认 <config.translate_model_path 所在目录>/onnx")
    parser.add_argument("--ckpt", default=config.translate_model_path, help="用于对比的 PyTorch 权重路径")
    parser.add_argument("--text", action="append", default=None, help="要翻译的英文句子，可重复指定")
    parser.add_argument("--from-dev", type=int, default=3, help="未指定 --text 时，从 dev.json 取前 N 句")
    parser.add_argument("--max-len", type=int, default=config.max_len, help="解码最大长度")
    parser.add_argument("--beam-size", type=int, default=config.beam_size, help="beam search 的 beam 大小")
    parser.add_argument("--decode", choices=["greedy", "beam", "both"], default="both", help="解码方式")
    parser.add_argument("--no-compare", dest="compare", action="store_false", help="跳过与 PyTorch 的逐 token 对比")
    parser.add_argument("--show-steps", action="store_true", help="打印贪心解码每一步的 log_prob")
    parser.add_argument("--coreml", action="store_true", help="使用 CoreML EP（更快，但可能与 PyTorch 有数值差异）")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    onnx_dir = args.onnx_dir or os.path.join(
        os.path.dirname(os.path.abspath(os.path.join(SCRIPT_DIR, args.ckpt))), "onnx")
    if not os.path.isdir(onnx_dir):
        raise FileNotFoundError(f"ONNX 目录不存在: {onnx_dir}，请先执行 python export_onnx.py")

    # 分词器（sentencepiece 模型里的 bos/eos 与 config 保持一致）
    en_tok = english_tokenizer_load()
    chn_tok = chinese_tokenizer_load()
    padding_idx, bos_idx, eos_idx = int(config.padding_idx), int(config.bos_idx), int(config.eos_idx)
    LOGGER.info("分词器加载完成: eng vocab=%d, chn vocab=%d", en_tok.GetPieceSize(), chn_tok.GetPieceSize())

    if args.text:
        pairs = [[t, ""] for t in args.text]
    else:
        pairs = load_dev_sentences(args.from_dev)

    enc, dec = load_sessions(onnx_dir, use_coreml=args.coreml)

    if args.compare:
        import torch  # noqa: F401  (仅确认 torch 可用)

    all_matched = True
    total_onnx_time = 0.0
    for idx, (en_sent, zh_ref) in enumerate(pairs, 1):
        src_ids = text_to_ids(en_sent, en_tok, bos_idx, eos_idx)

        # ---------- ONNX 推理 ----------
        t0 = time.time()
        greedy_ids = onnx_greedy_decode(enc, dec, src_ids, args.max_len, padding_idx,
                                        bos_idx, eos_idx, show_steps=args.show_steps)
        greedy_text = ids_to_text(greedy_ids, chn_tok, padding_idx, bos_idx, eos_idx)
        beam_text = None
        if args.decode in ("beam", "both"):
            beam_ids = onnx_beam_search(enc, dec, src_ids, args.max_len, args.beam_size,
                                        padding_idx, bos_idx, eos_idx)
            beam_text = ids_to_text(beam_ids, chn_tok, padding_idx, bos_idx, eos_idx)
        onnx_cost = time.time() - t0
        total_onnx_time += onnx_cost

        # ---------- PyTorch 对照 ----------
        matched = None
        torch_text = None
        if args.compare:
            torch_ids = torch_reference_decode(src_ids, args.ckpt, args.max_len,
                                               padding_idx, bos_idx, eos_idx)
            torch_text = ids_to_text(torch_ids, chn_tok, padding_idx, bos_idx, eos_idx)
            # 逐 token 对比（忽略末尾 EOS 之后的部分）
            cut = lambda seq: seq[: seq.index(eos_idx) + 1] if eos_idx in seq else seq  # noqa: E731
            matched = cut(greedy_ids) == cut(torch_ids)
            all_matched = all_matched and matched

        print("\n" + "=" * 100)
        print(f"[{idx}] EN        : {en_sent}")
        if zh_ref:
            print(f"    参考译文  : {zh_ref}")
        print(f"    ONNX 贪心 : {greedy_text}")
        if beam_text is not None:
            print(f"    ONNX 束搜索(--beam-size {args.beam_size}): {beam_text}")
        if args.compare:
            print(f"    PyTorch   : {torch_text}")
            print(f"    token 级一致: {matched}   (ONNX {len(greedy_ids)} tokens, "
                  f"PyTorch {len(torch_ids)} tokens)")
        print(f"    ONNX 解码耗时: {onnx_cost:.2f}s")

    print("\n" + "=" * 100)
    if args.compare:
        print(f"对比结论: {'全部句子 ONNX 与 PyTorch 逐 token 完全一致 ✅' if all_matched else '存在不一致 ❌'}")
    print(f"共 {len(pairs)} 句，ONNX 总耗时 {total_onnx_time:.2f}s")
    import onnxruntime as ort
    print(f"运行环境: onnxruntime {ort.__version__}, torch {__import__('torch').__version__}")
    return 0 if all_matched else 1


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
