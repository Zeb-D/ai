#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ONNXRuntime 推理 Demo：英译中（加载 [`export_onnx_from_hf.py`](export_onnx_from_hf.py:1)
导出的 ONNX 模型），CLI 与 [`hf_inference.py`](hf_inference.py:1) 保持一致。

与 PyTorch 版不同点：**不再需要 transformers 的 model 前向**，只用 tokenizer +
onnxruntime，beam search 在 numpy 上实现（无 past_key_values，短句够快）。

用法::

    # 1) 直接翻译若干句
    python hf_onnx_inference.py --text "The cat is sleeping on the sofa." --text "I love you."

    # 2) 翻译整个文件（每行一句）
    python hf_onnx_inference.py --file input.txt --out output.txt

    # 3) 交互式（不给 --text/--file 时默认进入）
    python hf_onnx_inference.py

    # 4) 指定 ONNX 目录
    python hf_onnx_inference.py --model transformers_learning/onnx_model --text "I love you."

只做推理（作为模块 import，不需要命令行）::

    from transformers_learning.demo_onnx_inference import translate, load_model

    translate("I love you.")                    # 单句 -> str
    translate(["I love you.", "Hello!"])        # 多句 -> list[str]

    engine = load_model("transformers_learning/onnx_model")   # 复用时只加载一次
    engine.translate(["I love you."], num_beams=5)

依赖：``transformers>=5 onnxruntime sentencepiece sacremoses``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 脚本所在目录：默认模型路径始终相对它解析，避免受运行时 CWD 影响
# （例如在 PyCharm 里直接 Run 时，CWD 会是 transformers_learning/）。
SCRIPT_DIR = Path(__file__).resolve().parent


def default_model_dir() -> str:
    """ONNX 模型默认目录（脚本同级的 onnx_model/）。"""
    return str(SCRIPT_DIR / "onnx_model")


def _resolve_model_dir(model_dir) -> Path:
    """把目录解析成真实存在的路径（兼容相对/绝对、不同 CWD）。"""
    raw = Path(model_dir).expanduser()
    if raw.is_absolute():
        return raw
    for cand in (raw, SCRIPT_DIR / raw, SCRIPT_DIR.parent / raw):
        if cand.exists():
            return cand
    return raw


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="hf_onnx_inference.py",
        description="ONNXRuntime 英译中 Demo（加载 export_onnx_from_hf.py 的产物）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=default_model_dir(),
                        help="ONNX 模型目录（export_onnx_from_hf.py --out 的产物）")
    parser.add_argument("--text", action="append", default=None,
                        help="要翻译的英文句子，可重复指定")
    parser.add_argument("--file", default=None, help="输入文件（每行一句英文）")
    parser.add_argument("--out", default=None, help="把结果写到该文件（默认打印到终端）")
    parser.add_argument("--max-length", type=int, default=60, help="生成长度上限")
    parser.add_argument("--num-beams", type=int, default=3, help="beam search 宽度；1 表示贪心")
    parser.add_argument("--batch-size", type=int, default=8, help="批量推理的 batch 大小")
    parser.add_argument("--length-penalty", type=float, default=1.0,
                        help="beam search 长度惩罚指数（1.0 为不惩罚）")
    parser.add_argument("--show-ids", action="store_true", help="额外打印源/目标 token id")
    parser.add_argument("--compare-hf", action="store_true",
                        help="同时用 transformers PyTorch 推理做对照（需 torch）")
    return parser.parse_args(argv)


class OnnxSeq2Seq:
    """封装 encoder/decoder 两个 onnxruntime session + tokenizer。"""

    def __init__(self, model_dir: str, num_threads: int = 0):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        d = _resolve_model_dir(model_dir)
        if not d.exists():
            raise FileNotFoundError(f"ONNX 目录不存在：{d}（先运行 export_onnx_from_hf.py）")

        meta_path = d / "onnx_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"缺少 {meta_path}，请先运行 export_onnx_from_hf.py")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            so.intra_op_num_threads = num_threads

        providers = ort.get_available_providers()
        # 优先加速后端，退化到 CPU
        for pref in ("CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
            if pref in providers:
                self.provider = pref
                break
        else:
            self.provider = providers[0]

        self.enc = ort.InferenceSession(str(d / "encoder_model.onnx"), so, providers=[self.provider])
        self.dec = ort.InferenceSession(str(d / "decoder_model.onnx"), so, providers=[self.provider])
        self.tokenizer = AutoTokenizer.from_pretrained(str(d))

        self.start_id = int(self.meta["decoder_start_token_id"])
        self.eos_id = int(self.meta["eos_token_id"])
        self.pad_id = int(self.meta["pad_token_id"])
        self.vocab_size = int(self.meta["vocab_size"])

    # ---------- 编码 ----------
    def encode(self, texts):
        enc = self.tokenizer(list(texts), padding=True, truncation=True)
        input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(enc["attention_mask"], dtype=np.int64)
        hidden = self.enc.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })[0]
        return input_ids, attention_mask, hidden

    # ---------- 解码一步 ----------
    def _decoder_logits(self, encoder_hidden, encoder_mask, decoder_input_ids):
        return self.dec.run(None, {
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": encoder_mask,
            "decoder_input_ids": decoder_input_ids,
        })[0]

    # ---------- 贪心 ----------
    def _greedy(self, hidden, mask, max_new_tokens):
        seq = np.full((1, 1), self.start_id, dtype=np.int64)
        for _ in range(max_new_tokens):
            logits = self._decoder_logits(hidden, mask, seq)[0, -1, :]
            nxt = int(np.argmax(logits))
            seq = np.concatenate([seq, np.array([[nxt]], dtype=np.int64)], axis=1)
            if nxt == self.eos_id:
                break
        return seq[0].tolist()

    # ---------- beam search ----------
    def _beam_search(self, hidden, mask, max_new_tokens, num_beams, length_penalty):
        K = max(1, int(num_beams))
        if K == 1:
            return self._greedy(hidden, mask, max_new_tokens)

        def log_softmax(x):
            m = np.max(x, axis=-1, keepdims=True)
            return x - m - np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))

        beams = [[self.start_id] for _ in range(K)]
        beam_scores = np.array([0.0] + [-1e9] * (K - 1), dtype=np.float32)
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_new_tokens):
            seq = np.asarray(beams, dtype=np.int64)                     # (K, T)
            exp_hidden = np.repeat(hidden, K, axis=0)
            exp_mask = np.repeat(mask, K, axis=0)
            logits = self._decoder_logits(exp_hidden, exp_mask, seq)[:, -1, :]  # (K, V)
            logprobs = log_softmax(logits) + beam_scores[:, None]        # (K, V)

            cand: list[tuple[float, list[int]]] = []
            for k in range(K):
                top = np.argsort(-logprobs[k])[:K]
                for tok in top:
                    cand.append((float(logprobs[k, tok]), beams[k] + [int(tok)]))
            cand.sort(key=lambda x: x[0], reverse=True)

            new_beams, new_scores = [], []
            for score, s in cand:
                if s[-1] == self.eos_id:
                    completed.append((score, s))
                else:
                    new_beams.append(s)
                    new_scores.append(score)
                if len(new_beams) >= K:
                    break
            if not new_beams:
                break
            beams, beam_scores = new_beams, np.asarray(new_scores, dtype=np.float32)

            # 早停：已完成的候选中最好的都不比所有在跑的 beam 差太多时停止
            if completed:
                best_done = max(c for c, _ in completed) / (max_new_tokens ** length_penalty)
                worst_alive = min(beam_scores) / (1.0)
                if best_done > worst_alive and len(completed) >= K:
                    break

        if completed:
            def norm(item):
                score, s = item
                return score / (max(1, len(s)) ** length_penalty)

            completed.sort(key=norm, reverse=True)
            return completed[0][1]

        return beams[int(np.argmax(beam_scores))]

    # ---------- 对外：批量翻译 ----------
    def translate(self, texts, max_length=60, num_beams=3, batch_size=8,
                  length_penalty=1.0, show_ids=False):
        results = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = list(texts[start:start + max(1, batch_size)])
            input_ids, attention_mask, hidden = self.encode(batch)
            for i, text in enumerate(batch):
                h = hidden[i:i + 1]
                m = attention_mask[i:i + 1]
                seq = self._beam_search(h, m, int(max_length), num_beams, length_penalty)
                results.append(self.tokenizer.decode(seq, skip_special_tokens=True))
                if show_ids:
                    src = input_ids[i][attention_mask[i].astype(bool)].tolist()
                    print(f"    src ids: {src}")
                    print(f"    tgt ids: {seq}")
                    _ = text
        return results


# ======================================================================
# 只做推理的极简 API（import 即用，不走命令行、不做 compare-hf）
# ----------------------------------------------------------------------
#   from transformers_learning.demo_onnx_inference import translate, load_model
#   translate("I love you.")                 # str  -> str
#   translate(["I love you.", "Hello!"])     # list -> list[str]
#   engine = load_model()                    # 需要复用时（按目录缓存）
#   engine.translate(["..."], num_beams=5)
# ======================================================================

DEFAULT_MODEL_DIR = default_model_dir()
# 兼容简写：也可以引用 demo_onnx_inference.MODEL_DIR
MODEL_DIR = DEFAULT_MODEL_DIR
_ENGINE_CACHE: dict[str, "OnnxSeq2Seq"] = {}


def load_model(model_dir: str = DEFAULT_MODEL_DIR) -> "OnnxSeq2Seq":
    """加载 ONNX 引擎（按目录缓存，避免重复加载 ~360MB 模型）。"""
    key = str(model_dir)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = OnnxSeq2Seq(key)
    return _ENGINE_CACHE[key]


def translate(texts, model_dir: str = DEFAULT_MODEL_DIR, *, max_length: int = 60,
              num_beams: int = 3, batch_size: int = 8, length_penalty: float = 1.0):
    """只做推理：``str`` 进 ``str`` 出，``list[str]`` 进 ``list[str]`` 出。

    ::

        translate("The cat is sleeping on the sofa.")
        translate(["I love you.", "Hello!"], num_beams=5)
    """
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    engine = load_model(model_dir)
    outs = engine.translate(batch, max_length=int(max_length), num_beams=int(num_beams),
                            batch_size=int(batch_size), length_penalty=float(length_penalty))
    return outs[0] if single else outs


__all__ = ["OnnxSeq2Seq", "load_model", "translate", "MODEL_DIR", "DEFAULT_MODEL_DIR"]


def read_inputs(args) -> list:
    if args.text:
        return list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as fp:
            return [line.strip() for line in fp if line.strip()]
    return []


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"[1/3] 加载 ONNX 模型：{args.model}")
    engine = OnnxSeq2Seq(args.model)
    print(f"      provider={engine.provider}  model_type={engine.meta.get('model_type')}  "
          f"vocab={engine.vocab_size}  beams={args.num_beams}")

    texts = read_inputs(args)

    # ---- 交互式 ----
    if not texts and not args.file:
        print("[2/3] 交互式翻译：输入英文句子回车（输入 q! 退出）")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line in ("q!", "quit", "exit"):
                break
            if not line:
                continue
            (translation,) = engine.translate([line], args.max_length, args.num_beams,
                                              args.batch_size, args.length_penalty,
                                              args.show_ids)
            print(f"  {translation}")
        return 0

    if not texts:
        print("没有输入（请用 --text/--file 或留空进入交互模式）")
        return 1

    # ---- 批量翻译 ----
    print(f"[2/3] 翻译 {len(texts)} 句（max_length={args.max_length}, num_beams={args.num_beams}）")
    translations = engine.translate(texts, args.max_length, args.num_beams,
                                    args.batch_size, args.length_penalty, args.show_ids)

    refs = None
    if args.compare_hf:
        print("      [对照] 用 transformers PyTorch 生成同样的句子 ...")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM

            src_id = engine.meta.get("source_model", "chou-lucas/transformer-en-zh")
            tok2 = engine.tokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained(src_id).eval()
            refs = []
            for start in range(0, len(texts), max(1, args.batch_size)):
                batch = texts[start:start + max(1, args.batch_size)]
                enc = tok2(batch, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    gen = model.generate(**enc, max_length=int(args.max_length),
                                         num_beams=int(args.num_beams), early_stopping=True)
                refs.extend(tok2.batch_decode(gen, skip_special_tokens=True))
        except Exception as exc:
            print(f"      [对照] 跳过（{type(exc).__name__}: {exc}）")

    # ---- 输出 ----
    lines = [f"{src}\t{tgt}" for src, tgt in zip(texts, translations)]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
        print(f"[3/3] 已写入：{args.out}")
    else:
        print("[3/3] 结果：")
        for i, (src, tgt) in enumerate(zip(texts, translations)):
            print(f"  EN: {src}")
            print(f"  ZH: {tgt}")
            if refs is not None:
                print(f"  HF: {refs[i]}")
            print("  " + "-" * 60)

    if refs is not None:
        same = sum(1 for a, b in zip(translations, refs) if a == b)
        print(f"与 HF PyTorch 结果完全一致：{same}/{len(refs)}")
    return 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
