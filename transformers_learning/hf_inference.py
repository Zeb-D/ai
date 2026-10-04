#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标准 HuggingFace 推理 Demo（英译中，零 .py 的内置 MarianMT 模型）。

演示"别人拿到 HF 仓库后如何**直接用**"——**不需要 `trust_remote_code`**，也不需要本工程的任何代码：

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("chou-lucas/transformer-en-zh")
    model = AutoModelForSeq2SeqLM.from_pretrained("chou-lucas/transformer-en-zh")
    out = model.generate(**tok(["The cat is sleeping on the sofa."], return_tensors="pt"),
                         max_length=60, num_beams=3)
    print(tok.batch_decode(out, skip_special_tokens=True))

依赖：``pip install "transformers>=5" torch sentencepiece sacremoses``

用法：

    # 1) 直接翻译若干句
    python hf_inference.py --text "The cat is sleeping on the sofa." --text "I love you."

    # 2) 翻译整个文件（每行一句）
    python hf_inference.py --file input.txt --out output.txt

    # 3) 交互式（不给 --text/--file 时默认进入）
    python hf_inference.py

    # 4) 用本地目录（例如训练产物）而不是 HF 仓库
    python hf_inference.py --model data/train/marian_exp --text "I love you."

    # 5) 兼容旧的"自定义架构"仓库（带 .py 时需要 trust_remote_code）
    python hf_inference.py --model chou-lucas/xxx --trust-remote-code --text "I love you."
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="hf_inference.py",
        description="标准 HuggingFace 推理 Demo：英译中（内置 MarianMT，默认无需 trust_remote_code）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="chou-lucas/transformer-en-zh",
                        help="HF 仓库 id 或本地目录")
    parser.add_argument("--text", action="append", default=None,
                        help="要翻译的英文句子，可重复指定")
    parser.add_argument("--file", default=None, help="输入文件（每行一句英文）")
    parser.add_argument("--out", default=None, help="把结果写到该文件（默认打印到终端）")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--max-length", type=int, default=60, help="生成长度上限")
    parser.add_argument("--num-beams", type=int, default=3, help="beam search 宽度；1 表示贪心")
    parser.add_argument("--batch-size", type=int, default=8, help="批量推理的 batch 大小")
    parser.add_argument("--show-ids", action="store_true", help="额外打印源/目标 token id")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="仅当模型仓库是自定义架构（带 .py）时才需要")
    return parser.parse_args(argv)


def pick_device(pref: str) -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str, trust_remote_code: bool = False):
    """按 HF 标准方式加载（默认不开启 trust_remote_code，证明是零 .py 的内置架构）。"""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                 trust_remote_code=trust_remote_code).to(device)
    model.eval()
    return tokenizer, model


def translate(tokenizer, model, texts, device: str, max_length: int, num_beams: int,
              batch_size: int, show_ids: bool = False):
    """批量英译中；返回译文列表。"""
    import torch

    results = []
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start:start + max(1, batch_size)]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=int(max_length),
                num_beams=int(num_beams),
                early_stopping=True,
            )
        results.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        if show_ids:
            for src, tgt in zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=False)):
                print(f"    src ids: {encoded['input_ids'][batch.index(src)].tolist()}")
                print(f"    tgt ids: {generated[batch.index(src)].tolist()}")
    return results


def read_inputs(args) -> list:
    if args.text:
        return list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as fp:
            return [line.strip() for line in fp if line.strip()]
    return []


def main(argv=None) -> int:
    args = parse_args(argv)
    device = pick_device(args.device)

    print(f"[1/3] 加载模型：{args.model}  (device={device}, trust_remote_code={args.trust_remote_code})")
    tokenizer, model = load_model(args.model, device, args.trust_remote_code)
    print(f"      tokenizer={type(tokenizer).__name__}  model={type(model).__name__}  "
          f"model_type={model.config.model_type}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

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
            (translation,) = translate(tokenizer, model, [line], device,
                                       args.max_length, args.num_beams, args.batch_size, args.show_ids)
            print(f"  {translation}")
        return 0

    if not texts:
        print("没有输入（请用 --text/--file 或留空进入交互模式）")
        return 1

    # ---- 批量翻译 ----
    print(f"[2/3] 翻译 {len(texts)} 句（max_length={args.max_length}, num_beams={args.num_beams}）")
    translations = translate(tokenizer, model, texts, device,
                             args.max_length, args.num_beams, args.batch_size, args.show_ids)

    # ---- 输出 ----
    lines = [f"{src}\t{tgt}" for src, tgt in zip(texts, translations)]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
        print(f"[3/3] 已写入：{args.out}")
    else:
        print("[3/3] 结果：")
        for src, tgt in zip(texts, translations):
            print(f"  EN: {src}")
            print(f"  ZH: {tgt}")
            print("  " + "-" * 60)
    return 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
