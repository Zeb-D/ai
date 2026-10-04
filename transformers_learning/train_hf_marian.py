#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""内置架构 MarianMT 训练 + 发布（**零 .py**，主流 HF 格式）。

为什么用 ``MarianMTModel``：

* 它是 ``transformers`` **内置**架构（代码在库里），所以仓库里**只有**：::

      config.json  generation_config.json  model.safetensors
      tokenizer_config.json  vocab.json  target_vocab.json  source.spm  target.spm  README.md

  **没有任何 .py**，加载时**不需要** ``trust_remote_code``：

      from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
      tok = AutoTokenizer.from_pretrained("chou-lucas/transformer-en-zh")
      model = AutoModelForSeq2SeqLM.from_pretrained("chou-lucas/transformer-en-zh")

* 这正是 ``Helsinki-NLP/opus-mt-*`` 系列的做法：``MarianTokenizer`` 用
  ``separate_vocabs=True`` 支持"源端 / 目标端各自一套 SentencePiece"，与我们的
  ``tokenizer/eng.model`` / ``tokenizer/chn.model`` 完全契合（模型侧共享一份 embedding，
  属于 Marian 的标准设计）。

与 ``train_hf.py``（自定义架构 + ``trust_remote_code``）的区别：本脚本产出零 ``.py`` 仓库，
但需要**从头训练**（Marian 的共享 embedding 与自定义模型的独立 embedding 不通用）。

依赖：``pip install "transformers>=5" torch datasets accelerate sacrebleu sentencepiece sacremoses safetensors``

用法（在 transformers_learning 目录下执行）：

    python train_hf_marian.py                                   # 用 config.py 超参，从头训练
    python train_hf_marian.py --max-train-samples 2000 --epochs 1     # 快速验证
    python train_hf_marian.py --push-to-hub --hub-model-id chou-lucas/transformer-en-zh
    python train_hf_marian.py --publish-only data/train/marian_exp --hub-model-id chou-lucas/transformer-en-zh

默认行为：若输出目录里已有上一次训练的 ``checkpoint-*``（或已保存的模型），会自动**接着上次的
产物继续训练**（恢复 模型/优化器/调度器/步数）。用 ``--additional-epochs N`` 表示"再练 N 轮"，
``--no-auto-resume`` 可关闭该行为从头训练。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# 内存 / 显存水位与回收：必须在 import torch 之前设置才生效（MPS / CUDA）
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config  # noqa: E402
from tools.tokenizer_utils import DEFAULT_SOURCE_SPM, DEFAULT_TARGET_SPM  # noqa: E402

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("train_marian")

# 旧的"自定义架构"仓库里的代码文件；发布新仓库时若存在则删除，保证零 .py
STALE_CUSTOM_CODE = ["modeling_transformer.py", "configuration_transformer.py",
                     "tokenization_transformer.py"]


# ===========================================================================
# 内存回收（与 train_hf.py 一致：MPS 上必须周期性释放）
# ===========================================================================
def release_memory(tag: str = "") -> None:
    gc.collect()
    freed = []
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            freed.append("mps")
        except Exception:  # noqa: BLE001
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            freed.append("cuda")
        except Exception:  # noqa: BLE001
            pass
    if freed:
        LOGGER.debug("已释放 %s 缓存%s", "+".join(freed), f"（{tag}）" if tag else "")


def process_rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 0.0


def device_allocated_gb() -> float:
    try:
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / (1024 ** 3)
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def make_memory_callback():
    from transformers import TrainerCallback

    class MemoryCleanupCallback(TrainerCallback):
        def __init__(self, every_steps: int = 50, verbose: bool = False):
            self.every_steps = int(every_steps)
            self.verbose = bool(verbose)

        def _clean(self, tag: str) -> None:
            release_memory(tag=tag)
            if self.verbose:
                LOGGER.info("[内存] %s：RSS=%.2fGB, 设备已分配=%.2fGB",
                            tag, process_rss_gb(), device_allocated_gb())

        def on_step_end(self, args, state, control, **kwargs):
            if self.every_steps > 0 and state.global_step % self.every_steps == 0:
                self._clean(f"step {state.global_step}")
            return control

        def on_log(self, args, state, control, **kwargs):
            if self.verbose:
                LOGGER.info("[内存] step %d：RSS=%.2fGB, 设备已分配=%.2fGB",
                            state.global_step, process_rss_gb(), device_allocated_gb())
            return control

        def on_evaluate(self, args, state, control, **kwargs):
            self._clean(f"eval @ step {state.global_step}")
            return control

        def on_save(self, args, state, control, **kwargs):
            self._clean(f"save @ step {state.global_step}")
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            self._clean(f"epoch {state.epoch}")
            return control

    return MemoryCleanupCallback


# ===========================================================================
# 模型 / 分词器
# ===========================================================================
def build_marian_config(args):
    from transformers import MarianConfig

    # 注意（transformers v5 的 MarianConfig 真实字段）：
    #   * share_encoder_decoder_embeddings：默认 True 表示"源/目标共用一份 embedding"，
    #     这只在**源/目标使用同一套联合词表**（如 opus-mt）时才成立。我们是两套独立 spm
    #     （id 语义不同），必须设为 False，让 encoder/decoder 各自持有 embedding，
    #     否则会像共享词表一样互相冲突、训练不收敛（BLEU≈0）。
    #   * scale_embedding：embedding 乘 sqrt(d_model)（等价于旧写的 normalize_embedding）。
    #   * decoder_vocab_size：不共享时必须显式给出目标端词表大小。
    return MarianConfig(
        vocab_size=int(args.vocab_size),                     # 源端词表
        decoder_vocab_size=int(args.target_vocab_size),      # 目标端词表
        d_model=int(args.d_model),
        encoder_layers=int(args.layers),
        decoder_layers=int(args.layers),
        encoder_attention_heads=int(args.heads),
        decoder_attention_heads=int(args.heads),
        encoder_ffn_dim=int(args.d_ff),
        decoder_ffn_dim=int(args.d_ff),
        activation_function=args.activation_function,
        scale_embedding=True,                       # embedding 乘 sqrt(d_model)
        max_position_embeddings=int(args.max_position_embeddings),
        dropout=float(args.dropout),
        attention_dropout=float(args.dropout),
        activation_dropout=float(args.dropout),
        pad_token_id=int(args.pad_token_id),
        bos_token_id=int(args.bos_token_id),
        eos_token_id=int(args.eos_token_id),
        decoder_start_token_id=int(args.decoder_start_token_id),
        forced_eos_token_id=None,                   # 不强制在 max_length 处用 pad(0) 充当 EOS
        share_encoder_decoder_embeddings=False,     # 源/目标独立 embedding（关键修复）
        tie_word_embeddings=False,
    )


def build_tokenizer(args):
    from transformers import MarianTokenizer

    vocab = args.source_vocab or os.path.join(os.path.dirname(os.path.abspath(args.source_spm)),
                                              "vocab.json")
    tgt_vocab = args.target_vocab or os.path.join(os.path.dirname(os.path.abspath(args.target_spm)),
                                                  "target_vocab.json")
    # 若 HF 词表文件不存在，则从 spm 现场生成
    if not os.path.isfile(vocab) or not os.path.isfile(tgt_vocab):
        from tokenizer.tokenization_transformer import build_vocab_files

        paths = build_vocab_files(args.source_spm, args.target_spm,
                                  os.path.dirname(os.path.abspath(vocab)))
        vocab, tgt_vocab = paths["vocab.json"], paths["target_vocab.json"]

    return MarianTokenizer(
        source_spm=args.source_spm,
        target_spm=args.target_spm,
        vocab=vocab,
        target_vocab_file=tgt_vocab,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        bos_token="<s>",
        model_max_length=int(args.max_source_length),
        separate_vocabs=True,
    )


# ===========================================================================
# 数据 / 指标
# ===========================================================================
def read_pairs(path: str, limit: int = 0) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    if limit:
        data = data[:limit]
    return [{"en": row[0], "zh": row[1]} for row in data]


def build_datasets(args, tokenizer):
    from datasets import Dataset

    raw_train = Dataset.from_list(read_pairs(args.train_file, args.max_train_samples))
    raw_eval = Dataset.from_list(read_pairs(args.dev_file, args.max_eval_samples))

    def preprocess(batch):
        # Marian 约定：源端 tokens + EOS；目标端 labels = tokens + EOS
        # （BOS 由模型 shift_tokens_right 用 decoder_start_token_id 补）
        model_inputs = tokenizer(batch["en"], max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["zh"], max_length=args.max_target_length,
                           truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns = raw_train.column_names
    train_ds = raw_train.map(preprocess, batched=True, remove_columns=columns, desc="Tokenizing train")
    eval_ds = raw_eval.map(preprocess, batched=True, remove_columns=columns, desc="Tokenizing dev")
    return train_ds, eval_ds


def build_compute_metrics(tokenizer, pad_token_id: int):
    import sacrebleu

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions).copy()
        labels = np.asarray(labels).copy()
        predictions[predictions == -100] = pad_token_id
        labels[labels == -100] = pad_token_id
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels], tokenize="zh")
        return {"bleu": round(float(bleu.score), 4)}

    return compute_metrics


# ===========================================================================
# 生成配置 / 模型卡
# ===========================================================================
def build_generation_config():
    from transformers import GenerationConfig

    return GenerationConfig(
        bos_token_id=int(config.bos_idx),
        eos_token_id=int(config.eos_idx),
        pad_token_id=int(config.padding_idx),
        decoder_start_token_id=int(config.bos_idx),
        max_length=int(config.max_len),
        min_length=0,
        num_beams=int(config.beam_size),
        num_return_sequences=1,
        early_stopping=True,
        length_penalty=1.0,
        do_sample=False,
        repetition_penalty=1.0,
    )


def build_model_card(repo_id: str, cfg, gen) -> str:
    return f"""---
language:
- en
- zh
library_name: transformers
pipeline_tag: translation
tags:
- translation
- marian
- encoder-decoder
- sentencepiece
---

# Transformer en → zh (MarianMT, standard HF)

基于 **transformers 内置架构 `MarianMTModel`** 训练的英译中模型（与 `Helsinki-NLP/opus-mt-*` 同类）。
仓库为**标准 HF 格式**：`config.json` + `model.safetensors` + 分词器文件，
**不含任何 .py**，加载**无需 `trust_remote_code`**。

## 快速开始

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

repo = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSeq2SeqLM.from_pretrained(repo)

sentences = ["The government has implemented various policies to improve the living standards of its citizens."]
inputs = tokenizer(sentences, return_tensors="pt", padding=True)
out = model.generate(**inputs, max_length={gen.max_length}, num_beams={gen.num_beams})
print(tokenizer.batch_decode(out, skip_special_tokens=True))
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `config.json` | `model_type = marian`（内置架构，无需自定义代码） |
| `model.safetensors` | 权重（HF 首选格式） |
| `generation_config.json` | 生成默认值（`max_length={gen.max_length}`、`num_beams={gen.num_beams}`） |
| `tokenizer_config.json` | `MarianTokenizer`（`separate_vocabs=True`） |
| `vocab.json` / `target_vocab.json` | 英文 / 中文词表 |
| `source.spm` / `target.spm` | 英文 / 中文 SentencePiece 模型 |

## 架构

| 项目 | 值 |
| --- | --- |
| model_type | marian（内置） |
| d_model / heads / layers | {int(cfg.d_model)} / {int(cfg.encoder_attention_heads)} / {int(cfg.encoder_layers)} |
| ffn dim | {int(cfg.encoder_ffn_dim)} |
| 位置编码 | 固定 sin-cos（Marian 内置） |
| embedding | 源/目标**各自独立** embedding（`share_encoder_decoder_embeddings=False`），并乘 sqrt(d_model)（`scale_embedding=True`） |
| 分词 | 源/目标各自 SentencePiece（`separate_vocabs=True`）；pad/unk/bos/eos = {int(cfg.pad_token_id)}/{int(getattr(cfg, "unk_token_id", 1))}/{int(cfg.bos_token_id)}/{int(cfg.eos_token_id)} |
"""


# ===========================================================================
# 发布（零 .py）
# ===========================================================================
def write_model_card_from_dir(src_dir: str, repo_id: str) -> None:
    """用本地目录里的 config.json / generation_config.json 生成模型卡（README.md）。

    这样即使训练末段写卡步骤异常（例如旧版用到了 MarianConfig 不存在的字段），
    ``--publish-only`` 也能补上完整的模型卡再上传。
    """
    from transformers import GenerationConfig, MarianConfig

    cfg = MarianConfig.from_pretrained(src_dir)
    gen_path = os.path.join(src_dir, "generation_config.json")
    gen = GenerationConfig.from_pretrained(src_dir) if os.path.isfile(gen_path) else build_generation_config()
    card_path = os.path.join(src_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as fp:
        fp.write(build_model_card(repo_id, cfg, gen))
    LOGGER.info("已生成模型卡: %s", card_path)


def publish_local_repo(src_dir: str, repo_id: str, token: str, private: bool,
                       verify: bool = True) -> dict:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)

    # 模型卡缺失或为空时重新生成，保证上传的仓库有完整 README
    card_path = os.path.join(src_dir, "README.md")
    if not os.path.isfile(card_path) or os.path.getsize(card_path) == 0:
        write_model_card_from_dir(src_dir, repo_id)

    # 确保 config.json 打开 use_cache：推理时启用 KV Cache（增量解码）。
    # 训练期 HF Trainer 会把它置为 False，若原样发布则第三人 generate() 默认不走 cache。
    cfg_path = os.path.join(src_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fp:
            _cfg = json.load(fp)
        if _cfg.get("use_cache") is not True:
            _cfg["use_cache"] = True
            with open(cfg_path, "w", encoding="utf-8") as fp:
                json.dump(_cfg, fp, ensure_ascii=False, indent=2)
            LOGGER.info("已将 config.json 的 use_cache 设为 true（启用 KV Cache 增量解码）")

    # 删除旧"自定义架构"遗留的 .py，确保仓库零 .py
    existing = {s.rfilename for s in api.model_info(repo_id, token=token).siblings}
    for name in STALE_CUSTOM_CODE:
        if name in existing:
            api.delete_file(path_in_repo=name, repo_id=repo_id, token=token,
                            commit_message=f"Remove {name} (migrate to built-in Marian, no custom code)")
            LOGGER.info("已删除旧仓库中的自定义代码文件: %s", name)

    ignores = ["checkpoint-*", "*.pth", "optimizer.pt", "scheduler.pt",
               "training_args.bin", "trainer_state.json", ".DS_Store"]
    t0 = time.time()
    commit = api.upload_folder(repo_id=repo_id, folder_path=src_dir, token=token,
                               commit_message="Upload standard HF MarianMT (en-zh, no custom code)",
                               ignore_patterns=ignores)
    report = {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
              "duration_s": round(time.time() - t0, 1), "revision": getattr(commit, "oid", None)}
    LOGGER.info("上传完成 %.1fs: %s", report["duration_s"], report["url"])

    files = sorted(s.rfilename for s in api.model_info(repo_id, token=token).siblings)
    py = [f for f in files if f.endswith(".py")]
    LOGGER.info("仓库文件: %s", files)
    LOGGER.info("仓库中的 .py 文件: %s", py or "无 ✅")
    report["files"] = files
    report["py_files"] = py

    if verify:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(repo_id, token=token)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, token=token).eval()
        sents = ["The cat is sleeping on the sofa.",
                 "The government has implemented various policies to improve the living standards of its citizens."]
        with torch.no_grad():
            out = model.generate(**tok(sents, return_tensors="pt", padding=True),
                                 max_length=int(config.max_len), num_beams=int(config.beam_size))
        report["verify"] = {"loaded_without_trust_remote_code": True,
                            "translations": tok.batch_decode(out, skip_special_tokens=True)}
        LOGGER.info("[校验] 无需 trust_remote_code 加载成功：%s", report["verify"]["translations"])
    return report


# ===========================================================================
# 默认"接着上次训练产物继续训练"
# ===========================================================================
def find_latest_checkpoint(output_dir: str) -> str | None:
    """返回 ``output_dir`` 下 step 最大的 ``checkpoint-*`` 目录（需含 trainer_state.json）。"""
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if (name.startswith("checkpoint-") and os.path.isdir(path)
                and os.path.isfile(os.path.join(path, "trainer_state.json"))):
            try:
                step = int(name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def read_epoch_from_checkpoint(ckpt_dir: str) -> float:
    """从 checkpoint 的 trainer_state.json 读取上次训练到第几个 epoch。"""
    try:
        with open(os.path.join(ckpt_dir, "trainer_state.json"), encoding="utf-8") as fp:
            return float(json.load(fp).get("epoch") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def auto_detect_resume(args) -> None:
    """默认行为：自动发现"上一次训练产物"并接着训练。

    优先级：
      1) ``output_dir`` 下最新的 ``checkpoint-*`` → 用 ``resume_from_checkpoint`` 续训
         （恢复 模型 + 优化器 + 调度器 + 步数，HF 标准做法）；
      2) ``output_dir`` 下已保存的模型（``model.safetensors`` + ``config.json``）→ 仅加载权重继续；
      3) 都没有 → 从头训练。

    可用 ``--additional-epochs N`` 表示"在已训练的轮数上再练 N 轮"；显式传
    ``--resume-from-checkpoint`` / ``--init-from`` 时不自动推断；``--no-auto-resume`` 关闭本行为。
    """
    if not args.auto_resume:
        if args.epochs is None:
            args.epochs = float(config.epoch_num)
        LOGGER.info("已关闭自动续训（--no-auto-resume），从头训练，总 epochs=%s", args.epochs)
        return

    out = args.output_dir
    prev_epoch = 0.0

    if args.resume_from_checkpoint is None and args.init_from is None:
        ckpt = find_latest_checkpoint(out)
        if ckpt:
            args.resume_from_checkpoint = ckpt
            prev_epoch = read_epoch_from_checkpoint(ckpt)
            LOGGER.info("[自动续训] 发现最新 checkpoint：%s（上次 epoch=%.2f）", ckpt, prev_epoch)
        elif (os.path.isfile(os.path.join(out, "config.json"))
              and os.path.isfile(os.path.join(out, "model.safetensors"))):
            args.init_from = out
            LOGGER.info("[自动续训] 未发现 checkpoint，改为从已保存模型继续（仅权重，优化器状态重置）：%s", out)

    if args.resume_from_checkpoint:
        prev_epoch = read_epoch_from_checkpoint(args.resume_from_checkpoint)

    # 计算本次"总 epochs"
    if args.additional_epochs:
        args.epochs = math.ceil(prev_epoch) + float(args.additional_epochs)
        LOGGER.info("[自动续训] 在已训练 %.2f 轮基础上追加 %s 轮 → 本次总 epochs=%d",
                    prev_epoch, args.additional_epochs, int(math.ceil(args.epochs)))
    elif args.epochs is None:
        args.epochs = math.ceil(prev_epoch) + float(config.epoch_num)
        LOGGER.info("[自动续训] 未指定 --epochs → 默认 %s 轮：上次 %.2f + config.epoch_num %s = 总 %d",
                    args.epochs, prev_epoch, config.epoch_num, int(math.ceil(args.epochs)))
    elif prev_epoch > 0 and args.epochs <= prev_epoch:
        args.epochs = math.ceil(prev_epoch) + max(1.0, float(args.epochs))
        LOGGER.warning("[自动续训] --epochs=%s 不大于上次 epoch=%.2f，已自动调整为总 epochs=%d",
                       args.epochs, prev_epoch, int(math.ceil(args.epochs)))
    else:
        LOGGER.info("[自动续训] 本次总 epochs=%s（上次 epoch=%.2f）", args.epochs, prev_epoch)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="train_hf_marian.py",
        description="内置架构 MarianMT 训练 + 零 .py 发布（标准 HF 格式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-file", default=config.train_data_path)
    parser.add_argument("--dev-file", default=config.dev_data_path)
    parser.add_argument("--output-dir", default=os.path.join(config.model_dir, "marian_exp"))
    parser.add_argument("--init-from", default=None, help="从已保存的 Marian 目录继续训练")

    parser.add_argument("--source-spm", default=DEFAULT_SOURCE_SPM)
    parser.add_argument("--target-spm", default=DEFAULT_TARGET_SPM)
    parser.add_argument("--source-vocab", default=None, help="默认取 source-spm 同目录的 vocab.json")
    parser.add_argument("--target-vocab", default=None, help="默认取 target-spm 同目录的 target_vocab.json")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--vocab-size", type=int, default=int(config.src_vocab_size),
                        help="源端词表大小（vocab_size）")
    parser.add_argument("--target-vocab-size", type=int, default=int(config.tgt_vocab_size),
                        help="目标端词表大小（decoder_vocab_size；源/目标独立 embedding 时必须正确）")

    # 结构（默认取 config.py）
    parser.add_argument("--d-model", type=int, default=config.d_model)
    parser.add_argument("--layers", type=int, default=config.n_layers)
    parser.add_argument("--heads", type=int, default=config.n_heads)
    parser.add_argument("--d-ff", type=int, default=config.d_ff)
    parser.add_argument("--dropout", type=float, default=config.dropout)
    parser.add_argument("--activation-function", default="relu")
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--pad-token-id", type=int, default=int(config.padding_idx))
    parser.add_argument("--unk-token-id", type=int, default=1)
    parser.add_argument("--bos-token-id", type=int, default=int(config.bos_idx))
    parser.add_argument("--eos-token-id", type=int, default=int(config.eos_idx))
    parser.add_argument("--decoder-start-token-id", type=int, default=int(config.bos_idx))

    # 训练超参
    parser.add_argument("--epochs", type=float, default=None,
                        help="本次训练的总轮数；默认：续训时=上次轮数+config.epoch_num，否则=config.epoch_num")
    parser.add_argument("--additional-epochs", type=float, default=3,
                        help="自动续训时额外追加的轮数（优先级高于 --epochs，如再练 12 轮）")
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false",
                        help="关闭默认自动续训（不自动查找 checkpoint/已保存模型，直接从头训练）")
    parser.add_argument("--batch-size", type=int, default=int(config.batch_size))
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--lr-scheduler-type", default="inverse_sqrt")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-pin-memory", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--memory-cleanup-steps", type=int, default=50)
    parser.add_argument("--print-memory", action="store_true")
    parser.add_argument("--mps-memory-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 表示全部")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 表示全部")
    parser.add_argument("--resume-from-checkpoint", default=None)

    # 发布
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--publish-only", default=None, help="只发布已存在的本地仓库目录")
    parser.add_argument("--hub-model-id", default=None, help="例如 chou-lucas/transformer-en-zh")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--hub-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- 仅发布模式 ----
    if args.publish_only:
        if not args.hub_model_id:
            raise SystemExit("--publish-only 需要 --hub-model-id")
        token = args.hub_token or os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("发布需要 token：--hub-token 或 HF_TOKEN")
        report = publish_local_repo(args.publish_only, args.hub_model_id, token,
                                    args.private, args.verify)
        LOGGER.info("发布报告: %s", json.dumps(report, ensure_ascii=False)[:500])
        return 0

    # ---- 默认：自动发现"上一次训练产物"并接着训练 ----
    auto_detect_resume(args)

    # ---- 依赖检查 ----
    try:
        from transformers import (DataCollatorForSeq2Seq, MarianMTModel, Seq2SeqTrainer,
                                  Seq2SeqTrainingArguments)
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(f"缺少 transformers：pip install 'transformers>=5'  ({exc})")
    try:
        import datasets  # noqa: F401
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(f"缺少 datasets：pip install datasets  ({exc})")

    logging.getLogger("transformers").setLevel(logging.WARNING)

    # ---- 内存设置 ----
    if args.mps_memory_fraction is not None and torch.backends.mps.is_available():
        fraction = float(args.mps_memory_fraction)
        torch.mps.set_per_process_memory_fraction(fraction)
        LOGGER.info("已限制 MPS 可用内存比例为 %.0f%%", fraction * 100)
    if args.dataloader_pin_memory == "on":
        pin_memory = True
    elif args.dataloader_pin_memory == "off":
        pin_memory = False
    else:
        pin_memory = torch.cuda.is_available()
    LOGGER.info("dataloader pin_memory=%s；内存回收周期=%s",
                pin_memory, f"{args.memory_cleanup_steps} step" if args.memory_cleanup_steps > 0 else "关闭")
    release_memory(tag="start")

    # ---- 分词器 / 模型 ----
    tokenizer = build_tokenizer(args)
    if args.init_from:
        LOGGER.info("从已保存的 Marian 目录继续训练: %s", args.init_from)
        model = MarianMTModel.from_pretrained(args.init_from)
    else:
        model = MarianMTModel(build_marian_config(args))
    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("模型参数：%.2fM（model_type=%s）", n_params / 1e6, model.config.model_type)

    # ---- 数据 ----
    train_ds, eval_ds = build_datasets(args, tokenizer)
    LOGGER.info("训练样本 %d，验证样本 %d", len(train_ds), len(eval_ds))
    steps_per_epoch = max(1, len(train_ds) // max(1, args.batch_size * args.grad_accum))
    LOGGER.info("每 epoch ≈ %d 步", steps_per_epoch)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # Marian 内置损失使用 CrossEntropyLoss(ignore_index=-100)
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # ---- 训练参数 ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=int(config.beam_size),
        load_best_model_at_end=(args.eval_strategy != "no"),
        metric_for_best_model="bleu",
        greater_is_better=True,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=pin_memory,
        report_to=[],
        seed=args.seed,
        label_names=["labels"],
    )

    MemoryCleanupCallback = make_memory_callback()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, int(model.config.pad_token_id)),
        callbacks=[MemoryCleanupCallback(every_steps=args.memory_cleanup_steps, verbose=args.print_memory)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    release_memory(tag="train finished")

    # ---- 保存标准 HF 产物（零 .py）----
    # 训练时 HF Trainer 会把 config.use_cache 置为 False；推理需要 KV Cache 增量解码，
    # 保存前显式打开（只影响生成是否复用 past_key_values，不影响权重与数值结果）。
    model.config.use_cache = True
    model.generation_config = build_generation_config()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as fp:
        fp.write(build_model_card(args.hub_model_id or "your-name/transformer-en-zh",
                                  model.config, model.generation_config))
    LOGGER.info("已保存标准 HF（零 .py）产物到: %s", args.output_dir)
    py = [f for f in os.listdir(args.output_dir) if f.endswith(".py")]
    LOGGER.info("本地产物中的 .py 文件: %s", py or "无 ✅")
    if getattr(trainer, "state", None) is not None and trainer.state.best_metric is not None:
        LOGGER.info("最佳验证 BLEU: %.4f", trainer.state.best_metric)

    # ---- 发布 ----
    if args.push_to_hub:
        if not args.hub_model_id:
            raise SystemExit("--push-to-hub 需要 --hub-model-id")
        token = args.hub_token or os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("推送需要 token：--hub-token 或 HF_TOKEN")
        publish_local_repo(args.output_dir, args.hub_model_id, token, args.private, args.verify)
    return 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
