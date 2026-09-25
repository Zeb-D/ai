#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 onnxruntime 加载 export_onnx_hf.py 导出的 HuggingFace 风格 ONNX 目录做英译中推理。

不依赖 PyTorch（只有 --compare 做逐 token 对照时才 import torch），全部信息来自导出目录里的
自包含文件：

    config.json              模型结构 + pad/unk/bos/eos id
    generation_config.json   max_length / num_beams / early_stopping / length_penalty
    tokenizer_config.json    sentencepiece 文件名 + special tokens
    source.spm / target.spm  英文 / 中文分词模型
    encoder_model.onnx       input_ids(int64,B,S) + attention_mask(int64,B,S) -> last_hidden_state(float32,B,S,d_model)
    decoder_model.onnx       input_ids(int64,B,T) + encoder_hidden_states + encoder_attention_mask -> logits(float32,B,T,V)

用法（在 transformers_learning 目录下执行）：

    # 1) 交互式循环翻译（默认，和 translate_main.py 一样的体验：q! 退出）
    python translate_onnx_hf.py

    # 2) 指定句子 / 指定 dev.json 句数（非交互）
    python translate_onnx_hf.py --text "The cat is sleeping on the sofa."
    python translate_onnx_hf.py --from-dev 5 --decode beam --no-compare

    # 3) 直接用 HuggingFace Hub 上的模型
    python translate_onnx_hf.py --repo-id chou-lucas/transformer-en-zh-base --from-dev 3

    # 4) 下载验证（真实性复现）：拉取 -> 逐文件哈希 -> HF/ONNX/本地 PyTorch 三方逐 token 对照
    python translate_onnx_hf.py --repo-id chou-lucas/transformer-en-zh-base --verify-hub --from-dev 3
    python translate_onnx_hf.py --repo-id chou-lucas/transformer-en-zh-base --verify-hub --no-hub-transformers

也可以当库用：

    from translate_onnx_hf import Seq2SeqOnnxModel, one_sentence_translate
    model = Seq2SeqOnnxModel.from_pretrained("chou-lucas/transformer-en-zh-base")   # 也可传本地目录
    text = one_sentence_translate("I love you.", model,
                                  model.source_sp, model.target_sp,
                                  model.bos_token_id, model.eos_token_id)
    print(text)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config  # noqa: E402

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("translate_onnx_hf")

ENCODER_FILE = "encoder_model.onnx"
DECODER_FILE = "decoder_model.onnx"


# ===========================================================================
# 1) ONNX 模型封装：加载 + 编码 + 自回归解码
# ===========================================================================
class Seq2SeqOnnxModel:
    """onnxruntime 加载 HF 风格 ONNX 目录；提供贪心与束搜索解码。"""

    def __init__(self, model_dir, encoder_session, decoder_session,
                 model_config: dict, generation_config: dict, output_mode: str):
        self.model_dir = model_dir
        self.encoder = encoder_session
        self.decoder = decoder_session
        self.config = model_config
        self.generation_config = generation_config
        self.output_mode = output_mode
        self.d_model = int(model_config.get("d_model", config.d_model))

        self.pad_token_id = int(model_config.get("pad_token_id", 0))
        self.unk_token_id = int(model_config.get("unk_token_id", 1))
        self.bos_token_id = int(model_config.get("bos_token_id", 2))
        self.eos_token_id = int(model_config.get("eos_token_id", 3))
        self.decoder_start_token_id = int(model_config.get("decoder_start_token_id",
                                                           self.bos_token_id))

        self.max_length = int(generation_config.get("max_length", config.max_len))
        self.num_beams = int(generation_config.get("num_beams", 1))
        self.early_stopping = bool(generation_config.get("early_stopping", True))
        self.length_penalty = float(generation_config.get("length_penalty", 1.0))

        self.source_sp = None
        self.target_sp = None

    # ---------------- 加载 ----------------
    @classmethod
    def from_pretrained(cls, model_dir: str, coreml: bool = False,
                        verbose: bool = True, revision: str = None,
                        local_dir: str = None, token: str = None) -> "Seq2SeqOnnxModel":
        import onnxruntime as ort

        # model_dir 既可以是本地目录，也可以直接是 HF 仓库 id（如 chou-lucas/transformer-en-zh-base）
        if looks_like_repo_id(model_dir):
            model_dir, _ = download_from_hub(model_dir, revision=revision, local_dir=local_dir,
                                             token=token, verbose=verbose)
        model_dir = os.path.abspath(model_dir)
        missing = [name for name in (ENCODER_FILE, DECODER_FILE, "config.json")
                   if not os.path.isfile(os.path.join(model_dir, name))]
        if missing:
            raise FileNotFoundError(f"{model_dir} 缺少 {missing}；可用 --repo-id <user/repo> 从 Hub 拉取，"
                                    f"或先执行 python export_onnx_hf.py --out-dir {model_dir}")

        def _read(name) -> dict:
            path = os.path.join(model_dir, name)
            if not os.path.isfile(path):
                return {}
            with open(path, encoding="utf-8") as fp:
                return json.load(fp)

        model_config = _read("config.json")
        generation_config = _read("generation_config.json")
        tokenizer_config = _read("tokenizer_config.json")

        # 默认只用 CPU：CoreML/GPU 可能把算子降精度，破坏与 PyTorch 的逐 token 一致性
        providers = ["CPUExecutionProvider"]
        if coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        # 模型是固定 shape 之外的动态维，开启全部图优化即可
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        encoder = ort.InferenceSession(os.path.join(model_dir, ENCODER_FILE),
                                       sess_options=session_options, providers=providers)
        decoder = ort.InferenceSession(os.path.join(model_dir, DECODER_FILE),
                                       sess_options=session_options, providers=providers)

        obj = cls(model_dir, encoder, decoder, model_config, generation_config,
                  model_config.get("output_mode", "logits"))
        obj._load_tokenizers(tokenizer_config)
        if verbose:
            LOGGER.info("onnxruntime %s 已加载 %s（output_mode=%s, providers=%s）",
                        ort.__version__, model_dir, obj.output_mode, providers)
            LOGGER.info("模型输入: encoder %s / decoder %s",
                        [i.name for i in encoder.get_inputs()],
                        [i.name for i in decoder.get_inputs()])
            LOGGER.info("解码配置: max_length=%d, num_beams=%d, early_stopping=%s, length_penalty=%.2f",
                        obj.max_length, obj.num_beams, obj.early_stopping, obj.length_penalty)
        return obj

    def _load_tokenizers(self, tokenizer_config: dict) -> None:
        import sentencepiece as spm

        names = {"source_spm": "source.spm", "target_spm": "target.spm"}
        for key in names:
            if tokenizer_config.get(key):
                names[key] = tokenizer_config[key]
        self.pad_token_id = int(tokenizer_config.get("pad_token_id", self.pad_token_id))
        self.bos_token_id = int(tokenizer_config.get("bos_token_id", self.bos_token_id))
        self.eos_token_id = int(tokenizer_config.get("eos_token_id", self.eos_token_id))

        for attr, key in (("source_sp", "source_spm"), ("target_sp", "target_spm")):
            path = os.path.join(self.model_dir, names[key])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"缺少分词模型 {path}")
            proc = spm.SentencePieceProcessor()
            proc.Load(path)
            setattr(self, attr, proc)

    # ---------------- 文本 <-> id ----------------
    def _padded_ids(self, sequences):
        """id 序列列表 -> (input_ids, attention_mask)，用 pad_token_id 右侧补齐。"""
        length = max(len(seq) for seq in sequences)
        input_ids = np.full((len(sequences), length), self.pad_token_id, dtype=np.int64)
        for row, ids in enumerate(sequences):
            input_ids[row, :len(ids)] = ids
        attention_mask = (input_ids != self.pad_token_id).astype(np.int64)
        return input_ids, attention_mask

    def encode_text(self, texts):
        """英文文本 -> (input_ids, attention_mask)，与训练侧 collate_fn 同样的 [BOS] ... [EOS]。"""
        if isinstance(texts, str):
            texts = [texts]
        sequences = [[self.bos_token_id] + self.source_sp.EncodeAsIds(t) + [self.eos_token_id]
                     for t in texts]
        return self._padded_ids(sequences)

    def decode_text(self, ids) -> str:
        """token id -> 中文文本（跳过 pad/bos，遇到 eos 停止）。"""
        return self.target_sp.decode_ids(self._clean_ids(ids))

    def _clean_ids(self, ids) -> list:
        cleaned = []
        for tid in ids:
            tid = int(tid)
            if tid == self.eos_token_id:
                break
            if tid in (self.pad_token_id, self.bos_token_id):
                continue
            cleaned.append(tid)
        return cleaned

    # ---------------- ONNX 前向 ----------------
    def encoder_forward(self, input_ids, attention_mask):
        return self.encoder.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })[0]

    def encode_ids(self, sequences):
        """token id 序列（已含 BOS/EOS）-> (memory, attention_mask)。"""
        input_ids, attention_mask = self._padded_ids([[int(i) for i in s] for s in sequences])
        return self.encoder_forward(input_ids, attention_mask), attention_mask

    def encode(self, texts):
        """HF 风格入口：文本 -> (memory, attention_mask, input_ids)，encoder 只跑一次。"""
        input_ids, attention_mask = self.encode_text(texts)
        return self.encoder_forward(input_ids, attention_mask), attention_mask, input_ids

    def _last_log_probs(self, logits: np.ndarray) -> np.ndarray:
        """取 decoder 最后一步；输出是 logits 时补一个 log_softmax（beam 需要累加 log 概率）。"""
        last = logits[:, -1, :]
        if self.output_mode == "log_probs":
            return last
        shifted = last - last.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    def _decoder_step(self, input_ids: np.ndarray, memory: np.ndarray,
                      attention_mask: np.ndarray) -> np.ndarray:
        logits = self.decoder.run(None, {
            "input_ids": input_ids,
            "encoder_hidden_states": memory,
            "encoder_attention_mask": attention_mask,
        })[0]
        return self._last_log_probs(logits)

    # ---------------- 解码 ----------------
    def greedy_decode(self, memory, attention_mask, max_length: int = None,
                      trace: list = None) -> list:
        """贪心解码（每步只保留 argmax），返回 token id 列表。"""
        max_length = int(max_length or self.max_length)
        generated = [self.decoder_start_token_id]
        for step in range(max_length - 1):
            cur = np.array([generated], dtype=np.int64)
            log_probs = self._decoder_step(cur, memory, attention_mask)
            nxt = int(log_probs[0].argmax())
            if trace is not None:
                trace.append((step + 1, nxt, float(log_probs[0][nxt])))
            generated.append(nxt)
            if nxt == self.eos_token_id:
                break
        return generated

    def beam_search(self, memory, attention_mask, max_length: int = None,
                    num_beams: int = None) -> list:
        """束搜索：把当前所有 beam 作为一个 batch 一次喂给 decoder（顺便验证动态 batch）。"""
        max_length = int(max_length or self.max_length)
        num_beams = int(num_beams or self.num_beams)
        if num_beams <= 1:
            return self.greedy_decode(memory, attention_mask, max_length)

        beams = [([self.decoder_start_token_id], 0.0)]
        finished = []
        for _ in range(max_length - 1):
            if not beams:
                break
            longest = max(len(ids) for ids, _ in beams)
            cur = np.full((len(beams), longest), self.pad_token_id, dtype=np.int64)
            for row, (ids, _) in enumerate(beams):
                cur[row, :len(ids)] = ids
            mem = np.repeat(memory, len(beams), axis=0)
            mask = np.repeat(attention_mask, len(beams), axis=0)
            log_probs = self._decoder_step(cur, mem, mask)

            candidates = []
            for row, (ids, score) in enumerate(beams):
                top_ids = np.argsort(-log_probs[row])[:num_beams]
                for tid in top_ids:
                    candidates.append((ids + [int(tid)], score + float(log_probs[row][tid])))
            candidates.sort(key=lambda item: item[1], reverse=True)

            beams = []
            for ids, score in candidates:
                if ids[-1] == self.eos_token_id:
                    finished.append((ids, score))
                elif len(beams) < num_beams:
                    beams.append((ids, score))
                if len(beams) >= num_beams and (not self.early_stopping
                                                or len(finished) >= num_beams):
                    break
            if self.early_stopping and len(finished) >= num_beams:
                break

        # 长度归一化：score / len^length_penalty，避免打分裂偏向短句
        pool = finished or beams
        if not pool:
            return [self.decoder_start_token_id]
        pool.sort(key=lambda item: item[1] / max(1, len(item[0]) - 1) ** self.length_penalty,
                  reverse=True)
        return pool[0][0]

    def generate(self, texts, max_length: int = None, num_beams: int = None) -> list:
        """一步到位：英文句子（str 或 list）-> 中文译文列表。"""
        if isinstance(texts, str):
            texts = [texts]
        outputs = []
        for text in texts:
            memory, attention_mask, _ = self.encode([text])
            if (num_beams if num_beams is not None else self.num_beams) > 1:
                ids = self.beam_search(memory, attention_mask, max_length, num_beams)
            else:
                ids = self.greedy_decode(memory, attention_mask, max_length)
            outputs.append(self.decode_text(ids))
        return outputs


# ===========================================================================
# 2) 单句翻译（函数签名与 translate_main.py 保持一致，便于对照阅读）
# ===========================================================================
def translate(src, model: Seq2SeqOnnxModel, chn_tokenizer, num_beams: int = None) -> str:
    """翻译单条已编码的句子；src 可以是 token id 的 list 或 torch tensor（一维/二维均可）。"""
    if hasattr(src, "tolist"):  # 兼容 torch.LongTensor / np.ndarray
        src = src.tolist()
    if src and isinstance(src[0], (list, tuple)):  # (1, S) -> S
        src = list(src[0])

    memory, attention_mask = model.encode_ids([src])
    beams = int(num_beams if num_beams is not None else model.num_beams)
    if beams > 1:
        generated = model.beam_search(memory, attention_mask, model.max_length, beams)
    else:
        generated = model.greedy_decode(memory, attention_mask, model.max_length)
    return chn_tokenizer.decode_ids(model._clean_ids(generated))


def one_sentence_translate(sent, model, en_tokenizer, chn_tokenizer, BOS, EOS,
                           num_beams: int = None) -> str:
    """翻译单句英文：加 BOS/EOS -> 编码 -> ONNX 解码 -> 中文分词器还原文本。"""
    src_ids = [BOS] + en_tokenizer.EncodeAsIds(sent) + [EOS]
    return translate(src_ids, model, chn_tokenizer, num_beams=num_beams)


# ===========================================================================
# HuggingFace Hub：下载（snapshot_download）+ 真实性校验（下载验证）
# ===========================================================================
def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def looks_like_repo_id(value: str) -> bool:
    """判断是否形如 user/repo 的 HF 仓库 id（本地存在同名目录时按本地路径处理）。"""
    if os.path.isdir(value):
        return False
    parts = value.strip("/").split("/")
    return len(parts) == 2 and all(parts) and not value.startswith((".", "/", "~"))


def download_from_hub(repo_id: str, revision: str = None, local_dir: str = None,
                      token: str = None, verbose: bool = True):
    """从 HuggingFace Hub 拉取仓库，返回 (本地目录, 仓库信息 dict)。"""
    if not _module_available("huggingface_hub"):
        raise ImportError("需要 huggingface_hub：pip install huggingface_hub")

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi(token=token).model_info(repo_id, revision=revision, files_metadata=True, token=token)
    local_dir = snapshot_download(repo_id, revision=revision, local_dir=local_dir, token=token)
    safe_tensors = getattr(info, "safetensors", None)
    hub_info = {
        "repo_id": info.id,
        "revision": info.sha,
        "private": bool(getattr(info, "private", False)),
        "last_modified": str(getattr(info, "last_modified", "")),
        "num_files": len(info.siblings),
        "file_sizes": {s.rfilename: int(s.size or 0) for s in info.siblings},
        "safetensors": ({"total": int(safe_tensors.total),
                         "parameters": {k: int(v) for k, v in (safe_tensors.parameters or {}).items()}}
                        if safe_tensors is not None else None),
        "local_dir": os.path.abspath(local_dir),
    }
    if verbose:
        LOGGER.info("已从 Hub 拉取 %s@%s -> %s（%d 个文件）", info.id, (info.sha or "")[:12],
                    hub_info["local_dir"], hub_info["num_files"])
    return hub_info["local_dir"], hub_info


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: str, chunk: int = 1 << 20) -> str:
    """Git blob 的 sha1（即 Hub 上非 LFS 文件的 blob_id），用于逐字节校验小文件。"""
    digest = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files_against_hub(local_dir: str, repo_id: str, revision: str = None,
                             token: str = None) -> dict:
    """把本地（刚下载的）文件与 Hub 元数据逐文件比对哈希，确认传输没有损坏。

    * 大文件（safetensors / onnx）走 LFS：比对 `lfs.sha256`
    * 小文件（json / py / spm / vocab）：比对 git blob sha1（Hub 的 `blob_id`）
    """
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(repo_id, revision=revision, files_metadata=True, token=token)
    files, all_match = [], True
    for sibling in info.siblings:
        name = sibling.rfilename
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path):
            files.append({"file": name, "status": "local-missing", "match": False})
            all_match = False
            continue
        if sibling.lfs is not None:
            algo, local_hash, remote_hash = "lfs.sha256", _sha256_file(path), sibling.lfs.sha256
        else:
            algo, local_hash, remote_hash = "git-blob-sha1", _git_blob_sha1(path), sibling.blob_id
        match = bool(remote_hash is None or local_hash == remote_hash)
        all_match = all_match and match
        files.append({"file": name, "bytes": os.path.getsize(path), "algo": algo,
                      "remote": remote_hash, "local": local_hash, "match": match})
    return {"repo_id": info.id, "revision": info.sha, "all_match": bool(all_match), "files": files}


def _report_sentences(args) -> list:
    """下载验证用的测试句：优先 --text，否则取 dev.json 前 (--from-dev 或 3) 句。"""
    if args.text:
        return list(args.text)
    count = max(1, int(args.from_dev or 3))
    path = os.path.join(SCRIPT_DIR, config.dev_data_path.lstrip("./"))
    if not os.path.isfile(path):
        return ["The cat is sleeping on the sofa."]
    with open(path, encoding="utf-8") as fp:
        return [row[0] for row in json.load(fp)[:count]]


def verify_hub_download(repo_id: str, sentences, max_len: int = None, beam_size: int = None,
                        revision: str = None, download_dir: str = None, token: str = None,
                        ckpt: str = None, atol: float = 1e-3, with_transformers: bool = True,
                        coreml: bool = False) -> dict:
    """下载验证（真实性复现）：Hub 拉取 -> 逐文件哈希 -> ONNX 推理 -> HF 推理 -> 本地 PyTorch 对照。

    返回的 dict 全部是 JSON 友好类型，可直接写进 export_meta.json。
    """
    max_len = int(max_len or config.max_len)
    beam_size = int(beam_size or config.beam_size)
    report = {"repo_id": repo_id, "max_len": max_len, "beam_size": beam_size, "sentences": []}

    # 1) 下载（未指定 download_dir 时落在 HF 缓存里）
    local_dir, hub_info = download_from_hub(repo_id, revision=revision,
                                            local_dir=download_dir, token=token)
    report["hub"] = hub_info

    # 2) 逐文件哈希：确认从 Hub 拉下来的字节与 Hub 元数据完全一致
    files_report = verify_files_against_hub(local_dir, repo_id, revision=revision, token=token)
    report["files"] = files_report
    LOGGER.info("[下载验证] 文件哈希: %d/%d 一致（revision=%s）",
                sum(1 for f in files_report["files"] if f.get("match")), len(files_report["files"]),
                (files_report["revision"] or "")[:12])

    # 3) ONNX：就用下载下来的文件跑
    onnx_model = Seq2SeqOnnxModel.from_pretrained(local_dir, coreml=coreml)

    # 4) 可选：从 Hub id 直接加载 transformers 模型（验证 trust_remote_code 链路）
    hf_tokenizer = hf_model = None
    if with_transformers and _module_available("transformers"):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True,
                                                         revision=revision, token=token)
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, trust_remote_code=True,
                                                             revision=revision, token=token)
            hf_model.eval()
            report["tokenizer_class"] = type(hf_tokenizer).__name__
            report["model_class"] = type(hf_model).__name__
            report["num_parameters"] = int(sum(p.numel() for p in hf_model.parameters()))
            LOGGER.info("[下载验证] transformers 侧: %s / %s（%.1fM 参数）",
                        report["tokenizer_class"], report["model_class"],
                        report["num_parameters"] / 1e6)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[下载验证] transformers 侧加载失败，跳过：%s: %s",
                           type(exc).__name__, str(exc)[:200])
            hf_tokenizer = hf_model = None

    # 5) 可选：本工程 PyTorch 原模型（.pth）作为第三路对照
    torch_model = load_reference_model(ckpt) if (ckpt and os.path.isfile(ckpt)) else None

    all_matched = bool(files_report["all_match"])
    for text in sentences:
        item = {"sentence": text}
        memory, attention_mask, _ = onnx_model.encode([text])
        onnx_greedy = onnx_model.greedy_decode(memory, attention_mask, max_len)
        onnx_beam = onnx_model.beam_search(memory, attention_mask, max_len, beam_size)
        item["onnx_greedy_text"] = onnx_model.decode_text(onnx_greedy)
        item["onnx_beam_text"] = onnx_model.decode_text(onnx_beam)

        if hf_model is not None:
            import torch

            inputs = hf_tokenizer([text], return_tensors="pt")
            manual_ids = [[onnx_model.bos_token_id] + onnx_model.source_sp.EncodeAsIds(text)
                          + [onnx_model.eos_token_id]]
            with torch.no_grad():
                hub_greedy = hf_model.generate(**inputs, max_length=max_len, num_beams=1,
                                               do_sample=False)[0].tolist()
                hub_beam = hf_model.generate(**inputs, max_length=max_len,
                                             num_beams=beam_size)[0].tolist()
                hf_logits = hf_model(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"],
                                     decoder_input_ids=torch.tensor([onnx_greedy])).logits.numpy()
            ort_logits = onnx_model.decoder.run(None, {
                "input_ids": np.array([onnx_greedy], dtype=np.int64),
                "encoder_hidden_states": memory,
                "encoder_attention_mask": attention_mask})[0]
            eos = onnx_model.eos_token_id
            item["tokenizer_ids_match"] = inputs["input_ids"].tolist() == manual_ids
            item["hub_greedy_text"] = hf_tokenizer.batch_decode([hub_greedy],
                                                                skip_special_tokens=True)[0]
            item["hub_beam_text"] = hf_tokenizer.batch_decode([hub_beam], skip_special_tokens=True)[0]
            item["hub_vs_onnx_token_match"] = cut_at_eos(hub_greedy, eos) == cut_at_eos(onnx_greedy, eos)
            item["hub_vs_onnx_beam_match"] = cut_at_eos(hub_beam, eos) == cut_at_eos(onnx_beam, eos)
            item["max_abs_logits_diff"] = float(np.abs(hf_logits - ort_logits).max())
            item["logits_allclose"] = bool(np.allclose(hf_logits, ort_logits, atol=atol, rtol=atol))
            all_matched = all_matched and bool(item["hub_vs_onnx_token_match"])

        if torch_model is not None:
            torch_ids = torch_reference_greedy(ckpt, onnx_model.source_sp, text, max_len,
                                               onnx_model.pad_token_id, onnx_model.bos_token_id,
                                               onnx_model.eos_token_id, model=torch_model)
            item["torch_greedy_text"] = onnx_model.decode_text(torch_ids)
            item["onnx_vs_torch_token_match"] = (
                    cut_at_eos(torch_ids, onnx_model.eos_token_id)
                    == cut_at_eos(onnx_greedy, onnx_model.eos_token_id))
            all_matched = all_matched and bool(item["onnx_vs_torch_token_match"])

        LOGGER.info("[下载验证] %s", text[:60])
        LOGGER.info("[下载验证]   Hub HF  : %s", item.get("hub_greedy_text", "（未加载 transformers）"))
        LOGGER.info("[下载验证]   Hub ONNX: %s", item["onnx_greedy_text"])
        if torch_model is not None:
            LOGGER.info("[下载验证]   本地原始: %s", item["torch_greedy_text"])
        report["sentences"].append(item)

    report["all_matched"] = bool(all_matched)
    return report


def print_hub_report(report: dict) -> None:
    """把下载验证报告打印成人能看的报告。"""
    hub = report["hub"]
    files = report["files"]["files"]
    ok = sum(1 for f in files if f.get("match"))
    print("\n" + "=" * 100)
    print(f"仓库        : {hub['repo_id']}  (private={hub['private']}, revision={hub['revision'][:12]})")
    print(f"拉取目录    : {hub['local_dir']}   文件数: {hub['num_files']}")
    if hub.get("safetensors"):
        print(f"safetensors : {hub['safetensors']}")
    print(f"文件哈希    : {ok}/{len(files)} 与 Hub 元数据一致"
          f"（大文件 lfs.sha256，小文件 git blob sha1）" + ("" if ok == len(files) else "  ⚠ 有不一致"))
    if report.get("model_class"):
        print(f"transformers: {report.get('tokenizer_class')} / {report['model_class']} "
              f"({report.get('num_parameters', 0) / 1e6:.1f}M 参数)")
    for idx, item in enumerate(report["sentences"], 1):
        print("-" * 100)
        print(f"[{idx}] EN            : {item['sentence'][:84]}")
        print(f"    Hub HF        : {item.get('hub_greedy_text', '（未加载 transformers）')}")
        print(f"    Hub ONNX      : {item['onnx_greedy_text']}")
        if "torch_greedy_text" in item:
            print(f"    本地 PyTorch  : {item['torch_greedy_text']}")
        flags = [f"HF==ONNX {item.get('hub_vs_onnx_token_match')}"]
        if "onnx_vs_torch_token_match" in item:
            flags.append(f"ONNX==本地PyTorch {item['onnx_vs_torch_token_match']}")
        if "max_abs_logits_diff" in item:
            flags.append(f"logits max_abs_diff {item['max_abs_logits_diff']:.3e}")
        print("    判定          : " + " | ".join(flags))
        print(f"    束搜索({report['beam_size']})    : {item['onnx_beam_text']}")
    print("=" * 100)
    print("下载验证结论:", "PASS ✅ 从 Hub 拉取的仓库可加载、可推理，且与本地 PyTorch 逐 token 一致"
    if report["all_matched"] else "FAIL ❌ 见上面明细")


# ===========================================================================
# 3) PyTorch 对照（可选，只为确认 ONNX 与训练框架逐 token 一致）
# ===========================================================================
def resolve_ckpt_path(path: str) -> str:
    """相对路径优先按当前工作目录解析，再按脚本所在目录解析。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(SCRIPT_DIR, path)
    return candidate if os.path.exists(candidate) else path


def load_reference_model(ckpt_path: str):
    """加载本工程 PyTorch 原模型（.pth）作为对照基准；只加载一次，可被多句复用。"""
    import torch
    from model.tf_model import make_model

    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.to("cpu")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v
                           for k, v in state_dict.items()})
    model.eval()
    return model


def torch_reference_greedy(ckpt_path, source_sp, text, max_len, padding_idx, bos_idx, eos_idx,
                           model=None):
    """PyTorch 贪心解码（对照基准）。传入 model 可复用已加载的权重，避免反复读 .pth。"""
    import torch

    if model is None:
        model = load_reference_model(ckpt_path)

    with torch.no_grad():
        src = torch.LongTensor([[bos_idx] + source_sp.EncodeAsIds(text) + [eos_idx]])
        src_mask = (src != padding_idx).unsqueeze(-2)
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), bos_idx, dtype=torch.long)
        for _ in range(max_len - 1):
            length = ys.size(1)
            tgt_mask = (ys != padding_idx).unsqueeze(-2) & torch.tril(
                torch.ones(length, length, dtype=torch.bool))
            out = model.decode(memory, src_mask, ys, tgt_mask)
            nxt = int(model.generator(out[:, -1]).argmax(dim=-1).item())
            ys = torch.cat([ys, torch.full((1, 1), nxt, dtype=torch.long)], dim=1)
            if nxt == eos_idx:
                break
    return ys[0].tolist()


def cut_at_eos(ids, eos_idx) -> list:
    return list(ids[: ids.index(eos_idx) + 1]) if eos_idx in ids else list(ids)


# ===========================================================================
# 4) 交互式循环翻译（默认入口，风格与 translate_main.py 一致）
# ===========================================================================
def translate_example(args, model: Seq2SeqOnnxModel = None):
    """单句翻译示例：模型/分词器只加载一次，然后循环读用户输入（q! 退出）。"""
    if model is None:
        model = Seq2SeqOnnxModel.from_pretrained(args.onnx_dir, coreml=args.coreml)

    # 模型与分词器已预先加载，循环里不再重复从磁盘读取
    en_tokenizer = model.source_sp
    chn_tokenizer = model.target_sp
    BOS, EOS = model.bos_token_id, model.eos_token_id
    beam_size = int(args.beam_size or model.num_beams)
    max_len = int(args.max_len or model.max_length)

    print("\nONNX 翻译已就绪（onnxruntime + encoder_model.onnx / decoder_model.onnx）")
    print(f"解码方式: 贪心" + (f" + {beam_size}-beam 束搜索" if beam_size > 1 else "")
          + f"，最长 {max_len} 个 token\n")

    # ========== 循环翻译 ==========
    while True:  # 使用循环，让用户可以反复输入句子
        try:
            sent = input("请输入英文句子进行翻译（输入 q! 退出）：")
        except (EOFError, KeyboardInterrupt):  # 支持 Ctrl-D / Ctrl-C 退出
            print("\n已退出翻译程序。")
            break

        # 判断是否退出
        if sent.strip() == "q!":
            print("已退出翻译程序。")
            break

        # 跳过空输入
        if not sent.strip():
            print("输入为空，请重新输入。")
            continue

        t0 = time.time()
        # 走的是 translate()/one_sentence_translate() 这套和 translate_main.py 同名的接口
        translation = one_sentence_translate(sent, model, en_tokenizer, chn_tokenizer,
                                             BOS, EOS, num_beams=1)  # 贪心
        print("翻译结果：", translation)

        if beam_size > 1:
            beam_translation = one_sentence_translate(sent, model, en_tokenizer, chn_tokenizer,
                                                      BOS, EOS, num_beams=beam_size)
            print(f"束搜索（{beam_size}-beam）：", beam_translation)
        print(f"（ONNX 解码耗时 {time.time() - t0:.2f}s）\n")
    return 0


# ===========================================================================
# 5) 批量/命令行模式
# ===========================================================================
def load_demo_pairs(args):
    if args.text:
        return [(t, "") for t in args.text]
    path = os.path.join(SCRIPT_DIR, config.dev_data_path.lstrip("./"))
    if not os.path.isfile(path):
        LOGGER.warning("找不到 %s，改用内置示例句", path)
        return [("I love you.", "我爱你。")][: max(1, args.from_dev)]
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return [(row[0], row[1]) for row in data[: max(1, args.from_dev)]]


def run_batch(args, model: Seq2SeqOnnxModel) -> int:
    """非交互模式：--text / --from-dev，输出翻译结果（可选与 PyTorch 逐 token 对比）。"""
    beam_size = int(args.beam_size or model.num_beams)
    max_len = int(args.max_len or model.max_length)
    pairs = load_demo_pairs(args)

    all_matched = True
    total_cost = 0.0
    for idx, (en_sent, zh_ref) in enumerate(pairs, 1):
        memory, attention_mask, input_ids = model.encode([en_sent])

        t0 = time.time()
        trace = [] if args.show_details else None
        greedy_ids = model.greedy_decode(memory, attention_mask, max_len, trace=trace)
        greedy_text = model.decode_text(greedy_ids)
        beam_text = None
        if args.decode in ("beam", "both") and beam_size > 1:
            beam_text = model.decode_text(
                model.beam_search(memory, attention_mask, max_len, beam_size))
        cost = time.time() - t0
        total_cost += cost

        torch_ids = torch_text = None
        matched = None
        if args.compare:
            ckpt_path = args.ckpt if os.path.exists(args.ckpt) else os.path.join(SCRIPT_DIR, args.ckpt)
            torch_ids = torch_reference_greedy(ckpt_path, model.source_sp, en_sent, max_len,
                                               model.pad_token_id, model.bos_token_id,
                                               model.eos_token_id)
            torch_text = model.decode_text(torch_ids)
            matched = cut_at_eos(greedy_ids, model.eos_token_id) == \
                      cut_at_eos(torch_ids, model.eos_token_id)
            all_matched = all_matched and matched

        print("\n" + "=" * 100)
        print(f"[{idx}] EN              : {en_sent}")
        print(f"    encoder input_ids  : shape={tuple(input_ids.shape)} "
              f"(前 10 个 id: {input_ids[0][:10].tolist()})")
        print(f"    encoder memory     : shape={tuple(memory.shape)}")
        if zh_ref:
            print(f"    参考译文          : {zh_ref}")
        print(f"    ONNX 贪心         : {greedy_text}")
        if beam_text is not None:
            print(f"    ONNX {beam_size}-beam{' ' * 8}: {beam_text}")
        if args.compare:
            print(f"    PyTorch 贪心      : {torch_text}")
            print(f"    token 级一致      : {matched}  (ONNX {len(greedy_ids)} tokens, "
                  f"PyTorch {len(torch_ids)} tokens)")
        print(f"    ONNX 解码耗时     : {cost:.2f}s ({len(greedy_ids)} 步自回归)")
        if trace:
            print("    贪心逐步明细      : " + ", ".join(
                f"step{step}:token{tid}(logp={lp:.2f})" for step, tid, lp in trace))

    print("\n" + "=" * 100)
    if args.compare:
        print("对比结论: " + ("ONNX 与 PyTorch 逐 token 完全一致 ✅" if all_matched
                              else "存在不一致 ❌"))
    import onnxruntime as ort
    print(f"共 {len(pairs)} 句，ONNX 总耗时 {total_cost:.2f}s；"
          f"运行环境 onnxruntime {ort.__version__}，providers={model.encoder.get_providers()}")
    return 0 if all_matched else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="translate_onnx_hf.py",
        description="onnxruntime 加载 HF 风格 ONNX 目录（encoder_model.onnx / decoder_model.onnx）做英译中",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_dir = os.path.join(os.path.dirname(os.path.abspath(
        os.path.join(SCRIPT_DIR, config.translate_model_path))), "onnx_hf")
    parser.add_argument("--onnx-dir", default=default_dir, help="ONNX 目录（含 config.json 等）")
    parser.add_argument("--repo-id", default=None,
                        help="直接从 HuggingFace Hub 拉取，如 chou-lucas/transformer-en-zh-base（覆盖 --onnx-dir）")
    parser.add_argument("--revision", default=None, help="仓库 revision（分支 / tag / commit），默认 main")
    parser.add_argument("--download-dir", default=None, help="从 Hub 拉取到哪个目录（默认用 HF 缓存）")
    parser.add_argument("--hub-token", default=os.environ.get("HF_TOKEN"), help="HF token，默认读环境变量 HF_TOKEN")
    parser.add_argument("--verify-hub", action="store_true",
                        help="下载验证：拉取后做「逐文件哈希 + HF/ONNX/本地 PyTorch 逐 token 对照」并打印报告")
    parser.add_argument("--no-hub-transformers", dest="hub_transformers", action="store_false",
                        help="下载验证时不加载 transformers 模型，只做 ONNX 侧校验")
    parser.add_argument("--text", action="append", default=None,
                        help="要翻译的英文句子，可重复指定（指定后进入非交互模式）")
    parser.add_argument("--from-dev", type=int, default=None,
                        help="取 dev.json 前 N 句（指定后进入非交互模式）")
    parser.add_argument("--decode", choices=["greedy", "beam", "both"], default="both",
                        help="非交互模式下输出哪种解码结果")
    parser.add_argument("--max-len", type=int, default=None,
                        help="解码最大长度，默认读 generation_config.json 的 max_length")
    parser.add_argument("--beam-size", type=int, default=None,
                        help="beam size，默认读 generation_config.json 的 num_beams")
    parser.add_argument("--ckpt", default=config.translate_model_path, help="--compare 用的 PyTorch 权重")
    parser.add_argument("--no-compare", dest="compare", action="store_false",
                        help="跳过与 PyTorch 原模型的逐 token 对比（跳过则不需要 torch）")
    parser.add_argument("--coreml", action="store_true", help="使用 CoreML EP（更快，但可能有数值差异）")
    parser.add_argument("--show-details", action="store_true", help="打印贪心解码每一步的 token/log_prob")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- 直接从 HuggingFace Hub 使用 / 做下载验证（--repo-id）----
    if args.repo_id:
        if args.verify_hub:
            report = verify_hub_download(
                args.repo_id, _report_sentences(args),
                max_len=int(args.max_len or config.max_len),
                beam_size=int(args.beam_size or config.beam_size),
                revision=args.revision, download_dir=args.download_dir, token=args.hub_token,
                ckpt=resolve_ckpt_path(args.ckpt) if args.compare else None,
                with_transformers=args.hub_transformers, coreml=args.coreml)
            print_hub_report(report)
            return 0 if report["all_matched"] else 1

        local_dir, hub_info = download_from_hub(args.repo_id, revision=args.revision,
                                                local_dir=args.download_dir, token=args.hub_token)
        print(f"仓库 {hub_info['repo_id']}@{hub_info['revision'][:12]} -> {local_dir}")
        args.onnx_dir = local_dir
        if args.text is None and args.from_dev is None:
            args.from_dev = 3

    if not os.path.isdir(args.onnx_dir):
        raise FileNotFoundError(f"ONNX 目录不存在: {args.onnx_dir}，请先执行 python export_onnx_hf.py")

    model = Seq2SeqOnnxModel.from_pretrained(args.onnx_dir, coreml=args.coreml)
    print(model)

    # 没有 --text / --from-dev 时进入交互式循环，体验与 translate_main.py 一致
    if args.text is None and args.from_dev is None:
        return translate_example(args, model=model)
    return run_batch(args, model)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
