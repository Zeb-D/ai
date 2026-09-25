
# 对应的文档：https://zhuanlan.zhihu.com/p/10223831998
# 这篇文章介绍了一种名为HyDE（Hypothetical Document Embeddings）的零样本（zero-shot）密集检索方法。HyDE的核心思想是通过一个假设文档（hypothetical document）来桥接查询和文档之间的相关性，从而在没有相关性标签的情况下进行有效的文档检索。以下是文章的原理、思想和流程图的总结：
#
# 原理和思想
# 1、零样本学习挑战：在没有相关性标签的情况下，创建有效的零样本密集检索系统是非常困难的。HyDE通过生成假设文档来解决这一问题。
# HyDE模型：HyDE模型包括两个主要步骤：
#
# 生成假设文档：使用指令跟随型语言模型（如InstructGPT）根据查询生成一个假设文档。这个文档捕捉了相关性模式，但可能包含虚假细节，并且是虚构的。
# 无监督对比学习编码：使用无监督对比学习编码器（如Contriever）将假设文档编码成嵌入向量。这个向量在语料库嵌入空间中识别一个邻域，基于向量相似性检索相似的真实文档。
# 相关性编码：HyDE不直接建模查询-文档相似度分数，而是将检索任务分解为两个自然语言理解和生成任务

"""
HyDE (Hypothetical Document Embeddings) 检索演示
修复：Ollama 模型切换间隙导致 /api/chat 502
"""

import os
import sys
import logging
import time
from typing import List

import numpy as np
import requests

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:
    np.set_printoptions(legacy="1.21")
except TypeError:
    np.set_printoptions(legacy=False)

from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ==================== 嵌入模型（原生 requests）====================
class RobustOllamaEmbeddings(Embeddings):
    def __init__(
        self,
        base_url: str,
        model: str,
        batch_size: int = 8,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._warm_up()

    def _call(self, input_data, timeout: int = 120):
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": input_data,
                "keep_alive": "10m",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _warm_up(self):
        logger.info("🔥 正在预热 Ollama 嵌入模型...")
        for attempt in range(self.max_retries):
            try:
                data = self._call("warmup", timeout=120)
                dim = len(data["embeddings"][0])
                logger.info(f"✅ 嵌入模型预热完成，维度: {dim}")
                return
            except Exception as e:
                logger.warning(f"预热失败 (尝试 {attempt + 1}): {e}")
                time.sleep(self.retry_delay)
        raise RuntimeError("嵌入模型预热失败")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        total = len(texts)
        all_embeddings: List[List[float]] = []
        use_batch = True

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            for attempt in range(self.max_retries):
                try:
                    if use_batch:
                        data = self._call(batch)
                        embs = data["embeddings"]
                    else:
                        embs = [self._call(t)["embeddings"][0] for t in batch]
                    all_embeddings.extend(embs)
                    break
                except Exception as e:
                    logger.warning(f"批次 {i//self.batch_size + 1} 失败: {str(e)[:100]}")
                    if use_batch and attempt == 0:
                        logger.info("🔄 降级为逐个嵌入...")
                        use_batch = False
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
            if i + self.batch_size < total:
                time.sleep(0.3)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call(text)["embeddings"][0]


# ==================== LLM 预热（防止模型切换 502）====================
def warm_up_llm(base_url: str, model: str, max_retries: int = 3):
    """在 embed 之后显式预热 LLM，确保模型已驻留内存"""
    logger.info("🔥 正在预热 LLM 模型（等待 Ollama 切换模型）...")
    time.sleep(2)  # 给 Ollama 卸载 embed 模型的时间
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "keep_alive": "10m",
                },
                timeout=120,
            )
            resp.raise_for_status()
            logger.info("✅ LLM 预热完成")
            return
        except Exception as e:
            logger.warning(f"LLM 预热失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(3)
    raise RuntimeError("LLM 预热失败")


# ==================== 工具函数 ====================
def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "（未检索到相关文档）"
    return "\n\n".join(
        f"[文档 {i+1}]\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    )


def load_documents(data_path: str, glob_pattern: str) -> List[Document]:
    abs_path = os.path.abspath(data_path)
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"数据目录不存在: {abs_path}")
    loader = DirectoryLoader(
        data_path,
        glob=glob_pattern,
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    logger.info(f"📄 成功加载 {len(docs)} 页文档")
    return docs


def get_vectorstore(chunks: List[Document], embedding_model, persist_dir: str) -> Chroma:
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        logger.info(f"📂 加载已有向量库: {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
    logger.info("🔨 正在创建向量库...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    logger.info(f"💾 向量库已保存到: {persist_dir}")
    return vectorstore


def build_hyde_chain(llm):
    hyde_prompt = ChatPromptTemplate.from_template(
        """请针对以下问题，撰写一段详细、准确的回答（约 500~1000 字），用于辅助后续文档检索。

问题：{question}

回答段落："""
    )
    return (
        {"question": RunnablePassthrough()}
        | hyde_prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(llm):
    rag_prompt = ChatPromptTemplate.from_template(
        """你是一个专业的研究助手。请严格基于以下参考资料回答问题。
如果参考资料不足以回答问题，请明确说明"根据现有资料无法确定"，不要编造信息。

参考资料：
{context}

问题：{question}

请给出详细、准确的回答："""
    )
    return rag_prompt | llm | StrOutputParser()


# ==================== 主程序 ====================
def main():
    OLLAMA_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "lrs33/bce-embedding-base_v1")
    LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.1")
    DATA_PATH       = os.getenv("DATA_PATH", "/Users/lucas/Documents/A学习/")
    GLOB_PATTERN    = os.getenv("GLOB_PATTERN", "AI-Agents-in-Depth-zh-CN.pdf")
    VECTORSTORE_DIR = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    QUESTION        = os.getenv("QUESTION", "agent是什么？")

    # 1. 初始化嵌入模型
    logger.info("🚀 初始化模型...")
    embeddings = RobustOllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBED_MODEL,
        batch_size=8,
    )

    # 🔥 关键修复：embed 完成后，等 Ollama 切换模型，再预热 LLM
    warm_up_llm(OLLAMA_URL, LLM_MODEL)

    # 2. 初始化 LLM（timeout=120 覆盖模型加载时间）
    llm = ChatOllama(
        base_url=OLLAMA_URL,
        model=LLM_MODEL,
        temperature=0,
        keep_alive="5m",
        timeout=120,  # 关键：给模型加载留足时间
    )

    # 3. 加载文档
    documents = load_documents(DATA_PATH, GLOB_PATTERN)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"✂️  文档分块完成，共 {len(chunks)} 块")

    # 4. 向量存储
    vectorstore = get_vectorstore(chunks, embeddings, VECTORSTORE_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 5. HyDE 检索
    logger.info(f"❓ 问题: {QUESTION}")

    standard_docs = retriever.invoke(QUESTION)
    logger.info(f"🔍 标准检索返回 {len(standard_docs)} 条结果")

    hyde_chain = build_hyde_chain(llm)
    hypothetical_doc = hyde_chain.invoke(QUESTION)

    logger.info("=" * 60)
    logger.info("📝 生成的假设文档（前 300 字）:")
    logger.info(hypothetical_doc[:300] + "..." if len(hypothetical_doc) > 300 else hypothetical_doc)
    logger.info("=" * 60)

    hyde_docs = retriever.invoke(hypothetical_doc)
    logger.info(f"🎯 HyDE 检索返回 {len(hyde_docs)} 条结果")

    # 6. 最终回答
    rag_chain = build_rag_chain(llm)

    hyde_answer = rag_chain.invoke({
        "context": format_docs(hyde_docs),
        "question": QUESTION,
    })

    standard_answer = rag_chain.invoke({
        "context": format_docs(standard_docs),
        "question": QUESTION,
    })

    print("\n" + "=" * 60)
    print("🤖 【HyDE 检索回答】")
    print("=" * 60)
    print(hyde_answer)

    print("\n" + "=" * 60)
    print("🤖 【标准检索回答（对比）】")
    print("=" * 60)
    print(standard_answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ 程序运行失败: {e}", exc_info=True)
        sys.exit(1)