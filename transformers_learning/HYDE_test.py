import numpy as np  # 确保numpy正确导入

# 解决NumPy版本冲突警告（可选）
np.set_printoptions(legacy='1.25')

from langchain_community.vectorstores import Chroma  # 修正导入路径

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
if __name__ == "__main__":
    import os.path
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # 修正ChatOllama和OllamaEmbeddings的导入路径
    from langchain_community.chat_models.ollama import ChatOllama
    from langchain_community.embeddings.ollama import OllamaEmbeddings

    # 使用更新后的包（推荐）
    # 需先安装：pip install langchain-ollama
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    # 初始化模型和嵌入
    base_embeddings = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="lrs33/bce-embedding-base_v1"
    )
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.1",
        temperature=0
    )

    # 文档加载（请设置正确的数据路径）
    data_path = "./data"  # 修正为空路径问题
    loader = DirectoryLoader(
        data_path,
        glob="2024世界经济展望报告.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(documents)

    # 向量存储初始化
    vectorstore_path = "data/vectorstore"
    if not os.path.exists(vectorstore_path):
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=base_embeddings,
            persist_directory=vectorstore_path
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=base_embeddings
        )

    # 检索器配置
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    # question = "What's the mainly content of Global Financial Stability Report?"
    question = "2024全球发展中国家合作进展发生了什么？"
    print("retriever: ", retriever.invoke(question))  # 修正参数格式

    # HyDE提示词模板
    from langchain.prompts import ChatPromptTemplate

    hyde_prompt = ChatPromptTemplate.from_template("""
        写一个段落回答下面的问题，字数限制1000字：
        问题：{question}
        段落：
    """)

    # 生成假设文档的链
    generate_doc_chain = (
            {'question': RunnablePassthrough()}
            | hyde_prompt
            | llm
            | StrOutputParser()
    )

    # 生成假设文档
    print("生成的假设文档: ", generate_doc_chain.invoke(question))

    # 检索链
    retrieval_chain = generate_doc_chain | retriever
    retrieved_docs = retrieval_chain.invoke(question)
    print("检索到的文档: ", retrieved_docs)

    # 最终回答链
    template = """根据提供的上下文回答问题：

    {context}

    问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
            prompt
            | llm
            | StrOutputParser()
    )

    print("最终回答: ", final_rag_chain.invoke({
        "context": retrieved_docs,
        "question": question
    }))
