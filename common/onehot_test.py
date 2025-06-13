# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 构建数组
    print("onehot")
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder()
    vectors = encoder.fit_transform([["我"], ["要"], ["关注"], ["架构师那些事儿"], ["想想"], ["就"], ["开心"]])
    print("vectors:", vectors)

    print("CountVectorizer")
    from sklearn.feature_extraction.text import CountVectorizer
    import jieba

    # 构建语料库
    corpus = [
        "我要关注架构师那些事儿",
        "想想就开心",
        "我要看架构师那些事儿想想就开心"
    ]

    # 定义分词器函数
    def chinese_tokenizer(text):
        return jieba.cut(text)

    # 初始化词袋模型（指定分词器）
    vectorizer = CountVectorizer(tokenizer=chinese_tokenizer)

    # 训练并转换语料库
    X = vectorizer.fit_transform(corpus)

    # 输出词表和对应的词袋矩阵
    print("词表：", vectorizer.get_feature_names_out())
    print("词袋矩阵：\n", X.toarray())


    print("\nTfidfVectorizer")
    from sklearn.feature_extraction.text import TfidfVectorizer

    # 构建语料库
    corpus = [
        "我要关注架构师那些事儿",
        "想想就开心",
        "我要看架构师那些事儿想想就开心"
    ]
    # 初始化TF-IDF模型
    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
    # 训练并转换语料库
    X = vectorizer.fit_transform(corpus)
    # 输出词表和对应的TF-IDF矩阵
    print("词表：", vectorizer.get_feature_names_out())
    print("TF-IDF矩阵：\n", X.toarray())