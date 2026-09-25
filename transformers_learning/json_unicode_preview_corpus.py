import json
import os

path = "dataset/test.json"

print(os.getcwd())

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(data[0])

ch_path = './data/corpus.ch'
en_path = './data/corpus.en'


def corpus_language(files):
    ch_lines = []
    en_lines = []

    for file in files:
        # 加载 JSON 格式的语料文件
        corpus = json.load(open("./dataset" + "/" + file + ".json", "r", encoding="utf-8"))
        for item in corpus:
            en_lines.append(item[0] + '\n')
            ch_lines.append(item[1] + '\n')

    with open(en_path, "w", encoding="utf-8") as f:
        f.writelines(en_lines)

    with open(ch_path, "w", encoding="utf-8") as f:
        f.writelines(ch_lines)

    print("lines of Chinese: ", len(ch_lines))
    print("lines of English: ", len(en_lines))


def analyzer_corpus(ch_path, en_path):
    """
    分析双语料库文件的详细信息
    """
    if not os.path.exists(ch_path):
        print("ch_path does not exist")
        return
    if not os.path.exists(en_path):
        print("en_path does not exist")
        return

    # 获取文件大小
    ch_size = os.path.getsize(ch_path)
    en_size = os.path.getsize(en_path)

    # 读取文件内容并统计行数
    with open(ch_path, "r", encoding="utf-8") as f:
        ch_lines = f.readlines()
    with open(en_path, "r", encoding="utf-8") as f:
        en_lines = f.readlines()

    # 计算字符数
    ch_chars = sum([len(line.strip()) for line in ch_lines])
    en_chars = sum([len(line.strip()) for line in en_lines])

    # 打印统计信息
    print("语料信息" + "=" * 20)
    print(f"{ch_size} {ch_chars:,} {len(ch_lines)}")
    print(f"{en_size} {en_chars:,} {len(en_lines)}")


if __name__ == "__main__":
    # 对原始语料文件json 进行切割中英文
    # corpus_language(['train', 'dev', 'test'])

    analyzer_corpus(ch_path, en_path)
