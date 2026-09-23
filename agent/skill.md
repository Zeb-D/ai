---
type: skill
name: baidu_search
description: "用 curl 访问百度网页搜索，抓取并解析搜索结果的标题/链接/摘要（纯命令行实现）"
tags: [搜索, 百度, 网络, curl]
---

# 技能：百度网页搜索（baidu_search）

> 本文件是「技能文档」示例：Agent 通过 `cat agent/skill.md` 读到它，然后照着下面的命令一步步执行。
> 技能文档不写代码逻辑，只写「用哪些命令、按什么顺序、看什么输出、失败了怎么办」。

## 何时使用

- 用户说「百度一下 XXX」「网上查一下 XXX」「搜一下最新的 XXX」；
- 任务需要**外部实时信息**（新闻、天气、股价、最新版本号、某个概念的最新说法等）；
- 需要给出可追溯的参考链接时。

## 前置准备（只做一次）

1. 确认 curl 可用：`curl --version | head -1`
2. 记录当前时间，写进最终结论（避免把旧信息当最新）：`date '+%Y-%m-%d %H:%M'`
3. **中文关键词必须 URL 编码**（用单引号包住 python 代码，避免引号冲突）：
   `python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' '关键词'`
   例如输出 `deepseek%20%E6%9C%80%E6%96%B0%E6%A8%A1%E5%9E%8B`，把它拼进下一步 URL 的 `wd=` 参数。

## 步骤预算（很重要）

- **全程最多 3 次搜索**（含换关键词）；每条结果页只解析一次，不要重复抓同一页。
- 只要标题/摘要里已经出现能回答问题的信息（版本号、日期、公司名、价格等），**立刻输出 `完成:`**，
  并注明依据是「搜索结果标题/摘要」；不要在追求「最权威」上反复换关键词。
- 命令总条数有上限（默认 10 条），超出会被打断；能合并的命令尽量用 `;` 合成一条
  （例如 `curl ... -o /tmp/f.html; wc -c /tmp/f.html`）。

## 执行步骤

### 第 1 步：把解析脚本落盘（一次写好，之后复用）

百度的 HTML 里既有 JSON 又有标签，用一条命令写完整个解析脚本最稳（`<<'PY'` 加引号可避免 `$`、反斜杠被 shell 解释）：

```bash
cat > /tmp/baidu_parse.py <<'PY'
import re, sys
from html import unescape
from html.parser import HTMLParser
from urllib.parse import unquote

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/baidu_search.html"
raw = open(path, encoding="utf-8", errors="ignore").read()


class _Text(HTMLParser):
    """用标准库 HTMLParser 取纯文本：它懂属性引号里的 '>'，比正则剥标签可靠"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(text):
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    parser = _Text()
    parser.feed(text)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def real_url(block):
    """百度把真实地址 URL 编码后放在 data-click 的 JSON 里"""
    match = re.search(r"(https?%3A%2F%2F[^\"&\\ ]+)", unescape(block))
    return unquote(match.group(1)) if match else ""


print(f"# 页面 {len(raw)} 字符")
for i, match in enumerate(re.finditer(r"<h3[^>]*>(.*?)</h3>", raw, re.S), 1):
    if i > 10:
        break
    block = match.group(1)          # 每个 <h3> 内部 = 一条结果的标题 + 跳转链接
    title = plain(block)
    if not title:
        continue
    jump = re.search(r'href="(https?://[^"]+)"', block)
    print(f"{i}. {title}")
    if jump:
        print(f"   链接: {jump.group(1)}")
    if real := real_url(block):
        print(f"   真实地址: {real}")
    # 摘要取标题之后的文本（卡片类结果的摘要由 JS 渲染，取不到是正常的）
    print(f"   摘要: {plain(raw[match.end(): match.end() + 2500])[:140]}")
PY
```

> 卡片类结果（百度百科 / 官网卡片）的摘要**经常为空**，这是正常的：
> 不要为此重抓页面，用「标题 + 真实地址」作为来源即可；确实要看细节才走第 5 步（最多 1 条链接）。

### 第 2 步：抓取搜索页

必须伪装 UA，中文关键词用上一步的编码结果（`rn=10` 表示一页 10 条）：

```bash
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'; curl -sL --compressed -A "$UA" -H 'Referer: https://www.baidu.com/' -H 'Accept-Language: zh-CN,zh;q=0.9' 'https://www.baidu.com/s?wd=<编码后关键词>&rn=10' -o /tmp/baidu_search.html; wc -c /tmp/baidu_search.html
```

### 第 3 步：确认抓到了内容

`wc -c` 只有几百字节、或页面里出现「百度安全验证 / 网络不给力」→ 跳到「被反爬拦住怎么办」，**不要重复执行第 2 步**：

```bash
grep -c -e 'result' -e '百度安全验证' /tmp/baidu_search.html
```

### 第 4 步：解析标题 / 链接 / 摘要

```bash
python3 /tmp/baidu_parse.py /tmp/baidu_search.html
```

### 第 5 步（可选）：抓某条结果的正文细节

挑第 4 步里最相关的一条链接，再抓一次并去标签（`-L` 会自动跟随百度跳转）：

```bash
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'; curl -sL --compressed -A "$UA" '链接' -o /tmp/baidu_page.html; python3 -c 'import html,re;t=open("/tmp/baidu_page.html",encoding="utf-8",errors="ignore").read();t=re.sub(r"(?is)<(script|style).*?</\1>"," ",t);print(re.sub(r"\s+"," ",html.unescape(re.sub("<[^>]+>"," ",t)))[:1500])'
```

### 第 6 步（可选）：换关键词或翻页

只改 `wd=` 的值；要第二页就在 URL 末尾追加 `&pn=10`，然后重复第 2~4 步。

## 被反爬拦住怎么办

按顺序尝试，任一步拿到正常页面就停止：

1. **换移动端页面**（结构更简单、拦截更少）：
   `curl -sL --compressed -A "$UA" 'https://m.baidu.com/s?word=<编码后关键词>' -o /tmp/baidu_search.html`
2. **补齐浏览器头**（Referer / Accept-Language 已写在第 2 步里，仍被拦时再加 `-H 'Cookie: BAIDUID=...'` 或换 UA）
3. **换备用搜索源**（结果结构类似，拿到内容即可）：
   `curl -sL --compressed -A "$UA" 'https://cn.bing.com/search?q=<编码后关键词>' -o /tmp/bing_search.html`
   再用 `grep -oE '<h2>.*?</h2>' /tmp/bing_search.html | sed -E 's/<[^>]+>//g' | head -10` 取标题。
4. 全都失败 → 不要继续重试，直接进入 `完成:` 如实说明。

## shell 引号避坑（重要）

- `python3 -c` 的代码**用单引号包裹**（代码内部一律用双引号）；写了双引号包裹 + 内部双引号的命令会直接语法报错（例如匹配 `href="..."` 时）。
- 代码超过一行、或含 `$`、反斜杠、引号嵌套时，**先用 heredoc 写成 `/tmp/xxx.py` 再执行**，不要硬塞进一行。
- 变量（如 `UA`）在同一个 `命令:` 里定义即可；跨命令不会保留。

## 输出要求

- `完成:` 里按「结论（1~3 句）+ 来源（3~5 条：标题 + 链接）」给出，并注明数据来自百度搜索、抓取时间（取自前置准备第 2 条的 `date`）。
- **绝对不要编造搜索结果或链接**：只总结命令真实抓到的标题/摘要/正文。
- 摘要为空不算失败：给出「标题 + 链接」同样是有效来源；不确定的结论要写清依据来自标题而非正文。
- 全部方案都被拦截时：说明「未能获取搜索结果」，并给出人工可点击的搜索链接 `https://www.baidu.com/s?wd=<编码后关键词>`。
- 时间敏感的信息，必须在结论里写出「信息抓取时间」。
