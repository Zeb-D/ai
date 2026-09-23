# 🐣 agent —— 入门 Agent Demo（DeepSeek）

一个**零第三方依赖**的小 demo，两种入口、共用同一份 LLM 请求代码：

| 入口 | 命令 | 说明 |
| --- | --- | --- |
| 🖥️ 控制台版 | `python3 agent/main.py` | 用「命令: / 完成:」文本协议驱动 Agent，模型输出命令 → `os.popen` 本机执行 → 结果回填继续推理 |
| 📱 网页版 | `python3 agent/web.py` | 手机浏览器输入 URL 即可聊天（SSE 打字机效果），可勾选「朗读回复」用语音念出来 |

```text
agent/
├── agent.md        # 【示例 1】Agent 定义：人设 + 输出协议（front-matter + 正文）
├── skill.md        # 【示例 2】Skill 定义：百度网页查询（curl 抓取 + 解析 + 反爬兜底）
├── main.py         # 🖥️ 控制台版入口：while 交互循环 + 命令循环 + 协议解析 + 预算强制收尾
├── web.py          # 📱 网页版入口：标准库 HTTP + SSE 流式聊天（静态文件在同目录）
├── static/         # 网页版前端：index.html / style.css / app.js
├── llm.py          # ⭐ DeepSeek 客户端（唯一实现）：chat() / stream() / complete()
├── config.py       # ⭐ 共用配置：路径、.env、预算、网页端口与人设
├── tools.py        # 控制台版专用：os.popen 执行命令 + 危险命令过滤（网页版不执行任何命令）
├── skills.py       # 控制台版专用：技能发现（markdown front-matter → 技能清单）
├── .env.example    # 复制为 .env 并填 DEEPSEEK_API_KEY
└── README.md
```

> 完整的开发过程、设计取舍与踩坑记录见 [`agent入门开发.md`](agent入门开发.md:1)。

## 快速开始

```bash
cd /Users/lucas/PycharmProjects/ai
cp agent/.env.example agent/.env      # 填入 DEEPSEEK_API_KEY=sk-xxx（无需 pip install）
```

**控制台版：**

```bash
python3 agent/main.py                                   # 交互模式
python3 agent/main.py -p "百度一下 deepseek 最新模型版本，给出 3 条来源链接"
python3 agent/main.py -p "agent 目录里有几个 python 文件？各多少行？"
```

交互命令：`/help`、`/skills`、`/history`、`/clear`、`/exit`。

**网页版（手机聊天）：**

```bash
python3 agent/web.py                 # 启动后会打印手机可访问的地址
python3 agent/web.py --port 8899     # 换端口
```

启动输出示例：

```text
模型：deepseek-chat
本机访问：http://127.0.0.1:8000
手机访问（手机与电脑连同一个 Wi-Fi，然后在浏览器输入）：
   👉 http://192.168.1.4:8000
```

手机浏览器打开那个 `http://192.168.1.4:8000` 就能聊天：

- 打字发送（回车发送 / Shift+Enter 换行），回答以**流式**方式逐字出现；
- 勾选顶栏「**朗读回复**」，之后每条回答都会用系统语音念出来（浏览器原生 `speechSynthesis`，不需要任何语音 API Key）；
- 不想听了再点掉开关或直接停止。

## 控制台版的「命令协议」

协议写在 [`agent.md`](agent/agent.md:14) 里，模型必须二选一：

```text
你的目标是完成用户的任务，你必须选择下面的其中一种格式进行回复:

1.如果你认为需要执行命令，则输出'命令:XXX'，XXX 为命令本身，不要用任何的格式，不要解释

2，如果你认为不需要执行命令，则输出'完成:XXX'，XXX 为你的总结信息
```

| 模型输出 | Agent 行为 |
| --- | --- |
| `命令:cat agent/skill.md` | 用 [`os.popen`](agent/tools.py:46) 执行，结果回填为观察结果，**继续**让模型判断 |
| `完成:……` | 本轮结束，展示总结 |

解析在 [`parse_reply()`](agent/main.py:65)，循环在 [`run_turn()`](agent/main.py:104)：

```text
外层：while True: 读用户输入 → 追加到 messages
内层：for step in 1..MAX_STEPS:
        模型回复 → 解析协议 → 是「命令:」就执行并回填 → 是「完成:」就结束
      预算用尽 → 自动提示模型立刻收尾（避免死循环）
```

**技能文档**（[`skill.md`](agent/skill.md:1)）只写命令步骤，摘要常驻提示、正文由模型自己 `cat` 读进来
（渐进式披露）；想加技能，在 `agent/skills/xxx.md` 放一个带 `type: skill` front-matter 的 markdown 即可。

## 网页版实现要点

- **纯标准库**：`http.server.ThreadingHTTPServer` 起服务，没有 Flask/FastAPI 等依赖；
- **SSE 流式**：[`POST /api/chat`](agent/web.py:156) 逐块 `data: {"delta": "..."}` 推给前端，前端用
  `fetch` + `ReadableStream` 边收边渲染（[`app.js`](agent/static/app.js:1)），所以有打字机效果；
- **多轮上下文**：按浏览器生成的 `session_id` 在内存里保存最近 20 条消息（[`get_history()`](agent/web.py:52)）；
- **同一个 LLM 请求代码**：网页版直接调 [`llm.stream()`](agent/llm.py:110)，控制台版调 [`llm.chat()`](agent/llm.py:126)，
  底层都是同一个 `POST /chat/completions`；
- **网页版不执行命令**：只有聊天，没有 shell 能力，比控制台版更安全，适合随手给家人用；
- **手机适配**：`viewport-fit=cover` + `env(safe-area-inset-*)` 适配刘海屏，输入框 16px 字号避免 iOS 自动缩放，深色模式自适应。

### 语音朗读说明

- 用的是浏览器自带的 TTS（`window.speechSynthesis`），**iOS Safari / Android Chrome 都支持**，无需联网第三方服务；
- iOS 有「必须由用户手势触发」的限制：勾选开关或点发送时已经算手势，所以能正常出声；
- 长回答会被自动分句后再播报（部分手机浏览器对单段长文本会截断）；
- 朗读前会去掉代码块、链接和 Markdown 符号，避免念出一堆符号。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `未找到 DEEPSEEK_API_KEY` | `cp agent/.env.example agent/.env` 并填 Key，或 `export DEEPSEEK_API_KEY=sk-xxx` |
| `HTTP 401 / 402 / 429` | Key 无效 / 余额不足 / 触发限流 |
| 手机打不开网页 | ① 手机与电脑要在**同一个 Wi-Fi**；② 电脑防火墙放行 Python（macOS：系统设置 → 网络 → 防火墙）；③ 用启动日志里打印的 `http://192.168.x.x:端口`，不要用 `127.0.0.1` |
| 网页没有打字机效果 | 浏览器太旧不支持 `ReadableStream`；换新版 Chrome/Safari/Edge |
| 勾了朗读但没声音 | 手机静音开关 / 媒体音量；iOS 需先点一次页面上的按钮（本项目已自动处理）；部分安卓机需在系统设置里装中文 TTS 语音包 |
| 回答带一堆 `**`、`#` | 网页只按纯文本显示；可把 `AGENT_WEB_SYSTEM_PROMPT` 调成「不要用 Markdown」的提示词 |
| 控制台版命令不按协议输出 | 把 `DEEPSEEK_TEMPERATURE` 调到 `0`；或检查 `agent.md` 的协议段落是否被改坏 |
| 命令数用尽没结论 | 已内置「强制收尾」，会自动让模型用已有信息总结；不够就调大 `AGENT_MAX_STEPS` |

## 修改指南（不用改代码）

| 想改什么 | 改哪里 |
| --- | --- |
| 控制台版人设 / 输出协议 / 模型 | [`agent.md`](agent/agent.md:1) 的 front-matter 与正文 |
| 新增/修改技能 | `agent/skills/*.md`（带 `type: skill` front-matter） |
| 网页版人设 | `agent/.env` 里的 `AGENT_WEB_SYSTEM_PROMPT`（或 [`config.py`](agent/config.py:66) 默认值） |
| 端口 / 监听地址 | `AGENT_WEB_HOST`、`AGENT_WEB_PORT`，或 `--host/--port` |
| 模型 / 温度 / 超时 | `DEEPSEEK_MODEL`、`DEEPSEEK_TEMPERATURE`、`DEEPSEEK_TIMEOUT`（两个版本共用） |
| 网页界面 | `agent/static/index.html`、`style.css`、`app.js` |

## 安全与边界

- 控制台版会在本机执行模型给的 shell 命令，[`tools.py`](agent/tools.py:22) 只有一层粗黑名单（`sudo`、`rm -rf /`、`mkfs`、fork bomb…），
  且命令固定在工作区根目录执行；`os.popen` 没有超时和沙箱，**只适合本地学习**，生产请换 `subprocess` + 超时 + 容器/白名单；
- 网页版**不执行任何命令**，只做 LLM 对话，风险主要是「同一局域网内谁都能访问」，需要的话用 `--host 127.0.0.1` 只监听本机；
- 会话存在内存里，服务重启即清空；`agent/.env` 里是明文 Key，请勿提交到仓库。
