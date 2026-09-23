/* 最小聊天前端：SSE 流式接收 + 可选的语音朗读（浏览器原生 speechSynthesis，免 Key、免依赖） */
(() => {
  'use strict';

  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const autoReadEl = document.getElementById('autoRead');

  // 会话 id 只用于让服务端区分不同浏览器（刷新不丢上下文）
  const sessionId = (() => {
    let id = localStorage.getItem('chat_session');
    if (!id) {
      id = 's' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
      localStorage.setItem('chat_session', id);
    }
    return id;
  })();

  let streaming = false;
  let controller = null;

  autoReadEl.checked = localStorage.getItem('chat_read_aloud') === '1';
  autoReadEl.addEventListener('change', () => {
    localStorage.setItem('chat_read_aloud', autoReadEl.checked ? '1' : '0');
    unlockSpeech(); // 用户点击过，正好用来解锁 iOS 的语音播放
    if (!autoReadEl.checked) stopSpeaking();
  });

  // ---------------------------------------------------------------- 渲染
  function addBubble(role) {
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollDown();
    return bubble;
  }

  function scrollDown() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setBusy(busy) {
    streaming = busy;
    sendBtn.textContent = busy ? '停止' : '发送';
    sendBtn.classList.toggle('stop', busy);
  }

  // ---------------------------------------------------------------- 发送
  async function send() {
    const text = inputEl.value.trim();
    if (!text || streaming) return;

    addBubble('user').textContent = text;
    inputEl.value = '';
    autoGrow();

    const bubble = addBubble('assistant');
    bubble.classList.add('typing');
    let answer = '';
    let finalAnswer = ''; // 服务端在 done 事件里给出的「完成:」总结，朗读时用它，避免念出命令过程

    setBusy(true);
    controller = new AbortController();
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status} ${await response.text()}`);
      if (!response.body) throw new Error('浏览器不支持流式读取');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let index;
        while ((index = buffer.indexOf('\n\n')) !== -1) {
          const event = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          for (const line of event.split('\n')) {
            if (!line.startsWith('data:')) continue;
            let payload;
            try {
              payload = JSON.parse(line.slice(5).trim());
            } catch {
              continue;
            }
            if (payload.delta) {
              answer += payload.delta;
              bubble.classList.remove('typing');
              bubble.textContent = answer;
              scrollDown();
            } else if (payload.done) {
              if (payload.answer) finalAnswer = payload.answer;
            } else if (payload.error) {
              answer += `\n⚠️ ${payload.error}`;
              bubble.classList.remove('typing');
              bubble.textContent = answer;
            }
          }
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        answer += `${answer ? '\n' : ''}⚠️ 出错了：${error.message}`;
        bubble.classList.remove('typing');
        bubble.textContent = answer;
      }
    } finally {
      setBusy(false);
      controller = null;
      bubble.classList.remove('typing');
      if (!answer) bubble.parentElement.remove();
      else if (autoReadEl.checked) speak(finalAnswer || answer);
    }
  }

  // ---------------------------------------------------------------- 语音朗读
  const canSpeak = 'speechSynthesis' in window;
  let voice = null;

  function pickVoice() {
    const voices = speechSynthesis.getVoices();
    voice =
      voices.find((item) => /zh[-_]?CN|zh[-_]?Hans|Chinese|中文/i.test(`${item.lang} ${item.name}`)) ||
      voices.find((item) => item.lang.startsWith('zh')) ||
      null;
  }

  if (canSpeak) {
    pickVoice();
    speechSynthesis.onvoiceschanged = pickVoice;
  }

  // iOS 需要一次用户手势才允许出声：用一段静音语音解锁
  let unlocked = false;
  function unlockSpeech() {
    if (!canSpeak || unlocked) return;
    try {
      const warmup = new SpeechSynthesisUtterance(' ');
      warmup.volume = 0;
      speechSynthesis.speak(warmup);
      speechSynthesis.resume();
      unlocked = true;
    } catch {
      /* 忽略：不支持就算了 */
    }
  }

  function stopSpeaking() {
    if (canSpeak) speechSynthesis.cancel();
  }

  function splitForSpeech(text, limit = 110) {
    const chunks = [];
    let buffer = '';
    for (const char of text) {
      buffer += char;
      if (buffer.length >= limit && /[。！？!?；;\n]/.test(char)) {
        chunks.push(buffer.trim());
        buffer = '';
      }
    }
    if (buffer.trim()) chunks.push(buffer.trim());
    return chunks.filter(Boolean);
  }

  function speak(text) {
    if (!canSpeak) return;
    unlockSpeech();
    stopSpeaking();
    const clean = text
      .replace(/```[\s\S]*?```/g, ' 代码略 ')
      .replace(/https?:\/\/\S+/g, ' 链接 ')
      .replace(/[*_#>`~]/g, '')
      .trim();
    // 分句朗读：整段太长时部分手机浏览器会截断
    for (const chunk of splitForSpeech(clean)) {
      const utterance = new SpeechSynthesisUtterance(chunk);
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang;
      } else {
        utterance.lang = 'zh-CN';
      }
      speechSynthesis.speak(utterance);
    }
  }

  // ---------------------------------------------------------------- 交互
  function autoGrow() {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, window.innerHeight * 0.4)}px`;
  }

  sendBtn.addEventListener('click', () => {
    unlockSpeech();
    if (streaming) {
      controller?.abort();
      setBusy(false);
      return;
    }
    send();
  });

  inputEl.addEventListener('input', autoGrow);
  inputEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      unlockSpeech();
      send();
    }
  });
})();
