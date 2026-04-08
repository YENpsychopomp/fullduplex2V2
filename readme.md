# full-duplex2 語音助理專案

全雙工語音串流示範專案，提供「前端錄音 -> 後端轉送 ASR -> VAD 斷句 -> 前端即時字幕 -> AI 語言模型思考 -> TTS 語音合成 -> 前端即時回覆及播放」的完整流程。

## 專案概述

本專案聚焦在語音與 AI 助理的串接，目標是讓前端可以用最簡單的方式接收即時的語音辨識與 AI 回覆。前端只需要處理 WebSocket 收到什麼訊息就顯示什麼畫面或播放什麼聲音。

目前後端會給前端三種主要的訊息：

1. **ASR (語音辨識) 字幕**：`response.asr_text`，這代表使用者講的話。
2. **Agent (AI) 回覆**：`response.agent_text`，這代表 AI 回答的話。
3. **Agent (AI) 語音**：`response.agent_audio`，這代表 AI 回答的聲音檔。

這兩種文字訊息都會有兩種狀態：
- `partial`：即時正在輸入的字（像打字機一樣，建議拿來更新畫面上的同一個對話泡泡或暫存區）。
- `final`：完整的一句話（代表講完了，建議拿來正式新增一個完整的對話泡泡到畫面上）。

核心運作流程：
1. 前端透過 WebSocket 持續送出錄音資料 (PCM 音訊)。
2. 後端收到後轉送給語音辨識分析。
3. 當後端發現「使用者停頓不講話了」(透過 VAD 技術)，就會告訴前端這句使用者講的話結束了 (`final`)。
4. 接著後端會自動把這句話丟給 AI 思考，並將 AI 的回覆一段一段透過 WebSocket (`response.agent_text`) 傳給前端。
5. AI 的文字回覆會自動送給 TTS (語音合成) 產生音訊檔，並透過 WebSocket (`response.agent_audio`) 傳給前端，前端依序播放。

## 架構

- 前端：只負責錄音、丟音軌給後端、取得文字時更新畫面及依序播放語音 (JS `Recorder` + 佇列機制)。
- 後端：FastAPI WebSocket，你的老家就是 `/ws`。
- ASR (語音辨識)：遠端即時辨識 (Qwen)。
- VAD (斷句系統)：我們用它去判斷使用者「什麼時候不講話了」，以此來觸發 AI 講話。
- Agent (AI)：內建天氣、路況查詢與自由聊天的語言模型 (LangChain + Azure OpenAI)。
- TTS (語音合成)：負責將 AI 代理回覆的文字串轉成音軌回傳給前端。

相關文件：

- [VAD運作原理及參數調整.md](VAD運作原理及參數調整.md)
- [sequenceDiagram.mmd](sequenceDiagram.mmd)
- [vad.mmd](vad.mmd)

## 環境需求

- Python 3.10+
- 可用的 ASR gateway
- 可用的 TTS 伺服器 (如 Fish Speech 等)
- 瀏覽器麥克風權限

## 後端設定與啟動

### 1. 安裝環境與依賴套件

建議使用虛擬環境：
```bash
pip install -r requirements.txt
```

若有需要本地快取與啟動 Qwen 模型，可執行 `python backend/download_qwen_asr_model.py` 預先下載。

### 2. 設定環境變數

[backend/main.py](backend/main.py) 會讀取：

- `WS_ASR_URL`
- `WS_ASR_API_KEY`
- `WS_ASR_MODEL_NAME`
- `TTS_URL`
- `TTS_API_KEY`

需要在專案根目錄建立 `.env`：

```env
AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
OPENAI_API_VERSION="your_api_version"
AZURE_OPENAI_API_KEY="your_azure_api_key"
AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
WS_ASR_URL=wss://your-asr-gateway/realtime
WS_ASR_API_KEY=your_key
WS_ASR_MODEL_NAME=qwen3-asr-1.7b
TTS_URL=http://your-tts-server/v1/tts
TTS_API_KEY=your_tts_key
```

### 3. 啟動服務

```bash
cd backend
python main.py
```

**預設網址：`http://127.0.0.1:7985`**
**線上體驗：`https://voice.54ucl.com/realtime/`**

## 前端串接步驟

你需要準備以下幾個動作：

1. 連線至 WebSocket：`ws://voice.54ucl.com/realtime/ws/`
2. WebSocket 開啟 (`onopen`) 後先發送： `{ type: "request.session" }`
3. 收到後端的 `response.session` 以後，就可以開始錄音。
4. 設定錄音器的頻率為 **16000Hz (16kHz), 16-bit, Mono(單聲道)**。
5. 設一個 `setInterval` (定時器)，每 100毫秒 (0.1秒) 將錄到的 PCM `ArrayBuffer` 發送 (send) 給 WebSocket。
6. 寫好 `onmessage` 專門接收 `response.asr_text` (使用者的話) 跟 `response.agent_text` (AI的話)，並用 JavaScript 更新畫面 (`innerText` 或 `innerHTML`)。

可以參考寫好的範例檔案：[frontend/js/all.js](frontend/js/all.js) 裡的 `connect_ws()` 以及 `startRecording()`。

## WebSocket 協定

### Client -> Server (JSON)

1. `request.ping`

```json
{ "type": "request.ping" }
```

2. `request.session`

```json
{ "type": "request.session" }
```

3. `request.set_system_prompt`

```json
{
  "type": "request.set_system_prompt",
  "session_id": "<session_id>",
  "system_prompt": "<Prompt text>"
}
```
4. `input_audio_buffer.append`
```json
{
  "type": "input_audio_buffer.append",
  "audio": "<Base64 encoded PCM data>"
}
```

### Client 發送 -> Server 接收 (Binary 聲音資料)

當開好 session 以後，只要一直發 `ws.send(PCM的ArrayBuffer)` 就可以了：

- 必須是：**16kHz / 16-bit / Mono(單聲道)** 的 PCM bytes (`ArrayBuffer`)
- *注意: 可以的話不要用 Base64 JSON，直接送二進位 `ArrayBuffer` 最穩 (錄音函式庫通常會提供 `.getNextData()` 等方法拿 `ArrayBuffer`)*。

### Server 回傳 -> Client 接收 (JSON 文字或 AI 訊息)

1. `response.ping`

```json
{ "type": "response.ping", "msg": "pong" }
```

2. `response.session`

```json
{
  "type": "response.session",
  "session_id": "<session_id>",
  "msg": "Session created with ID: <session_id>"
}
```

3. `response.asr_text`

```json
{
  "type": "response.asr_text",
  "text": "今天天氣如何？",
  "language": "",
  "status": "partial"
}
```

`status` 說明：

- `partial`：即時傳回來的碎片（打字機效果，你可以覆蓋畫面上某個指定的 `<span id="transcript">`）
- `final`：真正結束的一整段話（你可以清空 `<span id="transcript">`，並在畫面上新增對話框）

4. **`response.agent_text`** (AI 在說話，格式跟上方一模一樣)

```json
{
  "type": "response.agent_text",
  "text": "今天天氣很好",
  "status": "partial" // 或者 "final" 表示AI說完了
}
```

5. **`response.agent_audio`** (AI 在說話的語音檔案)

```json
{
  "type": "response.agent_audio",
  "audio": "<Base64 encoded WAV data>"
}
```

## 前端最小串接範例 (Copy & Paste)

只要看懂這段 JavaScript 結構，基本上你就會接了！

```javascript
/* 前端：建立 WebSocket 與處理畫面/播放語音 */
const ws = new WebSocket(`ws://voice.54ucl.com/realtime/ws/`);
let sessionId = null;

// 音訊播放佇列
let audioQueue = [];
let isPlayingAudio = false;

function playNextAudio() {
    if (audioQueue.length === 0) {
        isPlayingAudio = false;
        return;
    }
    isPlayingAudio = true;
    let audioSrc = audioQueue.shift();
    let audioObj = new Audio(audioSrc);
    audioObj.onended = () => { playNextAudio(); };
    audioObj.play().catch(e => console.error("音訊播放失敗: ", e));
}

ws.onopen = () => {
  // 1. 連線成功，告訴後端我要開房間 (session)
  ws.send(JSON.stringify({ type: "request.session" }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === "response.session") {
    sessionId = msg.session_id;

    // 2. 這時候就可以告訴錄音器開始丟 ArrayBuffer 過去了!
    // (需自己實作你的錄音工具並取得 pcm array buffer)
    // 範例： setInterval(() => ws.send( audioBuffer ), 100)
    
    // 你也可以設定這個人的 System Prompt (AI扮演誰)
    ws.send(JSON.stringify({
      type: "request.set_system_prompt",
      session_id: sessionId,
      system_prompt: "你是個超有禮貌的導遊助理。"
    }));
    return;
  }

  // 3. 處理「使用者」的語音變成文字
  if (msg.type === "response.asr_text") {
    if (msg.status === "partial") {
      // (打字機模式) 正在講，一直覆蓋同一個區塊文字就好
      document.querySelector("#user-live-text").textContent = msg.text;
    }

    if (msg.status === "final") {
      // (結束) 把最後這段話變成泡泡加進聊天記錄裡
      const li = document.createElement("li");
      li.textContent = "我: " + msg.text;
      document.querySelector("#messages").appendChild(li);
      // 清空暫存的打字機文字
      document.querySelector("#user-live-text").textContent = ""; 
    }
  }

  // 4. 處理「AI (Agent)」回答的文字
  if (msg.type === "response.agent_text") {
    if (msg.status === "partial") {
      document.querySelector("#ai-live-text").textContent = msg.text;
    }
    
    if (msg.status === "final") {
      const li = document.createElement("li");
      li.textContent = "AI: " + msg.text;
      document.querySelector("#messages").appendChild(li);
      document.querySelector("#ai-live-text").textContent = ""; 
    }
  }

  // 5. 處理「AI (Agent)」回答的語音
  if (msg.type === "response.agent_audio") {
    let audioSrc = "data:audio/wav;base64," + msg.audio;
    audioQueue.push(audioSrc);
    if (!isPlayingAudio) {
      playNextAudio();
    }
  }
};
```