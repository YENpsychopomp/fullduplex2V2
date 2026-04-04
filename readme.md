# full-duplex2

全雙工語音串流示範專案，提供「前端錄音 -> 後端轉送 ASR -> VAD 斷句 -> 前端即時字幕」完整流程。

## 專案概述

本專案聚焦在語音輸入鏈路，目標是讓前端可以穩定接收兩種字幕事件：

- `partial`：逐字更新的即時辨識結果
- `final`：一次完整斷句的最終結果

核心流程如下：

1. 前端透過 WebSocket 持續送出 PCM bytes 或經過Base64編碼的音訊。
2. 後端重採樣 (24kHz -> 16kHz) 並轉送 ASR。
3. 後端透過 VAD 偵測停頓並觸發 ASR 結算。
4. 後端回傳字幕事件給前端顯示。

## 架構

- 前端：錄音、上傳音訊、呈現字幕
- 後端：FastAPI WebSocket，端點為 `/ws`
- ASR：遠端即時辨識服務 (Qwen 系列)
- VAD：Silero VAD + Smart Turn

相關文件：

- [VAD運作原理及參數調整.md](VAD運作原理及參數調整.md)
- [sequenceDiagram.mmd](sequenceDiagram.mmd)
- [vad.mmd](vad.mmd)

## 環境需求

- Python 3.10+
- 可用的 ASR gateway
- 瀏覽器麥克風權限

## 後端設定與啟動

### 1. 設定環境變數

[backend/main.py](backend/main.py) 會讀取：

- `WS_ASR_URL`
- `WS_ASR_API_KEY`
- `WS_ASR_MODEL_NAME`

建議在專案根目錄建立 `.env`：

```env
WS_ASR_URL=wss://your-asr-gateway/realtime
WS_ASR_API_KEY=your_key
WS_ASR_MODEL_NAME=qwen3-asr-1.7b
```

### 2. 啟動服務

```bash
cd backend
python main.py
```

預設網址：`http://127.0.0.1:7985`

## 前端串接步驟

最小串接流程如下：

1. 連線 `ws://${location.host}/ws`
2. `onopen` 送 `request.session`
3. 收到 `response.session` 後送 `request.set_system_prompt`
4. 每 100ms 上傳一包 PCM `ArrayBuffer`
5. 依 `response.asr_text.status` 更新 UI

可直接參考 [frontend/js/all.js](frontend/js/all.js) 的 `connect_ws`、`startRecording`。

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
  "audio_data": "<Base64 encoded PCM data>"
}
```

### Client -> Server (Binary)

- 音訊型態：PCM bytes (`ArrayBuffer`)、Base64 編碼的 PCM 字串
- 建議規格：24kHz / 16-bit / mono
- 後端會重採樣為 16kHz 後送 ASR
- 如果前端選擇以json格式送出音訊，請確保base64編碼後的字串不會太大，以免造成WebSocket傳輸問題

### Server -> Client (JSON)

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

- `partial`：即時更新，通常用來更新右側字幕
- `final`：完整斷句，通常用來新增一則對話泡泡

## 前端最小範例

```javascript
const ws = new WebSocket(`ws://${location.host}/ws`);
let sessionId = null;

ws.onopen = () => {
  ws.send(JSON.stringify({ type: "request.session" }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === "response.session") {
    sessionId = msg.session_id;
    ws.send(JSON.stringify({
      type: "request.set_system_prompt",
      session_id: sessionId,
      system_prompt: "這裡放你想要的 System Prompt"
    }));
    return;
  }

  if (msg.type === "response.asr_text") {
    if (msg.status === "partial") {
      document.querySelector("#transcript").textContent = msg.text;
    }

    if (msg.status === "final") {
      const li = document.createElement("li");
      li.textContent = msg.text;
      document.querySelector("#messages").appendChild(li);
      document.querySelector("#transcript").textContent = "";
    }
  }
};

function sendPcmChunk(arrayBuffer) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(arrayBuffer);
  }
}
```

## 故障排查

1. 連不上 `/ws`

- 確認後端服務在 `127.0.0.1:7985`
- 確認前端不是連到錯誤 host 或 port

2. 有 `partial`，但沒有 `final`

- 確認後端 log 有 `VAD 偵測到停頓`
- 測試時請刻意停頓約 1 到 2 秒

3. 一直出現重複字幕

- 先檢查是否重複綁定 `onmessage`
- 前端可增加一層 final 去重 (以最後一句文字比對)

4. 文字內容有簡繁差異或標點不同

- 屬 ASR 輸出特性，非串接錯誤
5. 如果語音為英文，但 ASR 輸出中文這不是系統錯誤，而是 ASR 模型的語言預測結果。請聯絡開發團隊調整 ASR 模型的語言設定。

## 開發現況

- 已完成：即時串流、VAD 斷句、ASR 結果回推、session 音訊存檔
- 待擴充：LLM 回覆串流、TTS 回傳音訊