# full-duplex2

這個專案提供一個「前端即時錄音 + 後端轉送到 ASR + VAD 斷句判斷 + 即時回傳文字」的全雙工語音串流示範。
前端透過 WebSocket 傳送 PCM Bytes，後端轉送到 ASR 服務並把辨識結果回推給前端，同時用雙層 VAD 進行動態端點偵測。

## 主要功能

- **全雙工串流**：前端音訊 (Bytes) -> 後端 -> ASR；ASR 回傳文字 -> 後端 -> 前端。
- **雙層 VAD**：Silero VAD + Smart Turn Detector，降低呼吸聲與短停頓造成的誤斷句。
- **Session 管理**：`request.session` 建立 session id，後端管理緩衝音訊與結束存檔。
- **自動存檔**：前端斷線後，後端會把收集到的 PCM 存在 `backend/recorder/`。

## 架構與流程

- **前端 (Web UI)**：擷取麥克風音訊 (24kHz / 16-bit / mono PCM)，透過 WebSocket 傳送。
- **後端 (FastAPI WebSocket `/ws`)**：接收音訊、重採樣 (24kHz -> 16kHz)、轉送 ASR、回推結果。
- **ASR 服務**：`ws://127.0.0.1:8001/ws/asr`，目前採用 Qwen3-ASR 系列模型。
- **VAD 模型**：核心在 [backend/vad.py](backend/vad.py) 與 [backend/vad_inference.py](backend/vad_inference.py)。

參考文件：
- [VAD運作原理及參數調整.md](VAD運作原理及參數調整.md)

## 快速啟動

1. 確認 ASR Server 已啟動於 `ws://127.0.0.1:8001/ws/asr`
2. 進入 `backend` 目錄並啟動後端
```bash
cd backend
python main.py
```
3. 開啟瀏覽器 `http://127.0.0.1:7985/`

## WebSocket 訊息規格

### 前端 -> 後端 (JSON)

- `request.ping`
```json
{ "type": "request.ping" }
```

- `request.session`
```json
{ "type": "request.session" }
```

- `request.set_system_prompt`
```json
{
	"type": "request.set_system_prompt",
	"session_id": "<session_id>",
	"system_prompt": "你是一個帶有幽默感的語言模型，請用中文回答問題。"
}
```

### 前端 -> 後端 (Bytes)

- **PCM 音訊串流**：直接送出 `ArrayBuffer`，格式為 24kHz / 16-bit / mono。

### 後端 -> 前端 (JSON)

- `response.ping`
```json
{ "type": "response.ping", "msg": "pong" }
```

- `response.session`
```json
{ "type": "response.session", "session_id": "<session_id>", "msg": "Session created" }
```

- `response.asr_text`
```json
{
	"type": "response.asr_text",
	"text": "辨識到的文字",
	"language": "zh",
	"status": "streaming"
}
```

## 後續開發

- 整合 LLM 回覆 (Azure OpenAI)
- 整合 TTS 回覆 (如 Fish Speech)