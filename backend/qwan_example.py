# coding=utf-8
import os

# =====================================================================
# 【讀取本地模型設定】(必須在 import 其他套件前設定)
os.environ["HF_HOME"] = "./qwan_models_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
# =====================================================================

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from contextlib import asynccontextmanager

from qwen_asr import Qwen3ASRModel
import logging
import json

# 全域變數，用來存放模型
asr_model = None
ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
EXPECTED_SAMPLE_RATE = 16000
logger = logging.getLogger("uvicorn.error")

def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """簡單的重採樣 (24k -> 16k)"""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 啟動與關閉時的生命週期管理"""
    global asr_model
    print("正在載入 Qwen3-ASR 模型到 GPU... 請稍候...")
    asr_model = Qwen3ASRModel.LLM(
        model=ASR_MODEL_PATH,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_new_tokens=32,
    )
    print("模型載入完成！API 伺服器已啟動。")
    yield
    print("伺服器關閉，釋放資源。")

app = FastAPI(lifespan=lifespan, title="Qwen3-ASR Streaming API")

@app.websocket("/ws/asr")
async def asr_websocket(websocket: WebSocket):
    """處理即時音訊串流與控制指令的 WebSocket 端點"""
    await websocket.accept()
    logger.info("收到新的語音辨識連線！")
    
    # 1. 初始化狀態
    def reset_state():
        return asr_model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )

    state = reset_state()
    
    try:
        while True:
            # 使用 receive() 同時相容 bytes 與 text (JSON)
            message = await websocket.receive()
            
            # --- 處理二進位音訊 (PCM Bytes) ---
            if "bytes" in message:
                audio_bytes = message["bytes"]
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # 降頻處理
                wav16k = _resample_to_16k(audio_float32, EXPECTED_SAMPLE_RATE)
                
                # 推論
                asr_model.streaming_transcribe(wav16k, state)
                
                # 回傳中間結果
                await websocket.send_json({
                    "type": "response.streaming",
                    "text": state.text
                })

            # --- 處理 JSON 控制指令 (例如 finish_stream) ---
            elif "text" in message:
                data = json.loads(message["text"])
                
                if data.get("type") == "control.finish_stream":
                    reason = data.get("reason", "unknown")
                    logger.info(f"收到結算指令，原因: {reason}")
                    
                    # 執行結算流程
                    asr_model.finish_streaming_transcribe(state)
                    final_text = state.text
                    
                    # 回傳最終結果
                    await websocket.send_json({
                        "type": "response.final",
                        "text": final_text,
                        "reason": reason
                    })
                    
                    logger.info(f"結算完成: {final_text}")
                    
                    # 🌟 重要：結算後重置狀態，讓同一個連線可以開始下一句話
                    state = reset_state()

    except WebSocketDisconnect:
        logger.info("前端掛斷連線")
    except Exception as e:
        logger.error(f"ASR 迴圈發生錯誤: {e}")

if __name__ == "__main__":
    # 使用 0.0.0.0 讓外部 (Windows 主機) 可以連入
    uvicorn.run(app, host="0.0.0.0", port=8001)