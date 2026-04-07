from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydub import AudioSegment

import os
import asyncio
import json
import logging
import uuid
import requests
import uvicorn
import base64
import websockets
import numpy as np
import httpx
import aiofiles

from sessions_manager import SessionManager, SessionData, AudioFormat
from vad import record_and_predict, ensure_model
from agent import StreamingVoiceAgent

load_dotenv()
app = FastAPI(title="full-duplex2")
logger = logging.getLogger("uvicorn.error")
session_manager = SessionManager()
voice_agent = StreamingVoiceAgent(logger)
isfinish = False
asr_generation = 0
asr_finalized_generation = 0
VAD_MODEL_PATH = ensure_model()
WS_ASR_URL = os.getenv("WS_ASR_URL")
WS_ASR_API_KEY = os.getenv("WS_ASR_API_KEY")
WS_ASR_MODEL_NAME = os.getenv("WS_ASR_MODEL_NAME")

class _HTTPOnlyStaticFiles:
    def __init__(self, static_app: StaticFiles):
        self._static_app = static_app

    async def __call__(self, scope, receive, send):
        """
        對 StaticFiles 進行封裝，以確保只處理 HTTP 請求。
        """
        if scope.get("type") != "http":
            if scope.get("type") == "websocket":
                await send({"type": "websocket.close", "code": 1000})
            return
        await self._static_app(scope, receive, send)

async def _send_finish_stream(asr_ws, reason="VAD_detected_pause"):
    """當 VAD 偵測到停頓或前端掛斷時，觸發 Qwen ASR 結算流程"""
    global isfinish, asr_generation
    if isfinish:
        return
    logger.info(f"觸發 ASR 結算，原因: {reason}")
    await asr_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    isfinish = True
    asr_generation += 1

async def tts(text: str, websocket: WebSocket):
    # 確保每次呼叫產生獨立的 UUID 與時間戳
    # output_path = f"backend/recorder/{uuid.uuid4()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
    url = os.getenv("TTS_URL")
    headers = {
        "Authorization": os.getenv("TTS_API_KEY"),
        "Content-Type": "application/json"
    }

    files = ["backend/xiaoxin_audio1.mp3", "backend/xiaoxin_audio2.mp3", "backend/xiaoxin_audio3.mp3", "backend/xiaoxin_audio4.mp3", "backend/xiaoxin_audio5.mp3", "backend/xiaoxin_audio6.mp3", "backend/xiaoxin_audio7.mp3"]
    base64_files = []
    
    # 初始化階段的磁碟讀取可保持同步，因僅在啟動時執行一次
    for f in files:
        with open(f, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
            base64_files.append(encoded_string)
            
    references = [
        {"audio": base64_files[0], "text": "奪回足球大作戰喔!"},
        {"audio": base64_files[1], "text": "失敗了"},
        {"audio": base64_files[2], "text": "不愧是你最嚮往的醫生角色完全融入了喔!"},
        {"audio": base64_files[3], "text": "這張臉跟風間一模一樣欸!"},
        {"audio": base64_files[4], "text": "好!"},
        {"audio": base64_files[5], "text": "爸爸在這裡喔!"},
        {"audio": base64_files[6], "text": "丟掉了汪!而且他還在看你喔"},
    ]

    data = {
        "model": "fish-speech-server",
        "text": text,
        "references": references,
        "format": "wav",
        "normalize": True,
        "speed": 1.0
    }
    
    logger.info(f"正在發送 TTS 請求...")
    try:
        # 使用 httpx 完成非同步 POST 防止阻塞 Event Loop
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=60.0)
            
        if response.status_code == 200:
            # 使用非同步 I/O 寫入檔案，避免大檔案阻塞 Event Loop
            logger.info(f"response_info: {response.status_code}, {response.headers}")
            # async with aiofiles.open(output_path, "wb") as f:
            #     await f.write(response.content)
            # logger.info(f"✅ 語音合成成功！已儲存為 {output_path}")
            
            # 轉換為 Base64 傳給前端
            audio_base64 = base64.b64encode(response.content).decode("utf-8")
            await websocket.send_text(json.dumps({
                "type": "response.agent_audio",
                "audio": audio_base64
            }))
        else:
            logger.error(f"❌ 合成失敗，狀態碼: {response.status_code}, 訊息: {response.text}")

    except httpx.RequestError as e:
        logger.error(f"💥 網路請求發生錯誤: {e}")
    except Exception as e:
        logger.error(f"💥 發生未知錯誤: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_session_id = None
    full_transcript = ""
    asr_ws = None
    try:
        # 🌟 建立與 ASR 伺服器的連線
        asr_ws = await websockets.connect(f"{WS_ASR_URL}?api_key={WS_ASR_API_KEY}")
        
        # 1) 接收連線訊息並更新 Session
        await asr_ws.recv()
        await asr_ws.send(json.dumps({"type": "session.update", "model": WS_ASR_MODEL_NAME}))
        logger.info("成功連線到 ASR 伺服器！")
        
        async def _process_agent(transcript):
            """將字串交給 Agent 並串流回傳"""
            if not current_session_id:
                return
            session_info = session_manager.get_session_info(current_session_id)
            if not session_info:
                return
                
            history = session_info.chat_history[:-1] # 不含當前這句，因為 ASR 當前句子由下面 user_text 帶入
            system_prompt = session_info.system_prompt
            
            full_reply = ""
            temp_reply = ""
            end_of_part = [".", "!", "?", "\n", "。", "！", "？", ",", "，"] # 以這些符號作為暫時回覆的切分依據
            async for chunk in voice_agent.stream_chat(transcript, history, system_prompt):
                full_reply += chunk
                temp_reply += chunk
                await websocket.send_text(json.dumps({
                    "type": "response.agent_text",
                    "text": full_reply,
                    "status": "partial"
                }))
                if any(temp_reply.endswith(p) for p in end_of_part):
                    await tts(temp_reply, websocket)
                    temp_reply = ""  # 切分後清空暫存回覆，等待下一段
            # 傳送最終 Agent 回覆
            await websocket.send_text(json.dumps({
                "type": "response.agent_text",
                "text": full_reply,
                "status": "final"
            }))
            
            # 使用 TTS 合成語音並透過 WebSocket 發送給前端
            await tts(temp_reply, websocket)
            temp_reply = ""
            # 存入 Session 歷史紀錄
            session_manager.save_agent_result(current_session_id, full_reply)

        # ==========================================
        # 任務 A：專門負責接收 ASR 回傳的文字，並轉發給前端
        # ==========================================
        async def receive_from_asr():
            nonlocal full_transcript
            global isfinish, asr_generation, asr_finalized_generation

            async def _flush_final(asr_result, force_finalize=False):
                """將目前累積字幕視情況結算成 final，避免卡到下一輪才觸發。"""
                nonlocal full_transcript
                global isfinish, asr_generation, asr_finalized_generation

                transcript = full_transcript.strip()
                if transcript:
                    is_saved = session_manager.save_asr_result(current_session_id, transcript)
                    if is_saved:
                        await websocket.send_text(json.dumps({
                            "type": "response.asr_text",
                            "text": transcript,
                            "language": asr_result.get("language", ""),
                            "status": "final"
                        }))
                        logger.info(f"ASR 最終結果: {transcript}")
                        
                        # -------- 觸發 Agent 處理 --------
                        asyncio.create_task(_process_agent(transcript))

                    full_transcript = ""

                if force_finalize and isfinish:
                    isfinish = False
                    asr_finalized_generation = asr_generation

            while True:
                asr_response_str = await asr_ws.recv()
                asr_result = json.loads(asr_response_str)
                logger.info(f"ASR 回傳原始訊息: {asr_result}")
                if asr_generation == asr_finalized_generation:
                    continue
                if "delta" in asr_result:
                    text_chunk = asr_result["delta"]
                    if text_chunk in ["language", " English", " Chinese"]:
                        continue

                    if text_chunk == "<asr_text>":
                        # 有些 ASR 會在段落邊界再吐一次 <asr_text>
                        if full_transcript.strip():
                            await _flush_final(asr_result, force_finalize=isfinish)
                        continue

                    if text_chunk == "":
                        # 遇到空 delta 視為一個段落邊界，若正在等待 commit 結算則立即 final
                        if isfinish and full_transcript.strip():
                            await _flush_final(asr_result, force_finalize=True)
                        continue

                    # 轉發 ASR 結果給前端 (讓前端顯示)
                    full_transcript += text_chunk
                    await websocket.send_text(json.dumps({
                        "type": "response.asr_text",
                        "text": full_transcript,
                        "language": asr_result.get("language", ""),
                        "status": "partial"
                    }))
                        
                if asr_result.get("type") in ["transcription.done", "response.done"]:
                    await _flush_final(asr_result, force_finalize=True)

        # ==========================================
        # 任務 C：獨立的 VAD 駐列與處理，防止卡住收發
        # ==========================================
        vad_queue = asyncio.Queue()
        async def process_vad():
            while True:
                audio_bytes = await vad_queue.get()
                # 將 VAD 處理丟入獨立執行緒，避免阻塞 WebSocket Event Loop
                is_pause = await asyncio.to_thread(record_and_predict, audio_bytes)
                if is_pause:
                    logger.info("VAD 偵測到停頓，通知 ASR 結算...")
                    await _send_finish_stream(asr_ws, "pause_detected")

        # ==========================================
        # 任務 B：專門負責接收前端訊息，並轉發給 ASR
        # ==========================================
        async def receive_from_frontend():
            nonlocal current_session_id # 允許修改外部變數
            while True:
                # 1. 接收訊息
                raw_msg = await websocket.receive()

                if raw_msg.get("type") == "websocket.disconnect":
                    logger.info("收到前端斷線訊號，準備結束通話...")
                    raise WebSocketDisconnect(code=raw_msg.get("code", 1000))
                
                # 2. 安全處理 JSON 訊息
                if "text" in raw_msg and raw_msg["text"]:
                    try:
                        data = json.loads(raw_msg["text"])
                    except json.JSONDecodeError:
                        logger.error("收到的不是有效的 JSON 格式")
                        continue
                    
                    if data.get("type") == "request.ping":
                        await websocket.send_text(json.dumps({"type": "response.ping", "msg": "pong"}))
                        
                    elif data.get("type") == "request.session":
                        sid = session_manager.create_session()
                        current_session_id = sid
                        await websocket.send_text(json.dumps({
                            "type": "response.session",
                            "session_id": sid,
                            "msg": f"Session created with ID: {sid}"
                        }))
                        
                    elif data.get("type") == "request.set_system_prompt":
                        sid = data.get("session_id")
                        session_info = session_manager.get_session_info(sid)
                        if session_info:
                            session_info.system_prompt = data.get("system_prompt")
                    elif data.get("type") == "input_audio_buffer.append":
                        base64_audio = data.get("audio")
                        if base64_audio:
                            audio_bytes = base64.b64decode(base64_audio)
                            
                            if current_session_id:
                                session_info = session_manager.get_session_info(current_session_id)
                                if session_info:
                                    session_info.audio_buffer += audio_bytes
                                    
                            vad_queue.put_nowait(audio_bytes)
                            
                            # 傳送音訊給 ASR
                            await asr_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": base64_audio,
                            }))
                            
                elif "bytes" in raw_msg and raw_msg["bytes"]:
                    audio_bytes = raw_msg["bytes"]
                    
                    if current_session_id:
                        session_info = session_manager.get_session_info(current_session_id)
                        if session_info:
                            session_info.audio_buffer += audio_bytes

                    vad_queue.put_nowait(audio_bytes)

                    # 傳送音訊給 ASR
                    await asr_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                    }))

        # ==========================================
        # 啟動雙向監控：任一方斷線，另一方立刻安全終止
        # ==========================================
        task_asr = asyncio.create_task(receive_from_asr())
        task_frontend = asyncio.create_task(receive_from_frontend())
        task_vad = asyncio.create_task(process_vad())

        # 等待任務中「最先結束」的那個 (FIRST_COMPLETED)
        done, pending = await asyncio.wait(
            [task_asr, task_frontend, task_vad],
            return_when=asyncio.FIRST_COMPLETED
        )

        # 將還卡在 await 接收狀態的另一個任務取消，防止成為殭屍連線
        for task in pending:
            task.cancel()

        # 將任務裡面的報錯印出來 (如果有的話)
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, websockets.exceptions.ConnectionClosed)):
                logger.error(f"任務發生異常: {exc}")

    except ConnectionRefusedError:
        logger.error("無法連線到 ASR 伺服器！請確認環境。")
        await websocket.close(code=1011, reason="ASR Server Unavailable")
    except WebSocketDisconnect:
        logger.info("前端正常掛斷。")
    except Exception as e:
        logger.error(f"WebSocket 發生未預期錯誤: {e}")
    finally:
        if asr_ws is not None and not asr_ws.close:
            await asr_ws.close()
        # 不管是哪邊斷線，最後都會掉進這裡執行完美的善後處理
        if current_session_id:
            logger.info("開始執行音訊存檔與清理流程...")
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "recorder"))
            saved_path = session_manager.save_session_audio(current_session_id, base_dir)
            
            if saved_path:
                logger.info(f"錄音已成功儲存: {saved_path}")
            else:
                logger.warning("錄音未儲存：原因可能是完全沒有收到音訊")
                
            session_manager.close_session(current_session_id)

app.mount("/", _HTTPOnlyStaticFiles(StaticFiles(directory="frontend", html=True)), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=7985, reload=True)