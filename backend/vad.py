# record_and_predict.py 
import os
import time
import math
import urllib.request
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import numpy as np
from scipy.io import wavfile
import onnxruntime as ort

from vad_inference import predict_endpoint
import logging

logger = logging.getLogger("uvicorn.error")

# --- 基础配置（固定 16 kHz 单声道，512 样本块）---
RATE = 16000
CHUNK = 512                     # Silero VAD 在 16 kHz 下需要 512 个样本
# FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- VAD 配置 ---
VAD_THRESHOLD = 0.7             # 语音概率阈值
PRE_SPEECH_MS = 200             # 触发前保留的毫秒数

# --- 动态端点检测配置 ---
EARLY_CHECK_MS = 1500            # 静音后多久开始第一次检测
CHECK_INTERVAL_MS = 150         # 基础检测间隔
MIN_CHECK_INTERVAL_MS = 100      # 高置信度时的最小检测间隔
MAX_STOP_MS = 2500              # 最大静音等待时间（兜底）
MAX_DURATION_SECONDS = 20        # 每段音频的最大时长上限

# --- 置信度阈值 ---
HIGH_CONFIDENCE = 0.70          # 高置信度阈值，可立即结束
MEDIUM_CONFIDENCE = 0.50        # 中等置信度
LOW_CONFIDENCE = 0.30           # 低置信度，需继续等待

# --- 调试配置 ---
DEBUG_SAVE_WAV = False
TEMP_OUTPUT_WAV = "temp_output.wav"
DEBUG_LOG = True                # 是否打印检测日志

# --- Silero ONNX 模型 ---
ONNX_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
ONNX_MODEL_PATH = "backend\\vad_model\\silero_vad.onnx"
MODEL_RESET_STATES_TIME = 5.0

class SileroVAD:
    """Silero VAD ONNX 封装类，适用于 16 kHz 单声道，块大小为 512。"""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.context_size = 64
        self._state = None
        self._context = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= MODEL_RESET_STATES_TIME:
            self._init_states()
            self._last_reset_time = time.time()

    def prob(self, chunk_f32: np.ndarray) -> float:
        """计算一个长度为 512 的音频块的语音概率。"""
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            raise ValueError(f"期望 {CHUNK} 個樣本，實際得到 {x.shape[1]} 個樣本")
        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64)
        }
        out, self._state = self.session.run(None, ort_inputs)
        self._context = x[:, -self.context_size:]
        self.maybe_reset()

        return float(out[0][0])

class AsyncSmartTurnDetector:
    """异步 Smart Turn 检测器，支持预热和异步推理。"""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.current_future: Optional[Future] = None
        self.last_result: Optional[dict] = None
        self.last_audio_hash: Optional[int] = None
        self._lock = threading.Lock()

        # 预热模型（首次加载较慢）
        self._warmup()

    def _warmup(self):
        """预热模型，减少首次推理延迟。"""
        logger.info("正在預熱 Smart Turn 模型...")
        dummy_audio = np.zeros(RATE, dtype=np.float32)  # 1 秒静音
        predict_endpoint(dummy_audio)
        logger.info("Smart Turn 模型預熱完成。")

    def submit_async(self, audio_segment: np.ndarray) -> Future:
        """异步提交推理任务。"""
        with self._lock:
            # 取消之前的任务（如果还在运行）
            if self.current_future and not self.current_future.done():
                self.current_future.cancel()

            audio_hash = hash(audio_segment.tobytes()[-8000:])  # 只用最后 0.5 秒做哈希
            self.last_audio_hash = audio_hash
            self.current_future = self.executor.submit(self._run_inference, audio_segment, audio_hash)
            return self.current_future

    def _run_inference(self, audio_segment: np.ndarray, audio_hash: int) -> dict:
        """执行推理。"""
        t0 = time.perf_counter()
        result = predict_endpoint(audio_segment)
        result['inference_time_ms'] = (time.perf_counter() - t0) * 1000.0
        result['audio_hash'] = audio_hash

        with self._lock:
            self.last_result = result

        return result

    def get_result_if_ready(self) -> Optional[dict]:
        """非阻塞获取结果（如果已完成）。"""
        with self._lock:
            if self.current_future and self.current_future.done():
                try:
                    return self.current_future.result(timeout=0)
                except Exception:
                    return None
            return None

    def get_result_blocking(self, timeout: float = 0.5) -> Optional[dict]:
        """阻塞等待结果。"""
        with self._lock:
            future = self.current_future

        if future:
            try:
                return future.result(timeout=timeout)
            except Exception:
                return None
        return None

    def shutdown(self):
        """关闭线程池。"""
        self.executor.shutdown(wait=False)

class DynamicEndpointDetector:
    """动态端点检测器，实现智能静音判断策略。"""

    def __init__(self):
        self.chunk_ms = (CHUNK / RATE) * 1000.0

        # 转换为 chunk 数量
        self.early_check_chunks = math.ceil(EARLY_CHECK_MS / self.chunk_ms)
        self.base_interval_chunks = math.ceil(CHECK_INTERVAL_MS / self.chunk_ms)
        self.min_interval_chunks = math.ceil(MIN_CHECK_INTERVAL_MS / self.chunk_ms)
        self.max_stop_chunks = math.ceil(MAX_STOP_MS / self.chunk_ms)

        # 状态
        self.last_check_silence_chunks = 0
        self.last_probability = 0.0
        self.check_count = 0
        self.pending_inference = False

    def reset(self):
        """重置检测状态。"""
        self.last_check_silence_chunks = 0
        self.last_probability = 0.0
        self.check_count = 0
        self.pending_inference = False

    def get_dynamic_interval(self) -> int:
        """根据上次概率动态计算检测间隔（chunk 数量）。"""
        if self.last_probability >= HIGH_CONFIDENCE:
            # 高置信度：更频繁检测
            return self.min_interval_chunks
        elif self.last_probability >= MEDIUM_CONFIDENCE:
            # 中等置信度：正常间隔
            return self.base_interval_chunks
        else:
            # 低置信度：稍长间隔
            return int(self.base_interval_chunks * 1.5)

    def should_check(self, trailing_silence_chunks: int) -> bool:
        """判断是否应该进行端点检测。"""
        if self.pending_inference:
            return False

        # 首次检测
        if self.check_count == 0 and trailing_silence_chunks >= self.early_check_chunks:
            return True

        # 后续检测：基于动态间隔
        if self.check_count > 0:
            chunks_since_last = trailing_silence_chunks - self.last_check_silence_chunks
            interval = self.get_dynamic_interval()
            if chunks_since_last >= interval:
                return True

        return False

    def should_force_end(self, trailing_silence_chunks: int, since_trigger_chunks: int, max_chunks: int) -> bool:
        """判断是否强制结束。"""
        return (trailing_silence_chunks >= self.max_stop_chunks or
                since_trigger_chunks >= max_chunks)

    def on_check_started(self, trailing_silence_chunks: int):
        """记录检测开始。"""
        self.pending_inference = True

    def on_check_completed(self, trailing_silence_chunks: int, probability: float):
        """记录检测完成。"""
        self.last_check_silence_chunks = trailing_silence_chunks
        self.last_probability = probability
        self.check_count += 1
        self.pending_inference = False

    def should_end_by_confidence(self, probability: float, trailing_silence_chunks: int) -> bool:
        """基于置信度判断是否应该结束。"""
        silence_ms = trailing_silence_chunks * self.chunk_ms

        if probability >= HIGH_CONFIDENCE:
            # 高置信度：200ms 静音即可结束
            return silence_ms >= EARLY_CHECK_MS
        elif probability >= MEDIUM_CONFIDENCE:
            # 中等置信度：需要更多静音确认
            required_ms = EARLY_CHECK_MS + (HIGH_CONFIDENCE - probability) * 500
            return silence_ms >= required_ms
        else:
            # 低置信度：继续等待
            return False

def ensure_model(path: str = ONNX_MODEL_PATH, url: str = ONNX_MODEL_URL) -> str:
    if not os.path.exists(path):
        logger.info("正在下載 Silero VAD ONNX 模型...")
        urllib.request.urlretrieve(url, path)
        logger.info("ONNX 模型下載完成。")
    return path

class StreamingVAD:
    def __init__(self):
        self.vad = SileroVAD(ensure_model())
        self.detector = DynamicEndpointDetector()
        self.smart_turn = AsyncSmartTurnDetector()
        self.chunk_ms = (CHUNK / RATE) * 1000.0
        self.pre_chunks = math.ceil(PRE_SPEECH_MS / self.chunk_ms)
        self.max_chunks = math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE))
        self.pre_buffer = deque(maxlen=self.pre_chunks)
        self.segment = []
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.audio_buffer = bytearray()

    def process_chunk(self, audio_byte: bytes) -> bool:
        """處理新進來的音訊，如果偵測到端點則回傳 True"""
        self.audio_buffer.extend(audio_byte)
        endpoint_detected = False

        while len(self.audio_buffer) >= CHUNK * 2:
            data = self.audio_buffer[:CHUNK * 2]
            self.audio_buffer = self.audio_buffer[CHUNK * 2:]
            
            int16 = np.frombuffer(data, dtype=np.int16)
            f32 = (int16.astype(np.float32)) / 32768.0

            # VAD 检测
            is_speech = self.vad.prob(f32) > VAD_THRESHOLD

            if not self.speech_active:
                self.pre_buffer.append(f32)
                if is_speech:
                    self.segment = list(self.pre_buffer)
                    self.segment.append(f32)
                    self.speech_active = True
                    self.trailing_silence = 0
                    self.since_trigger_chunks = 1
                    self.detector.reset()
            else:
                self.segment.append(f32)
                self.since_trigger_chunks += 1

                if is_speech:
                    if self.trailing_silence > 0:
                        self.detector.reset()
                    self.trailing_silence = 0
                else:
                    self.trailing_silence += 1

                if self.detector.should_force_end(self.trailing_silence, self.since_trigger_chunks, self.max_chunks):
                    audio_segment = np.concatenate(self.segment, dtype=np.float32)
                    result = self.smart_turn.get_result_blocking(timeout=0.1) or predict_endpoint(audio_segment)
                    _process_segment(audio_segment, result, "強制結束", self.trailing_silence * self.chunk_ms)
                    self._reset_state()
                    endpoint_detected = True
                    continue

                if self.detector.should_check(self.trailing_silence):
                    self.detector.on_check_started(self.trailing_silence)
                    audio_segment = np.concatenate(self.segment, dtype=np.float32)
                    self.smart_turn.submit_async(audio_segment.copy())

                result = self.smart_turn.get_result_if_ready()
                if result and self.detector.pending_inference:
                    prob = result.get("probability", 0)
                    self.detector.on_check_completed(self.trailing_silence, prob)

                    if self.detector.should_end_by_confidence(prob, self.trailing_silence):
                        audio_segment = np.concatenate(self.segment, dtype=np.float32)
                        _process_segment(audio_segment, result, "信賴程度判斷", self.trailing_silence * self.chunk_ms)
                        self._reset_state()
                        endpoint_detected = True

        return endpoint_detected

    def _reset_state(self):
        self.segment.clear()
        self.pre_buffer.clear()
        self.detector.reset()
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0

    def shutdown(self):
        self.smart_turn.shutdown()

streaming_vad_instance = StreamingVAD()

def record_and_predict(audio_byte: bytes) -> bool:
    return streaming_vad_instance.process_chunk(audio_byte)


def _reset_state(segment: list, pre_buffer: deque, detector: DynamicEndpointDetector):
    """重置状态。"""
    segment.clear()
    pre_buffer.clear()
    detector.reset()


def _process_segment(segment_audio_f32: np.ndarray, result: dict, end_reason: str = "", silence_ms: float = 0):
    """处理完成的音频段。"""
    if segment_audio_f32.size == 0:
        logger.warning("檢測到停頓，但音頻段為空，跳過處理。")
        return

    if DEBUG_SAVE_WAV:
        wavfile.write(TEMP_OUTPUT_WAV, RATE, (segment_audio_f32 * 32767.0).astype(np.int16))

    dur_sec = segment_audio_f32.size / RATE
    pred = result.get("prediction", 0)
    prob = result.get("probability", float("nan"))
    inference_time = result.get("inference_time_ms", 0)

    logger.info(f"檢測到使用者停頓，結束原因: {end_reason}，靜音時間: {silence_ms:.0f} ms，預測結果: 表達{'完整' if pred == 1 else '不完整'}，概率: {prob:.4f}")
    if inference_time > 0:
        logger.info(f"Smart Turn 模型推理時間: {inference_time:.1f} ms")

if __name__ == "__main__":
    record_and_predict()