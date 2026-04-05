import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import logging
logger = logging.getLogger("uvicorn.error")


def _safe_get_json(url: str, params: Dict[str, Any], timeout_sec: float = 3.0) -> Optional[Dict[str, Any]]:
	"""穩定優先的 HTTP JSON 請求：短 timeout、可預期失敗回傳。"""
	try:
		resp = requests.get(url, params=params, timeout=timeout_sec)
		if resp.status_code != 200:
			return None
		return resp.json()
	except Exception:
		return None


def weather_forecast_tool(location: str) -> str:
	"""查詢天氣預報（Open-Meteo，免 API key）。"""
	if not location:
		return "請提供地點，例如：高雄、台北。"

	geo = _safe_get_json(
		"https://geocoding-api.open-meteo.com/v1/search",
		{"name": location, "count": 1, "language": "zh", "format": "json"},
	)
	if not geo or not geo.get("results"):
		return f"目前找不到 {location} 的定位資料。"

	point = geo["results"][0]
	lat = point.get("latitude")
	lon = point.get("longitude")
	city = point.get("name", location)

	weather = _safe_get_json(
		"https://api.open-meteo.com/v1/forecast",
		{
			"latitude": lat,
			"longitude": lon,
			"current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
			"timezone": "Asia/Taipei",
		},
	)
	if not weather or not weather.get("current"):
		return f"{city} 天氣服務暫時不可用。"

	c = weather["current"]
	return (
		f"{city} 即時天氣：氣溫 {c.get('temperature_2m', 'N/A')}°C、"
		f"降雨 {c.get('precipitation', 'N/A')} mm、"
		f"風速 {c.get('wind_speed_10m', 'N/A')} km/h。"
	)


def road_congestion_tool(area: str) -> str:
	"""道路擁擠工具。若無即時交通金鑰，使用穩定 fallback。"""
	area_text = (area or "目前區域").strip()
	tomtom_key = os.getenv("TOMTOM_API_KEY", "").strip()

	# 優先維持穩定：無 key 時提供可理解且固定格式的結果
	if not tomtom_key:
		return (
			f"{area_text} 路況推估：中度擁擠。"
			"建議避開尖峰 07:30-09:00 與 17:30-19:00，"
			"優先走主要幹道替代路線。"
		)

	geo = _safe_get_json(
		"https://api.tomtom.com/search/2/geocode/{0}.json".format(area_text),
		{"key": tomtom_key, "limit": 1},
	)
	if not geo or not geo.get("results"):
		return f"暫時無法取得 {area_text} 的路況定位資訊。"

	pos = geo["results"][0].get("position", {})
	lat = pos.get("lat")
	lon = pos.get("lon")
	if lat is None or lon is None:
		return f"暫時無法取得 {area_text} 的座標資訊。"

	traffic = _safe_get_json(
		"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
		{"key": tomtom_key, "point": f"{lat},{lon}"},
	)
	flow = (traffic or {}).get("flowSegmentData", {})
	if not flow:
		return f"{area_text} 即時交通資料暫時不可用。"

	cur = flow.get("currentSpeed")
	free = flow.get("freeFlowSpeed")
	ratio = (cur / free) if cur and free else None
	level = "順暢"
	if ratio is not None:
		if ratio < 0.4:
			level = "高度擁擠"
		elif ratio < 0.7:
			level = "中度擁擠"
	return f"{area_text} 即時路況：{level}（目前 {cur} km/h，自由流 {free} km/h）。"


def air_quality_tool(location: str) -> str:
	"""空氣品質工具（Open-Meteo air quality）。"""
	if not location:
		return "請提供地點，例如：台中。"

	geo = _safe_get_json(
		"https://geocoding-api.open-meteo.com/v1/search",
		{"name": location, "count": 1, "language": "zh", "format": "json"},
	)
	if not geo or not geo.get("results"):
		return f"目前找不到 {location} 的定位資料。"

	point = geo["results"][0]
	city = point.get("name", location)
	lat = point.get("latitude")
	lon = point.get("longitude")

	aq = _safe_get_json(
		"https://air-quality-api.open-meteo.com/v1/air-quality",
		{
			"latitude": lat,
			"longitude": lon,
			"current": "us_aqi,pm2_5,pm10",
			"timezone": "Asia/Taipei",
		},
	)
	if not aq or not aq.get("current"):
		return f"{city} 空氣品質資料暫時不可用。"

	cur = aq["current"]
	return (
		f"{city} 空氣品質：AQI {cur.get('us_aqi', 'N/A')}、"
		f"PM2.5 {cur.get('pm2_5', 'N/A')}、PM10 {cur.get('pm10', 'N/A')}。"
	)


def emergency_hotline_tool(query: str) -> str:
	"""語音通話常用緊急資訊工具。"""
	text = (query or "").strip()
	if "火" in text or "救護" in text:
		return "緊急協助：119（消防與救護）。"
	if "報案" in text or "警" in text:
		return "緊急協助：110（警察報案）。"
	return "常用緊急電話：110（警察）、119（消防/救護）、1968（高速公路路況）。"


def list_voice_call_tools() -> List[Dict[str, str]]:
	"""前端可直接顯示的工具清單。"""
	return [
		{
			"name": "weather_forecast_tool",
			"display_name": "天氣預報",
			"description": "查詢指定地點即時天氣，回傳氣溫、降雨、風速。",
			"example": "高雄今天天氣如何？",
		},
		{
			"name": "road_congestion_tool",
			"display_name": "道路擁擠",
			"description": "查詢或推估區域路況，適用通話中導航建議。",
			"example": "中山高現在塞不塞？",
		},
		{
			"name": "air_quality_tool",
			"display_name": "空氣品質",
			"description": "查詢 AQI、PM2.5、PM10。",
			"example": "台中空氣品質如何？",
		},
		{
			"name": "emergency_hotline_tool",
			"display_name": "緊急電話",
			"description": "快速提供 110/119/1968 等常用資訊。",
			"example": "要打哪支電話叫救護車？",
		},
	]


class StreamingVoiceAgent:
	"""可串流輸出的語音通話 LLM Agent，穩定優先。"""

	def __init__(self, logger: Optional[logging.Logger] = None):
		self.logger = logger or logging.getLogger("uvicorn.error")
		self._llm = self._build_llm()

	def _build_llm(self) -> Optional[AzureChatOpenAI]:
		endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
		api_key = os.getenv("AZURE_OPENAI_API_KEY")
		deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_MODEL")
		api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")

		if not endpoint or not api_key or not deployment:
			self.logger.warning("Azure OpenAI 環境變數不完整，LLM 將使用 fallback 模式。")
			return None

		try:
			return AzureChatOpenAI(
				azure_endpoint=endpoint,
				api_key=api_key,
				azure_deployment=deployment,
				api_version=api_version,
				temperature=0.2,
				max_retries=2,
				timeout=15,
			)
		except Exception as exc:
			self.logger.error(f"建立 AzureChatOpenAI 失敗: {exc}")
			return None

	@staticmethod
	def _extract_tool_arg(text: str) -> str:
		cleaned = (text or "").replace("？", "").replace("?", "").strip()
		tokens = ["天氣", "路況", "擁擠", "空氣", "AQI", "救護", "報案"]
		for token in tokens:
			cleaned = cleaned.replace(token, "")
		cleaned = cleaned.strip(" ,，。")
		return cleaned or "高雄"

	def _route_tool(self, user_text: str) -> Optional[str]:
		t = (user_text or "").lower()
		arg = self._extract_tool_arg(user_text)

		if "天氣" in user_text:
			return weather_forecast_tool(arg)
		if "路況" in user_text or "擁擠" in user_text or "塞" in user_text:
			return road_congestion_tool(arg)
		if "空氣" in user_text or "aqi" in t or "pm2.5" in t:
			return air_quality_tool(arg)
		if "110" in user_text or "119" in user_text or "救護" in user_text or "報案" in user_text:
			return emergency_hotline_tool(user_text)
		return None

	@staticmethod
	def _chunk_text(text: str, size: int = 24) -> List[str]:
		return [text[i:i + size] for i in range(0, len(text), size)] or [""]

	async def stream_chat(
		self,
		user_text: str,
		history: Optional[list] = None,
		system_prompt: Optional[str] = None,
	) -> AsyncIterator[str]:
		"""主串流入口：先工具路由，否則走 LLM 串流。"""
		history = history or []
		now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

		routed = self._route_tool(user_text)
		if routed is not None:
			for part in self._chunk_text(routed):
				yield part
				await asyncio.sleep(0)
			return

		if self._llm is None:
			fallback = (
				"目前語言模型服務暫時不可用，但語音工具仍可使用："
				"天氣預報、道路擁擠、空氣品質、緊急電話。"
			)
			for part in self._chunk_text(fallback):
				yield part
				await asyncio.sleep(0)
			return

		messages: List[Any] = [
			SystemMessage(
				content=(
					(system_prompt or "你是語音通話助理，請用繁體中文簡潔回答。")
					+ f"\n目前時間：{now}\n"
					+ "若可由內建工具回答（天氣/路況/空氣/緊急電話），請優先提供可執行建議。"
				)
			)
		]

		# history 可支援：LangChain message 物件 或簡單 dict({role, content})
		for msg in history:
			if msg is None:
				continue

			if isinstance(msg, SystemMessage):
				# system_prompt 由參數統一控制，避免混入多個 system
				continue

			if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
				content = getattr(msg, "content", "")
				if content:
					messages.append(msg)
				continue

			if isinstance(msg, dict):
				role = (msg.get("role") or "").lower()
				content = msg.get("content") or ""
				if not content:
					continue
				if role == "assistant":
					messages.append(AIMessage(content=content))
				elif role == "user":
					messages.append(HumanMessage(content=content))
				continue

			# 最後 fallback：直接轉字串當 user
			messages.append(HumanMessage(content=str(msg)))
		messages.append(HumanMessage(content=user_text))
		try:
			async for chunk in self._llm.astream(messages):
				content = getattr(chunk, "content", "")
				if isinstance(content, str) and content:
					yield content
				elif isinstance(content, list):
					for item in content:
						text = item.get("text") if isinstance(item, dict) else str(item)
						if text:
							yield text
		except Exception as exc:
			self.logger.error(f"LLM 串流失敗: {exc}")
			err_text = "目前回覆服務繁忙，請稍後再試。"
			for part in self._chunk_text(err_text):
				yield part
				await asyncio.sleep(0)


def build_tool_catalog_message() -> Dict[str, Any]:
	return {
		"type": "response.llm_tools",
		"category": "voice_call_tools",
		"tools": list_voice_call_tools(),
	}

