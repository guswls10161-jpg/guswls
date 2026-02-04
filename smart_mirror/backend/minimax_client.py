# -*- coding: utf-8 -*-
import os
import time
import json
import requests
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, Tuple, Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

MINIMAX_API_KEY = (os.getenv("MINIMAX_API_KEY") or "").strip()
MINIMAX_MODEL = (os.getenv("MINIMAX_MODEL") or "").strip() or "MiniMax-Text-01"
MINIMAX_BASE_URL = (os.getenv("MINIMAX_BASE_URL") or "").strip().rstrip("/")

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT") or "15")
LLM_RETRY = int(os.getenv("LLM_RETRY") or "2")
LLM_BACKOFF = float(os.getenv("LLM_BACKOFF") or "0.7")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or "0.7")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS") or "256")

def _log(msg):
    # print(msg) 
    pass

def _short(s, L=100):
    if not s: return ""
    return s if len(s) <= L else s[:L] + "..."

def _post_json(url, payload, headers, timeout):
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        try:
            js = resp.json()
        except:
            js = None
        return resp.status_code, resp.text, js
    except Exception as e:
        return 999, str(e), None

def _extract_text(js):
    if not js: return None
    # 구조에 따라 파싱
    if "reply" in js: return js["reply"]
    if "choices" in js and len(js["choices"]) > 0:
        return js["choices"][0]["message"]["text"]
    if "base_resp" in js and js["base_resp"].get("status_code") != 0:
        return None
    return None

def ask_minimax(prompt: str, system_text: str = "You are a helpful assistant.") -> Optional[str]:
    if not MINIMAX_API_KEY:
        _log("[minimax_client] No API Key")
        return "API Key가 설정되지 않았습니다."

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # MiniMax API 구조에 맞게 payload 구성
    # (실제 API 버전에 따라 다를 수 있으나, 기존 코드 유지)
    payload = {
        "model": "abab5.5-chat", # 혹은 MINIMAX_MODEL
        "tokens_to_generate": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "top_p": 0.9,
        "messages": [
            {"sender_type": "USER", "sender_name": "User", "text": prompt}
        ],
        "bot_setting": [
            {
                "bot_name": "Smart Mirror",
                "content": system_text
            }
        ],
        "reply_constraints": {"sender_type": "BOT", "sender_name": "Smart Mirror"}
    }
    
    # URL 후보 (환경변수 없으면 기본값)
    url = "https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=" + (os.getenv("MINIMAX_GROUP_ID") or "")

    for attempt in range(LLM_RETRY + 1):
        t0 = time.time()
        status, raw, js = _post_json(url, payload, headers, timeout=LLM_TIMEOUT)
        
        if status == 200:
            text = _extract_text(js)
            if text:
                return text
        
        # 실패 시 재시도 대기
        time.sleep(LLM_BACKOFF * (2 ** attempt))
        
    return "죄송합니다. AI 응답을 받을 수 없습니다."