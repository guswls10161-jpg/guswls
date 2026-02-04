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
LLM_RETRY = int(os.getenv("LLM_RETRY") or "2")                 # 재시도 횟수(기본 2회)
LLM_BACKOFF = float(os.getenv("LLM_BACKOFF") or "0.7")         # 백오프 base (기본 0.7s)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or "0.7") # 기본 temperature
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS") or "256")      # 길이 제한(기본 256)

# 디버그/로깅
LLM_DEBUG = (os.getenv("LLM_DEBUG") or "0").strip().lower() in ("1","true","yes","y","on")
LLM_LOG_FILE = (os.getenv("LLM_LOG_FILE") or "").strip()       # 예: backend/logs/llm_client.log

DEFAULT_SYSTEM = "너는 스마트미러용 한국어 도우미야. 짧고 정확하게 답해줘."

def _require_env():
    if not MINIMAX_API_KEY or not MINIMAX_BASE_URL:
        raise RuntimeError("MINIMAX_API_KEY / MINIMAX_BASE_URL 가 비어있습니다. 루트 .env를 확인하세요.")

def _short(text: str, n: int = 800) -> str:
    t = text or ""
    if len(t) <= n:
        return t
    return t[:n] + " …(truncated)"

def _log(line: str):
    if LLM_DEBUG:
        print(line)
    if LLM_LOG_FILE:
        try:
            os.makedirs(os.path.dirname(LLM_LOG_FILE), exist_ok=True)
            with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            pass

def _extract_text(js: Any) -> Optional[str]:
    """
    가능한 응답 형태들을 우선순위로 파싱해서 '답변 텍스트'만 반환.
    """
    if js is None:
        return None

    # 1) OpenAI-like: choices[0].message.content
    if isinstance(js, dict):
        choices = js.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0] or {}
            msg = c0.get("message") or {}
            content = msg.get("content")
            if content:
                return str(content).strip()
            # 일부 변형: choices[0].text
            txt = c0.get("text")
            if txt:
                return str(txt).strip()

    # 2) MiniMax/Legacy-like keys
    if isinstance(js, dict):
        for key in ("output_text", "result", "text", "reply", "answer", "content"):
            v = js.get(key)
            if v:
                return str(v).strip()

    # 3) dict 안에 nested로 있는 경우(가끔)
    if isinstance(js, dict):
        data = js.get("data")
        if isinstance(data, dict):
            for key in ("output_text", "text", "reply", "result"):
                v = data.get(key)
                if v:
                    return str(v).strip()

    # 4) list 형태는 "정답일 가능성 낮음"이라 매우 보수적으로 처리
    #    - list of dict에서 text/reply 같은게 있으면 그것만
    if isinstance(js, list) and js:
        first = js[0]
        if isinstance(first, dict):
            for key in ("output_text", "text", "reply", "result", "content"):
                v = first.get(key)
                if v:
                    return str(v).strip()

    return None

def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Tuple[int, str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    status = r.status_code
    raw = r.text or ""
    try:
        js = r.json()
    except Exception:
        js = None
    return status, raw, js

def ask_minimax(user_text: str, system_text: str = DEFAULT_SYSTEM) -> str:
    _require_env()

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    # 후보 엔드포인트들
    candidates = [
        (
            f"{MINIMAX_BASE_URL}/v1/chat/completions",
            {
                "model": MINIMAX_MODEL,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
            "openai"
        ),
        (
            f"{MINIMAX_BASE_URL}/v1/text/chatcompletion",
            {
                "model": MINIMAX_MODEL,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
            },
            "legacy1"
        ),
    ]

    # 각 후보에 대해: (재시도 포함) -> 성공하면 즉시 반환
    last_err = None
    for url, payload, mode in candidates:
        for attempt in range(LLM_RETRY + 1):
            t0 = time.time()
            try:
                _log(f"[minimax_client] try mode={mode} attempt={attempt+1}/{LLM_RETRY+1} url={url} model={MINIMAX_MODEL} timeout={LLM_TIMEOUT}")

                status, raw, js = _post_json(url, payload, headers, timeout=LLM_TIMEOUT)
                latency_ms = int((time.time() - t0) * 1000)

                if status >= 400:
                    last_err = f"{mode} http{status} ({latency_ms}ms): {_short(raw)}"
                    _log(f"[minimax_client] fail {last_err}")
                else:
                    text = _extract_text(js)
                    if text:
                        _log(f"[minimax_client] ok mode={mode} ({latency_ms}ms) chars={len(text)}")
                        return text
                    last_err = f"{mode} parse_failed ({latency_ms}ms): {_short(json.dumps(js, ensure_ascii=False)) if js is not None else _short(raw)}"
                    _log(f"[minimax_client] fail {last_err}")

            except Exception as e:
                last_err = f"{mode} exception: {e}"
                _log(f"[minimax_client] fail {last_err}")

            # 백오프 후 재시도
            if attempt < LLM_RETRY:
                time.sleep(LLM_BACKOFF * (2 ** attempt))

    raise RuntimeError(f"MiniMax 호출 실패: {last_err}")
