#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import re
import sys
from datetime import datetime
from pathlib import Path

# --- 라이브러리 설치 확인 (없을 경우 안내) ---
try:
    from dotenv import load_dotenv
    from anthropic import Anthropic
except ImportError as e:
    print("\n[ERROR] 필수 라이브러리가 설치되지 않았습니다.")
    print(f"오류 내용: {e}")
    print("터미널에서 다음 명령어를 실행하세요: pip install anthropic python-dotenv\n")
    sys.exit(1)

# --- Project root / paths 설정 ---
# 이 파일(chat_client.py)이 src/ 폴더 안에 있다고 가정합니다.
# parents[1]은 src의 상위 폴더인 프로젝트 루트(smart_mirror1)를 가리킵니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
LOG_FILE = EVIDENCE_DIR / "m1_log.txt"
ENV_FILE = PROJECT_ROOT / ".env"

def ensure_dirs():
    """로그 저장용 폴더가 없으면 생성"""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

def append_log(line: str):
    """로그 파일에 대화 내용을 추가"""
    ensure_dirs()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception as e:
        print(f"[WARN] 로그 저장 실패: {e}")

def trim_history(messages, keep_turns=10):
    """대화 기록이 너무 길어지지 않게 최근 N개 턴만 유지"""
    max_msgs = keep_turns * 2
    return messages[-max_msgs:] if len(messages) > max_msgs else messages

def clean_text(text: str) -> str:
    """<think> 태그 제거 및 텍스트 정리"""
    if text is None:
        return ""
    s = text

    # 1) <think>...</think> 제거 (사고 과정 숨기기)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

    # 2) 한자 범위 제거 (안전장치)
    s = re.sub(r"[\u4e00-\u9fff]+", "", s)

    # 3) 공백/쉼표 정리
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s

def main():
    # --- .env 파일 로드 ---
    if not ENV_FILE.exists():
        print(f"\n[ERROR] .env 파일을 찾을 수 없습니다.")
        print(f"예상 경로: {ENV_FILE}")
        print("프로젝트 루트 폴더에 .env 파일을 생성해주세요.\n")
        return

    load_dotenv(dotenv_path=ENV_FILE, override=False)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")
    model = os.getenv("MINIMAX_MODEL", "abab6.5-chat")
    timeout_s = float(os.getenv("LLM_TIMEOUT", "30"))

    if not api_key:
        print("\n[ERROR] .env 파일에 ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        return

    # 연결 정보 출력 (디버깅용)
    print("-" * 50)
    print(f"[INFO] WSL Environment Detected")
    print(f"[DEBUG] Project Root : {PROJECT_ROOT}")
    print(f"[DEBUG] Model        : {model}")
    print("-" * 50)

    try:
        client = Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )
    except Exception as e:
        print(f"[ERROR] 클라이언트 초기화 실패: {e}")
        return

    # 시스템 프롬프트 설정 (한국어 응답 강제)
    SYSTEM_PROMPT = (
        "You are an AI assistant running via the MiniMax API (Anthropic-compatible endpoint). "
        "Never claim to be ChatGPT, OpenAI, Claude, or Anthropic. "
        "If asked who you are, say: 'I am an AI assistant running via the MiniMax API.'\n"
        "When the user speaks Korean, respond in Korean only. Do not use any Chinese characters. "
        "Keep the answer concise and helpful."
    )

    DEVELOPER_PROMPT = (
        "Follow these rules:\n"
        "1) Be safe and polite.\n"
        "2) If the user asks for code, provide runnable steps.\n"
        "3) Prefer short, clear Korean explanations when user speaks Korean.\n"
        "4) Do not mention OpenAI/ChatGPT/Claude/Anthropic as your identity.\n"
    )

    system_block = f"{SYSTEM_PROMPT}\n\n[Developer]\n{DEVELOPER_PROMPT}"

    messages = []
    print("\n💬 MiniMax Chat CLI 시작 (종료하려면 'exit' 또는 'quit' 입력)")

    while True:
        try:
            user_text = input("\nUser: ").strip()
        except KeyboardInterrupt:
            # Ctrl+C 입력 시 안전하게 종료
            print("\n\n프로그램을 종료합니다.")
            break

        if user_text.lower() in ("exit", "quit"):
            print("대화를 종료합니다.")
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        messages = trim_history(messages, keep_turns=10)

        # 재시도 로직 (최대 3회)
        for attempt in range(3):
            try:
                t0 = time.time()
                print("...", end="", flush=True) # 생각 중 표시

                resp = client.messages.create(
                    model=model,
                    system=system_block,
                    max_tokens=1024,
                    temperature=0.7,
                    messages=messages,
                )
                latency_ms = int((time.time() - t0) * 1000)
                print("\r", end="") # 생각 중 표시 지움

                # 응답 처리
                answer_parts = []
                if resp.content:
                    for block in resp.content:
                        if getattr(block, "type", None) == "text":
                            answer_parts.append(getattr(block, "text", ""))
                
                raw_answer = "".join(answer_parts).strip()
                answer = clean_text(raw_answer) or raw_answer

                print(f"AI: {answer}")

                # 로그 기록용 통계
                ts = datetime.now().isoformat(timespec="seconds")
                usage = getattr(resp, "usage", None)
                
                if usage:
                    stat = (
                        f"latency={latency_ms}ms "
                        f"in={getattr(usage,'input_tokens',0)} "
                        f"out={getattr(usage,'output_tokens',0)}"
                    )
                else:
                    stat = f"latency={latency_ms}ms (usage info N/A)"

                log_line = f"{ts} | User: {user_text} | AI: {answer} | [{stat}]"
                append_log(log_line)

                messages.append({"role": "assistant", "content": answer})
                messages = trim_history(messages, keep_turns=10)
                break # 성공 시 재시도 루프 탈출

            except Exception as e:
                print("\r", end="") # 줄바꿈 정리
                if attempt == 2:
                    print(f"\n[ERROR] 3회 시도 실패: {type(e).__name__}: {e}")
                    break
                
                wait_s = 0.5 * (2 ** attempt)
                print(f"\n[WARN] 응답 지연 (재시도 {attempt+1}/3)...")
                time.sleep(wait_s)

if __name__ == "__main__":
    main()