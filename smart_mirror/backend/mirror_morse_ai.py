# -*- coding: utf-8 -*-
"""
mirror_morse_ai.py (AI 챗봇 서버)
"""
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

PROJECT_ROOT = os.path.dirname(current_dir)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

try:
    from minimax_client import ask_minimax
    from supabase_rest import fetch_rows, insert_row, patch_rows, update_by_id
except ImportError as e:
    print(f"[Fatal Error] {e}")
    sys.exit(1)

POLL_SEC = 1.0
DEVICE_ID = (os.getenv("DEVICE_ID") or "WSL_DEVICE").strip()
DEFAULT_SYSTEM = "당신은 스마트 미러 AI입니다. 짧고 간결하게 대답하세요."

def claim_request(rid):
    """중복 처리를 막기 위해 상태를 processing으로 변경"""
    res = update_by_id("requests", rid, {"status": "processing"})
    return bool(res)

def main_loop():
    print(f">> [Morse AI] 시작 (Device: {DEVICE_ID})")
    
    while True:
        try:
            # 1. pending 상태인 요청 가져오기
            rows = fetch_rows("requests", match={"status": "pending"}, limit=1)
            if not rows:
                time.sleep(POLL_SEC)
                continue
                
            req = rows[0]
            rid = req['id']
            question = req.get('phrase_text', '')
            
            print(f">> [New Question] {question}")
            
            # 2. 처리중 표시
            if not claim_request(rid):
                continue
                
            # 3. AI 답변 생성
            answer = ask_minimax(question, system_text=DEFAULT_SYSTEM)
            
            # 4. 결과 저장 (done 상태로 변경 -> TTS가 읽음)
            if answer:
                update_by_id("requests", rid, {"status": "done", "result_text": answer})
                print(f">> [Answer Saved] {answer[:20]}...")
            else:
                update_by_id("requests", rid, {"status": "error", "result_text": "AI Error"})

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[AI Loop Error] {e}")
            time.sleep(POLL_SEC)

if __name__ == "__main__":
    main_loop()