# -*- coding: utf-8 -*-
import os
import time
import pyttsx3
import sys
from dotenv import load_dotenv
from datetime import datetime

# DB 관련 함수 임포트
try:
    from supabase_rest import fetch_rows, update_by_id, patch_rows, insert_row
except ImportError:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

POLL_SEC = float(os.getenv("POLL_SEC") or "1.0")
DEVICE_ID = (os.getenv("DEVICE_ID") or "WSL_DEVICE").strip()
SESSION_ID = (os.getenv("SESSION_ID") or "").strip()

def now_iso():
    return datetime.now().isoformat()

def safe_insert_event(event_type, data):
    try:
        if 'insert_row' in globals():
            insert_row(event_type, data)
    except:
        pass

def init_engine():
    """OS에 따라 적절한 TTS 드라이버 초기화"""
    try:
        if sys.platform == 'linux':
            # WSL/Linux에서는 espeak 사용
            engine = pyttsx3.init('espeak')
        else:
            # 윈도우는 sapi5
            engine = pyttsx3.init('sapi5')
            
        # 속도/볼륨 설정
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        print(f"[TTS Init Error] {e}")
        print("WSL 사용 시: sudo apt install libespeak1 실행 필요")
        return None

def main_loop():
    engine = init_engine()
    if not engine:
        print("TTS 엔진 초기화 실패. 종료합니다.")
        return

    print(f">> [TTS Speaker] 시작 (Device: {DEVICE_ID})")

    while True:
        try:
            # 1. DB에서 읽을 요청 가져오기 (status='done')
            # 본인 device_id인 것만 가져오도록 필터링 가능
            rows = fetch_rows("requests", match={"status": "done"}, limit=1)
            
            if not rows:
                time.sleep(POLL_SEC)
                continue

            req = rows[0]
            rid = req['id']
            ans = req.get('result_text', '')
            q = req.get('phrase_text', '')
            
            # 2. 말하기
            if ans:
                print(f"[Speaking] {ans[:30]}...")
                try:
                    engine.say(ans)
                    engine.runAndWait()
                    
                    # 성공 처리
                    update_by_id("requests", rid, {"status": "spoken"})
                    
                except Exception as e:
                    print(f"[TTS Error] {e}")
                    # 에러 처리
                    update_by_id("requests", rid, {"status": "error"})
                    # 엔진 재부팅 시도
                    engine = init_engine()
            else:
                # 텍스트가 없으면 그냥 spoken 처리
                update_by_id("requests", rid, {"status": "spoken"})

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Loop Error] {e}")
            time.sleep(POLL_SEC)

if __name__ == "__main__":
    main_loop()