# -*- coding: utf-8 -*-
"""
mirror_morse_ai.py (AI 챗봇 서버)
- 역할: DB에 올라온 질문(requests)을 감시하다가, AI에게 물어보고 답을 달아줍니다.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# [중요 1] 현재 폴더(backend)를 시스템 경로에 추가 (ImportError 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# [중요 2] 상위 폴더(루트)에서 .env 찾기
PROJECT_ROOT = os.path.dirname(current_dir) # backend의 부모 = root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

# 이제 같은 폴더에 있는 모듈들을 불러옵니다.
try:
    from minimax_client import ask_minimax
    from supabase_rest import fetch_rows, insert_row, patch_rows, update_by_id
except ImportError as e:
    print(f"[Fatal Error] 필수 라이브러리를 찾을 수 없습니다: {e}")
    print("pip install requests python-dotenv supabase solapi-libs 명령어를 실행했는지 확인하세요.")
    sys.exit(1)

POLL_SEC = float(os.getenv("POLL_SEC") or "1.0")
MAX_RETRY = int(os.getenv("MAX_RETRY") or "2")
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE") or "0.7")
PROCESSING_TIMEOUT_SEC = float(os.getenv("PROCESSING_TIMEOUT_SEC") or "180")

DEVICE_ID = (os.getenv("DEVICE_ID") or "mirror_device").strip()
DEFAULT_SYSTEM = "너는 스마트미러용 한국어 도우미야. 짧고 정확하게 1-2문장으로 답해줘."

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def safe_insert_qa_log(question, answer, request_id, session_id, device_id):
    try:
        insert_row("qa_logs", {
            "question": question,
            "answer": answer,
            "request_id": request_id,
            "session_id": session_id,
            "device_id": device_id,
        })
    except Exception as e:
        print(f"[Warn] QA 로그 저장 실패: {e}")

def claim_request(rid: str) -> bool:
    """
    pending 상태인 질문을 'processing'으로 바꿔서, 내가 처리하겠다고 찜하는 함수
    """
    try:
        js = patch_rows(
            "requests",
            {"id": f"eq.{rid}", "status": "eq.pending"},
            {"status": "processing"}
        )
        # 업데이트 성공 시 리스트가 반환됨
        if isinstance(js, list) and len(js) > 0:
            return True
        return False
    except Exception:
        return False

def recover_stuck_processing():
    """
    혹시라도 AI가 답변하다가 강제종료되어서, 
    영원히 'processing' 상태로 멈춰있는 질문들을 구해내는 함수
    """
    try:
        # 3분 이상 지난 processing 상태의 요청 찾기
        rows = fetch_rows(
            "requests",
            select="id,created_at",
            filters={"status": "eq.processing"},
            limit=10
        )
        
        if not rows: return

        now = datetime.utcnow()
        for r in rows:
            rid = r.get("id")
            created_at = r.get("created_at")
            if not rid or not created_at: continue

            # 시간 계산 (ISO 포맷 파싱)
            try:
                ts_str = created_at.replace("Z", "")
                ts_dt = datetime.fromisoformat(ts_str)
                age = (now - ts_dt).total_seconds()

                if age >= PROCESSING_TIMEOUT_SEC:
                    # 너무 오래걸리면 다시 'pending'으로 되돌림 (재시도 기회 부여)
                    update_by_id("requests", rid, {"status": "pending"})
                    print(f"[Recover] 멈춰있던 요청 복구함: {rid}")
            except:
                continue
    except:
        pass

def main():
    print(f"\n>> [AI Server] 챗봇 엔진 가동 시작! (Device: {DEVICE_ID})")
    print(f">> [System] .env 위치: {os.path.join(PROJECT_ROOT, '.env')}")
    print(">> Ctrl+C를 누르면 종료됩니다.\n")

    while True:
        try:
            # 1. 멈춰있는 요청 복구
            recover_stuck_processing()

            # 2. 'pending' 상태인 질문 가져오기
            rows = fetch_rows(
                "requests",
                select="id,phrase_text,session_id,device_id",
                filters={"status": "eq.pending"},
                order="created_at.asc",
                limit=1,
            )

            if not rows:
                time.sleep(POLL_SEC)
                continue

            # 3. 데이터 추출
            req = rows[0]
            rid = req.get("id")
            question = req.get("phrase_text")
            session_id = req.get("session_id")
            device_id = req.get("device_id") or DEVICE_ID

            if not question:
                update_by_id("requests", rid, {"status": "error", "result_text": "질문 내용이 없습니다."})
                continue

            print(f">> [New Question] 발견: {question}")

            # 4. 처리 시작 (상태 변경: pending -> processing)
            if not claim_request(rid):
                print(">> [Info] 다른 프로세스가 이미 처리 중입니다.")
                continue

            # 5. 미니맥스 AI 호출
            answer = None
            try:
                answer = ask_minimax(question, system_text=DEFAULT_SYSTEM)
            except Exception as e:
                print(f">> [Error] AI 호출 실패: {e}")
                update_by_id("requests", rid, {"status": "error", "result_text": "AI 서버 연결 오류"})
                continue

            # 6. 결과 저장 (상태 변경: processing -> done)
            if answer:
                print(f">> [Answer] 생성 완료: {answer[:30]}...")
                update_by_id("requests", rid, {"status": "done", "result_text": answer})
                safe_insert_qa_log(question, answer, rid, session_id, device_id)
            else:
                update_by_id("requests", rid, {"status": "error", "result_text": "AI 응답이 비어있습니다."})

        except KeyboardInterrupt:
            print("\n>> [AI Server] 종료합니다.")
            break
        except Exception as e:
            print(f">> [System Error] 루프 에러: {e}")
            time.sleep(1.0)

if __name__ == "__main__":
    main()