#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
from supabase import create_client, Client

# --- 1. 환경 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# Supabase 설정
sb_url = os.getenv("SUPABASE_URL")
sb_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

# MiniMax(Anthropic) 설정
ai_key = os.getenv("ANTHROPIC_API_KEY")
ai_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")
ai_model = os.getenv("MINIMAX_MODEL", "abab6.5-chat")

def main():
    print("\n🔗 [통합 테스트] AI 답변 생성 -> DB 저장 시나리오 시작\n")

    # 1. 클라이언트 초기화
    if not (sb_url and sb_key and ai_key):
        print("[ERROR] .env 키 설정이 부족합니다.")
        return

    try:
        supabase: Client = create_client(sb_url, sb_key)
        ai_client = Anthropic(api_key=ai_key, base_url=ai_url)
        print("✅ 클라이언트 초기화 완료 (Supabase + MiniMax)")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    # 2. AI에게 질문하기
    user_question = "짧게 인사를 해줘."  # 테스트용 질문
    print(f"\n🗣️ User: {user_question}")
    print("⏳ AI 생각 중...")

    try:
        resp = ai_client.messages.create(
            model=ai_model,
            max_tokens=200,
            messages=[{"role": "user", "content": user_question}]
        )
        
        # 응답 추출
        answer_parts = [block.text for block in resp.content if block.type == 'text']
        ai_answer = "".join(answer_parts).strip()
        print(f"🤖 AI: {ai_answer}")

    except Exception as e:
        print(f"❌ AI 응답 실패: {e}")
        return

    # 3. DB에 대화 내용 저장하기 (qa_logs 테이블)
    # SQL 스키마에 정의된 public.qa_logs 테이블 사용
    print("\n💾 DB 저장 시도 중 (qa_logs 테이블)...")
    
    log_data = {
        "question": user_question,
        "answer": ai_answer,
        "session_id": "TEST_SESSION_001", # 테스트용 세션 ID
        "device_id": "WSL_TEST_DEVICE",
        "created_at": datetime.now().isoformat()
    }

    try:
        # insert().execute() 패턴 사용
        result = supabase.table("qa_logs").insert(log_data).execute()
        
        # 결과 확인
        if result.data:
            print(f"✅ DB 저장 성공! (ID: {result.data[0]['id']})")
            print("   -> Supabase 대시보드 'qa_logs' 테이블에서 확인 가능합니다.")
        else:
            print("⚠️ DB 저장 요청은 갔으나 반환 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ DB 저장 실패: {e}")

if __name__ == "__main__":
    main()