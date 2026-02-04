#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# --- 1. 환경 설정 및 .env 로드 ---
# 현재 파일(src/db_test.py)의 상위 상위 폴더(smart_mirror1)를 루트로 잡음
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

print(f"[INFO] 프로젝트 루트: {PROJECT_ROOT}")
print(f"[INFO] .env 파일 경로: {ENV_PATH}")

if not ENV_PATH.exists():
    print("[ERROR] .env 파일을 찾을 수 없습니다.")
    sys.exit(1)

# .env 파일 로드
load_dotenv(dotenv_path=ENV_PATH)

# --- 2. Supabase 키 가져오기 (사용자 환경 맞춤) ---
url = os.getenv("SUPABASE_URL")
# 보내주신 사진에 있는 이름(ANON_KEY)을 우선 사용하고, 없으면 KEY를 찾음
key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

if not url or not key:
    print("\n[ERROR] .env 파일에서 Supabase 키를 찾을 수 없습니다.")
    print(f" - SUPABASE_URL 상태: {'OK' if url else 'MISSING'}")
    print(f" - SUPABASE_ANON_KEY 상태: {'OK' if key else 'MISSING'}")
    sys.exit(1)

print(f"[INFO] URL 감지됨: {url[:20]}...")
print(f"[INFO] KEY 감지됨: {key[:10]}... (보안을 위해 일부만 표시)")

# --- 3. 연결 및 데이터 조회 테스트 ---
try:
    print("\n[STEP 1] 클라이언트 생성 중...")
    supabase: Client = create_client(url, key)
    print(" -> 클라이언트 생성 완료.")

    print("\n[STEP 2] DB 연결 테스트 ('requests' 테이블 조회)...")
    # requests 테이블에서 데이터 1개만 가져와서 연결 확인
    response = supabase.table("requests").select("*").limit(1).execute()
    
    data = response.data
    print(f" -> [성공] 데이터를 성공적으로 가져왔습니다!")
    
    if len(data) > 0:
        print(f" -> 첫 번째 데이터 샘플 ID: {data[0].get('id')}")
        print(f" -> 첫 번째 데이터 내용: {data[0].get('phrase_text', '내용없음')}")
    else:
        print(" -> (테이블이 비어있지만, 에러 없이 연결되었습니다)")

except Exception as e:
    print(f"\n[FAIL] 테스트 실패: {e}")
    print("TIP: URL이 올바른지, KEY가 만료되지 않았는지 확인해주세요.")