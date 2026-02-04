# test_db.py
import os
from supabase_rest import supabase, fetch_rows

print(">> [1단계] DB 연결 테스트 시작...")

if supabase:
    print("✅ Supabase 클라이언트 객체 생성 성공!")
    
    # 실제 데이터 읽어보기 (requests 테이블)
    try:
        rows = fetch_rows("requests", limit=1)
        print(f"✅ DB 통신 성공! (가져온 데이터 개수: {len(rows)})")
        if rows:
            print(f"   -> 최근 데이터 샘플: {rows[0]}")
    except Exception as e:
        print(f"❌ DB 통신 실패: {e}")
        print("   -> .env 파일의 SUPABASE URL/KEY를 확인하세요.")
else:
    print("❌ Supabase 연결 실패. .env 파일을 찾을 수 없거나 키가 없습니다.")