import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------------
# [경로 수정] .env 파일이 'web' 폴더 안에 있으므로 위치를 지정해줍니다.
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))   # src 폴더 위치
project_root = os.path.dirname(current_dir)                # smart_mirror1 (루트)
env_path = os.path.join(project_root, 'web', '.env')       # web/.env 경로 지정

# 명시한 경로의 .env 로드
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------
# [변수명 수정] 사용자님의 .env 파일 변수명(SUPABASE_ANON_KEY) 사용
# -------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")  # 여기가 수정되었습니다!

# (디버깅용) 키가 잘 읽혔는지 확인
if not SUPABASE_KEY:
    print(f"⚠️ 경고: .env 파일을 찾았으나 'SUPABASE_ANON_KEY'가 없습니다.")
    print(f"👉 확인된 경로: {env_path}")

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# -------------------------------------------------------------------
# 1. 질문 저장 (INSERT)
# -------------------------------------------------------------------
def insert_request(question_text):
    url = f"{SUPABASE_URL}/rest/v1/smart_mirror"
    
    payload = {
        "question": question_text,
        "status": "sending",
        "created_at": datetime.now().isoformat()
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            return data[0]['id']
        return None
    except Exception as e:
        print(f"❌ DB Insert Error: {e}")
        return None

# -------------------------------------------------------------------
# 2. 답변 업데이트 (UPDATE)
# -------------------------------------------------------------------
def update_by_id(request_id, ai_response_text):
    url = f"{SUPABASE_URL}/rest/v1/smart_mirror?id=eq.{request_id}"
    
    payload = {
        "response": ai_response_text,
        "status": "completed"
    }

    try:
        response = requests.patch(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"✅ DB Update Success (ID: {request_id})")
        return True
    except Exception as e:
        print(f"❌ DB Update Error: {e}")
        return False