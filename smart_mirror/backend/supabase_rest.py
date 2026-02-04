import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv
import datetime

# ==========================================
# 1. .env 파일 위치 찾기 (경로 호환성 강화)
# ==========================================
current_file_path = os.path.abspath(__file__)
backend_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(backend_dir)

# 우선순위 1: web 폴더 안의 .env
env_path = os.path.join(root_dir, 'web', '.env')
# 우선순위 2: 없으면 최상위 폴더 확인
if not os.path.exists(env_path):
    env_path = os.path.join(root_dir, '.env')

load_dotenv(env_path)

# ==========================================
# 2. Supabase 연결
# ==========================================
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")

supabase: Client = None

if url and key:
    try:
        supabase = create_client(url, key)
        # print(f">> [DB] 연결 성공! (참조 파일: {env_path})")
    except Exception as e:
        print(f">> [DB] 연결 실패: {e}")
else:
    print(f">> [오류] .env 파일을 찾을 수 없거나 키가 없습니다. ({env_path})")

# ==========================================
# 3. 공통 함수 (Select / Insert / Patch)
# ==========================================
def fetch_rows(table_name, match=None, order_col="created_at", limit=10):
    if not supabase: return []
    try:
        query = supabase.table(table_name).select("*")
        if match:
            for k, v in match.items():
                # WSL/Linux 환경에서 eq 처리
                if isinstance(v, str) and v.startswith("eq."):
                    query = query.eq(k, v.split("eq.")[1])
                else:
                    query = query.eq(k, v)
        
        # order 처리
        if order_col:
            query = query.order(order_col, desc=True)
            
        if limit:
            query = query.limit(limit)
            
        response = query.execute()
        return response.data
    except Exception as e:
        print(f">> DB Fetch Error ({table_name}): {e}")
        return []

def insert_row(table_name, data):
    if not supabase: return None
    try:
        response = supabase.table(table_name).insert(data).execute()
        return response.data
    except Exception as e:
        print(f">> DB Insert Error ({table_name}): {e}")
        return None

def patch_rows(table_name, match, data):
    if not supabase: return None
    try:
        query = supabase.table(table_name).update(data)
        for k, v in match.items():
            if isinstance(v, str) and v.startswith("eq."):
                query = query.eq(k, v.split("eq.")[1])
            else:
                query = query.eq(k, v)
                
        response = query.execute()
        return response.data
    except Exception as e:
        print(f">> DB Update Error ({table_name}): {e}")
        return None

def update_by_id(table_name, row_id, data):
    return patch_rows(table_name, {"id": f"eq.{row_id}"}, data)


# ==========================================
# 4. 편의 기능 함수들
# ==========================================
def insert_request(text, user_id, device_id):
    data = {
        "phrase_text": text,
        "user_id": user_id,
        "device_id": device_id,
        "status": "pending",
        "session_id": datetime.datetime.now().strftime("S%Y%m%d_%H%M%S")
    }
    return insert_row("requests", data)

def fetch_report_data(table_name, limit=50):
    return fetch_rows(table_name, limit=limit)

def insert_body_data(height, weight, muscle_mass, body_fat):
    data = {
        "height": height,
        "weight": weight,
        "muscle_mass": muscle_mass,
        "body_fat": body_fat
    }
    return insert_row("body_data", data)

def get_latest_body_data():
    rows = fetch_rows("body_data", limit=1)
    if rows:
        return rows[0]
    return None