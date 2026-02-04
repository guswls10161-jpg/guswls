import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv
import datetime

# ==========================================
# 1. .env 파일 위치 찾기
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
        print(f">> [DB] 연결 성공! (참조 파일: {env_path})")
    except Exception as e:
        print(f">> [DB] 연결 실패: {e}")
else:
    print(f">> [오류] .env 파일을 찾았으나 URL/KEY가 없습니다. ({env_path})")


# ==========================================
# 3. 핵심 DB 함수들 (여기가 수정됨!)
# ==========================================

def insert_row(table_name, data):
    """데이터 추가 함수"""
    if not supabase: return None
    try:
        # .select() 제거 버전 (버전 충돌 방지)
        response = supabase.table(table_name).insert(data).execute()
        return response.data 
    except Exception as e:
        print(f">> DB Insert Error ({table_name}): {e}")
        return None

def fetch_rows(table_name, select="*", match=None, filters=None, order=None, limit=None):
    """데이터 조회 함수 (eq. 접두사 자동 처리)"""
    if not supabase: return []
    try:
        query = supabase.table(table_name).select(select)
        
        # 1. match 처리
        if match:
            for k, v in match.items():
                if isinstance(v, str) and v.startswith("eq."):
                    query = query.eq(k, v.split("eq.")[1]) # "eq." 제거
                else:
                    query = query.eq(k, v)
        
        # 2. filters 처리 (main.py 호환용)
        if filters:
            for k, v in filters.items():
                if isinstance(v, str) and v.startswith("eq."):
                    query = query.eq(k, v.split("eq.")[1]) # "eq." 제거
                else:
                    query = query.eq(k, v)
        
        if order:
            col, direction = order.split(".")
            query = query.order(col, desc=(direction == "desc"))
        if limit:
            query = query.limit(limit)
            
        response = query.execute()
        return response.data
    except Exception as e:
        print(f">> DB Fetch Error ({table_name}): {e}")
        return []

def patch_rows(table_name, match, data):
    """데이터 수정 함수 (AI 답변 저장용 - eq. 접두사 자동 처리)"""
    if not supabase: return None
    try:
        query = supabase.table(table_name).update(data)
        
        # match 조건 처리 (여기서 eq.를 처리해야 AI가 답변을 저장함)
        for k, v in match.items():
            if isinstance(v, str) and v.startswith("eq."):
                query = query.eq(k, v.split("eq.")[1]) # "eq." 제거
            else:
                query = query.eq(k, v)
                
        response = query.execute()
        return response.data
    except Exception as e:
        print(f">> DB Update Error ({table_name}): {e}")
        return None

def update_by_id(table_name, row_id, data):
    # main.py/AI 코드가 'eq.'를 붙여서 보내도 patch_rows가 해결함
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
    return fetch_rows(table_name, order="created_at.desc", limit=limit)

def insert_body_log(parts_text, percent):
    data = {"parts_text": parts_text, "percent": float(percent)}
    return insert_row("body_check_logs", data)

def insert_exercise_log(ex_type, result, count):
    data = {"ex_type": ex_type, "result": result, "count": count}
    return insert_row("exercise_logs", data)