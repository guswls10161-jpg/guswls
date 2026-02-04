import os
import requests
import json
from dotenv import load_dotenv

# -------------------------------------------------------------------
# [경로 수정] web/.env 파일을 강제로 찾아갑니다.
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, 'web', '.env')

load_dotenv(dotenv_path=env_path)

MINIMAX_API_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

def ask_minimax(question):
    # 키가 없는 경우 미리 에러 방지
    if not MINIMAX_API_KEY:
        return "오류: .env 파일에 MINIMAX_API_KEY가 없습니다."

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "abab5.5-chat",
        "messages": [
            {
                "sender_type": "USER",
                "sender_name": "User",
                "text": question
            }
        ],
        "reply_constraints": {
            "sender_type": "BOT",
            "sender_name": "SmartMirror"
        },
        "bot_setting": [
            {
                "bot_name": "SmartMirror",
                "content": "You are a helpful smart mirror AI assistant. Answer briefly in Korean."
            }
        ]
    }

    try:
        response = requests.post(MINIMAX_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if 'reply' in result:
            return result['reply']
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['text']
            
        return "죄송해요, AI 응답 형식을 해석할 수 없어요."

    except Exception as e:
        print(f"❌ Minimax API Error: {e}")
        return f"에러 발생: {str(e)}"