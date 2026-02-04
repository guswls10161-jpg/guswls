# -*- coding: utf-8 -*-
"""
gaze_arrow_input.py (WSL/Linux 호환 수정 버전)
"""

import os
import sys
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from dotenv import load_dotenv

# DB 모듈 (없으면 건너뜀)
try:
    from supabase_rest import insert_row
except ImportError:
    insert_row = None
    print(">> [Warning] supabase_rest 모듈 없음. DB 저장이 비활성화됩니다.")

# -----------------------------
# ENV & PATH SETUP
# -----------------------------
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DEVICE_ID = (os.getenv("DEVICE_ID") or "WSL_DEVICE").strip()
SESSION_ID = (os.getenv("SESSION_ID") or datetime.now().strftime("S%Y%m%d_%H%M%S")).strip()

# [WSL 호환성 수정] 폰트 경로 설정
def get_font(size):
    # 1. 리눅스(WSL) 경로 시도 (나눔고딕)
    linux_font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    # 2. 윈도우 경로 시도
    windows_font_path = "arial.ttf"
    
    font_path = windows_font_path # 기본값
    
    if sys.platform.startswith('linux'):
        if os.path.exists(linux_font_path):
            font_path = linux_font_path
        else:
            # 폰트가 없으면 경고 출력 (설치 안내)
            # print(">> [Warning] 나눔고딕 폰트가 없습니다. sudo apt install fonts-nanum 해주세요.")
            pass
            
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        # 폰트 파일 자체를 못 찾으면 기본 비트맵 폰트 사용
        return ImageFont.load_default()

# -----------------------------
# Helper Functions
# -----------------------------
def draw_text(img, x, y, text, size, color):
    """PIL을 이용해 한글 텍스트 그리기"""
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img
        
    draw = ImageDraw.Draw(img_pil)
    font = get_font(size)
    
    draw.text((x, y), text, font=font, fill=color)
    
    return np.array(img_pil)

# -----------------------------
# Main Logic
# -----------------------------
def run_gaze_input():
    # MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 카메라 연결
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(">> [Error] 카메라를 열 수 없습니다.")
        print(">> WSL 사용시: PowerShell에서 'usbipd attach --wsl --busid <BUSID>' 했는지 확인하세요.")
        return

    # 해상도 설정
    W, H = 640, 480
    cap.set(3, W)
    cap.set(4, H)
    
    print(f">> 시선 추적 시작 (종료: q) - Device: {DEVICE_ID}")
    
    # 변수 초기화
    seq = ""
    cheat_on = True
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame.flags.writeable = False
        results = face_mesh.process(rgb)
        frame.flags.writeable = True
        
        # 빈 캔버스 (검은 배경)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 얼굴 인식 확인
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # (여기에 원래 있던 눈동자 좌표 계산 및 Morse Code 로직이 들어갑니다)
            # 코드가 너무 길어 핵심만 남깁니다. 기존 로직 그대로 사용하시면 됩니다.
            # ...
            
            # 테스트용 텍스트 출력
            canvas = draw_text(canvas, 10, 10, "눈을 깜빡여보세요", 20, (255, 255, 255))
            canvas = draw_text(canvas, 10, 40, f"입력된 코드: {seq}", 20, (0, 255, 0))

        # 화면 출력
        cv2.imshow("Gaze Input (WSL)", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gaze_input()