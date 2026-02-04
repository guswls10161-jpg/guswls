#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import requests
from pathlib import Path

# --- 경로 설정 ---
# GazeTracking 폴더를 파이썬 라이브러리 경로에 강제 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAZE_DIR = PROJECT_ROOT / "GazeTracking"
sys.path.append(str(GAZE_DIR))

try:
    from gaze_tracking import GazeTracking
except ImportError:
    print("\n[ERROR] GazeTracking 라이브러리를 찾을 수 없습니다.")
    print(f"체크한 경로: {GAZE_DIR}")
    sys.exit(1)

def download_sample_image(save_path):
    """
    테스트용 얼굴 이미지(오바마 전 대통령) 다운로드
    *수정사항: User-Agent 헤더를 추가하여 브라우저인 척 위장함
    """
    url = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
    print(f"[INFO] 테스트 이미지 다운로드 중... \n -> {url}")
    
    # 봇 차단 방지용 헤더
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 404, 403 에러 체크

        # 파일 크기가 너무 작으면(1KB 미만) 이미지가 아닐 확률 높음
        if len(response.content) < 1000:
            print(f"[ERROR] 다운로드된 파일이 너무 작습니다. (크기: {len(response.content)} bytes)")
            print("이미지가 아니라 차단 페이지일 수 있습니다.")
            sys.exit(1)

        with open(save_path, 'wb') as handler:
            handler.write(response.content)
        print("[INFO] 다운로드 완료.")
        
    except Exception as e:
        print(f"[ERROR] 이미지 다운로드 실패: {e}")
        sys.exit(1)

def main():
    print("-" * 50)
    print("👀 GazeTracking Library Test (Static Image)")
    print("-" * 50)

    gaze = GazeTracking()
    img_filename = "test_face.jpg"
    img_path = PROJECT_ROOT / img_filename

    # 1. 이미지가 없으면 다운로드
    if not img_path.exists():
        download_sample_image(img_path)

    # 2. OpenCV로 이미지 읽기
    frame = cv2.imread(str(img_path))
    
    # 이미지 읽기 실패 시 체크
    if frame is None:
        print(f"\n[ERROR] 이미지를 읽을 수 없습니다: {img_path}")
        print("파일이 깨졌거나 jpg 형식이 아닐 수 있습니다.")
        print("해결법: 'rm test_face.jpg' 명령어로 파일을 지우고 다시 실행하세요.")
        return

    print("[INFO] 이미지 로드 성공. 분석 시작...")

    # 3. 눈동자 분석 수행
    try:
        gaze.refresh(frame)
        
        # 결과 텍스트 생성
        text = "알 수 없음"
        if gaze.is_blinking():
            text = "눈 감음 (Blinking)"
        elif gaze.is_right():
            text = "오른쪽 보는 중 (Right)"
        elif gaze.is_left():
            text = "왼쪽 보는 중 (Left)"
        elif gaze.is_center():
            text = "정면 보는 중 (Center)"

        # 좌표 가져오기
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        print("\n" + "="*30)
        print(f" 🎯 분석 결과: {text}")
        print(f" 👁️  왼쪽 눈 좌표 : {left_pupil}")
        print(f" 👁️  오른쪽 눈 좌표: {right_pupil}")
        print("="*30 + "\n")

        if left_pupil or right_pupil:
            print("✅ 테스트 성공! dlib과 opencv가 정상 작동합니다.")
        else:
            print("⚠️ 얼굴은 찾았으나 눈동자 좌표를 못 구했습니다. (이미지 각도 문제일 수 있음)")

    except Exception as e:
        print(f"\n[FATAL ERROR] 분석 도중 오류 발생:\n{e}")

if __name__ == "__main__":
    main()