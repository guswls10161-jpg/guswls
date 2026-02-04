import sys
import os
import csv
import datetime
import time
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import subprocess 
import re 
from collections import deque 

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import textwrap 

# [WSL 호환성 수정 1] Solapi 라이브러리 체크 (없어도 실행되게 처리)
try:
    from solapi.services.message_service import SolapiMessageService
    from solapi.model.request.send_message_request import SendMessageRequest
    from solapi.model.request.message import Message
    HAS_SOLAPI = True
except ImportError:
    HAS_SOLAPI = False
    print(">> [Warning] Solapi 라이브러리가 없습니다. 긴급 호출 기능이 제한됩니다.")

# [WSL 호환성 수정 2] 운영체제에 따른 폰트 설정 (한글 깨짐 방지)
if sys.platform.startswith('linux'):
    plt.rc('font', family='NanumGothic') # 리눅스(WSL)
    print(">> [System] 리눅스 모드: NanumGothic 폰트 적용")
else:
    plt.rc('font', family='Malgun Gothic') # 윈도우
    print(">> [System] 윈도우 모드: Malgun Gothic 폰트 적용")

plt.rcParams['axes.unicode_minus'] = False

# ==========================================\
# [1] 기본 설정 및 경로
# ==========================================\
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(CURRENT_DIR, 'backend')
WEB_DIR = os.path.join(CURRENT_DIR, 'web') 
if not os.path.exists(WEB_DIR): os.makedirs(WEB_DIR)

sys.path.append(BACKEND_DIR)

# backend 폴더의 모듈 import
try:
    from supabase_rest import insert_request, fetch_rows, insert_row, update_by_id
except ImportError:
    print(">> [Error] backend 모듈을 찾을 수 없습니다. 경로를 확인하세요.")

# ==========================================\
# [2] 메인 윈도우 클래스
# ==========================================\
class MirrorWindow(QMainWindow):
    def __init__(self):
        super(MirrorWindow, self).__init__()
        
        # UI 파일 로드
        ui_path = os.path.join(CURRENT_DIR, "mirror_design.ui")
        loadUi(ui_path, self)
        
        # 화면 설정
        self.setWindowTitle("Smart Mirror Interface")
        # WSL에서는 전체화면 시 창 이동이 불편할 수 있어 일반 모드로 시작 (필요시 showFullScreen 사용)
        self.show() 

        # 페이지 설정 (Stacked Widget)
        self.stackedWidget.setCurrentIndex(7)  # 초기 화면 (홈)

        # ----------------------------------
        # 버튼 이벤트 연결
        # ----------------------------------
        # 1. 홈 화면 버튼
        self.btn_start_mouth.clicked.connect(lambda: self.go_to_page(0))  # 구강운동
        self.btn_start_breath.clicked.connect(lambda: self.go_to_page(1)) # 호흡재활
        self.btn_start_body_check.clicked.connect(lambda: self.go_to_page(2)) # 욕창체크
        self.btn_start_chatbot.clicked.connect(self.start_chatbot_mode)   # 챗봇(눈동자)
        self.btn_guardian_menu.clicked.connect(lambda: self.go_to_page(6)) # 보호자 메뉴
        self.btn_emergency_call.clicked.connect(self.send_emergency_message) # 긴급호출

        # 2. 각 페이지의 '뒤로가기/홈으로' 버튼
        self.btn_back_home_1.clicked.connect(self.go_home)
        self.btn_back_home_2.clicked.connect(self.go_home)
        self.btn_back_home_3.clicked.connect(self.go_home)
        # (필요시 추가 버튼 연결)

        # ----------------------------------
        # 카메라 및 타이머 설정
        # ----------------------------------
        self.cap = cv2.VideoCapture(0) # USB 웹캠 연결
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # 30ms 마다 갱신

        # 그래프 초기화 (호흡 재활용)
        self.init_breath_graph()
        
        # MediaPipe 설정
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def go_to_page(self, index):
        self.stackedWidget.setCurrentIndex(index)

    def go_home(self):
        self.stackedWidget.setCurrentIndex(7)

    def init_breath_graph(self):
        # 호흡 그래프용 matplotlib 캔버스
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=80)
        self.canvas = FigureCanvas(self.fig)
        
        # UI에 있는 레이아웃에 추가 (예: layout_breath_graph)
        # self.layout_breath_graph.addWidget(self.canvas) 
        # (UI 파일에 해당 레이아웃이 있다고 가정)
        
        self.breath_data = deque(maxlen=50)
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_ylim(0, 100)
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # OpenCV (BGR) -> Qt (RGB) 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1) # 거울모드

        # 현재 페이지에 따른 로직 수행
        curr_idx = self.stackedWidget.currentIndex()
        
        if curr_idx == 0: # 구강 운동
            pass # 로직 추가
        elif curr_idx == 1: # 호흡 재활
            frame = self.process_breath(frame)
        elif curr_idx == 2: # 욕창 체크
            frame = self.process_body_check(frame)

        # 화면 출력
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 현재 페이지의 라벨에 이미지 표시 (예: label_cam_1, label_cam_2 ...)
        # 여기서는 편의상 모든 페이지의 카메라 라벨 이름이 label_camera라고 가정하거나
        # 페이지별로 찾아서 설정해야 함.
        # 예시:
        # if curr_idx == 0: self.label_cam_mouth.setPixmap(QPixmap.fromImage(qimg))
        
    def process_body_check(self, frame):
        # MediaPipe Pose 적용
        frame.flags.writeable = False
        results = self.pose.process(frame)
        frame.flags.writeable = True
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
        return frame

    def process_breath(self, frame):
        # 호흡 감지 로직 (그래프 업데이트 예시)
        # 실제로는 어깨 들썩임 등을 감지하여 데이터 추가
        new_val = np.random.randint(40, 60) # 더미 데이터
        self.breath_data.append(new_val)
        
        self.line.set_data(range(len(self.breath_data)), self.breath_data)
        self.ax.set_xlim(0, 50)
        self.canvas.draw()
        
        # 캔버스의 이미지를 프레임에 오버레이 하거나 별도 위젯으로 표시
        return frame

    def start_chatbot_mode(self):
        # 챗봇 모드 (별도 프로세스로 실행하거나 화면 전환)
        # 여기서는 gaze_arrow_input.py를 실행하는 방식 사용
        print(">> 챗봇(시선 추적) 모드 시작")
        try:
            # WSL 환경에서 python3 명령어로 실행
            subprocess.Popen(['python3', 'gaze_arrow_input.py'])
        except Exception as e:
            print(f">> 실행 실패: {e}")

    def send_emergency_message(self):
        print(">> [SOS] 보호자 호출 시도")
        
        if not HAS_SOLAPI:
            QMessageBox.warning(self, "오류", "Solapi 라이브러리가 설치되지 않았습니다.")
            return

        try:
            # 인증키 (보안상 환경변수 사용 권장)
            api_key = 'NCSPGFN72DWVC9WE'
            api_secret = 'NRN6VWYLA4MAN0DLJBWZ2YV21XUQDTLP'
            
            service = SolapiMessageService(api_key, api_secret)
            
            msg = Message(
                to='01043994541',
                from_='01098113416', 
                type='SMS', # 혹은 VOICE
                text='[긴급] 환자가 보호자를 호출하였습니다.'
            )
            
            response = service.send(messages=[msg])
            print(f">> 전송 결과: {response}")
            QMessageBox.information(self, "알림", "보호자에게 메시지를 전송했습니다.")
            
        except Exception as e:
            print(f">> 전송 실패: {e}")
            QMessageBox.critical(self, "실패", f"전송 중 오류가 발생했습니다.\n{e}")

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MirrorWindow()
    sys.exit(app.exec_())