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

# [Solapi 라이브러리 체크 (긴급호출용)]
try:
    from solapi.services.message_service import SolapiMessageService
    from solapi.model.request.send_message_request import SendMessageRequest
    from solapi.model.request.message import Message
    HAS_SOLAPI = True
except ImportError:
    HAS_SOLAPI = False

# ==========================================
# [1] 기본 설정 및 경로
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(CURRENT_DIR, 'backend')
WEB_DIR = os.path.join(CURRENT_DIR, 'web') 
if not os.path.exists(WEB_DIR): os.makedirs(WEB_DIR)

sys.path.append(BACKEND_DIR)

# .env 로드 (web 폴더 우선)
env_path = os.path.join(WEB_DIR, '.env')
if not os.path.exists(env_path): env_path = os.path.join(CURRENT_DIR, '.env')
load_dotenv(env_path)

# [중요] 백엔드 모듈 연결
try:
    from supabase_rest import insert_row, fetch_rows, insert_request, insert_body_log, insert_exercise_log, fetch_report_data
    print("✅ Supabase AI 모듈 로드 성공!")
except ImportError:
    print("⚠️ backend/supabase_rest.py를 찾을 수 없습니다. 챗봇/DB 기능이 제한됩니다.")
    insert_request = None; fetch_rows = None; insert_body_log = None; insert_exercise_log = None; fetch_report_data = None

# 그래프 폰트 설정 (한글 깨짐 방지: 라즈베리파이에서는 NanumGothic 추천)
try:
    # 윈도우용
    plt.rcParams['font.family'] = 'Malgun Gothic' 
except:
    # 리눅스용 (나눔고딕 설치되어 있다면)
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# ==========================================
# [2] 유틸리티
# ==========================================
def remove_emojis(text):
    return re.sub(r'[^ ㄱ-ㅣ가-힣a-zA-Z0-9\.,\?!]+', '', text)

def draw_text_korean(img_bgr, text, x, y, size=32, color=(255, 255, 255), max_width=0):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try: font = ImageFont.truetype("fonts/malgun.ttf", size) 
    except: font = ImageFont.load_default()
    
    lines = textwrap.wrap(text, width=int(max_width/(size*0.8))) if max_width > 0 else [text]
    curr_y = y
    for line in lines:
        draw.text((x, curr_y), line, font=font, fill=color)
        curr_y += int(size * 1.3)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 자소 조합 데이터
CONSONANTS_19 = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
VOWELS_21 = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
ARROWS3 = ["↑", "←", "→"]
CHO_INDEX = {c: i for i, c in enumerate(CONSONANTS_19)}
JUNG_INDEX = {v: i for i, v in enumerate(VOWELS_21)}
JONG_LIST = [""] + ["ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ", "ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
JONG_INDEX = {c: i for i, c in enumerate(JONG_LIST)}
SUFFIX27 = [a+b+c for a in ARROWS3 for b in ARROWS3 for c in ARROWS3]
SUFFIX_TO_INDEX = {s: i for i, s in enumerate(SUFFIX27)}
MODE_MAP = {"↑": "CONS", "←": "VOW", "→": "CMD"}
CMD_MAP = {0: "전송", 1: "띄어쓰기", 2: "지우기", 3: "치트"}

def decode_token(seq4):
    if len(seq4) != 4: return None
    mode = MODE_MAP.get(seq4[0])
    idx = SUFFIX_TO_INDEX.get(seq4[1:]) 
    if mode is None or idx is None: return None
    if mode == "CONS": return CONSONANTS_19[idx] if idx < len(CONSONANTS_19) else None
    if mode == "VOW": return VOWELS_21[idx] if idx < len(VOWELS_21) else None
    if mode == "CMD": return CMD_MAP.get(idx, None)
    return None

def compose_hangul(jamos):
    out = []
    i = 0
    n = len(jamos)
    def is_con(x): return x in CHO_INDEX
    def is_vow(x): return x in JUNG_INDEX
    while i < n:
        t = jamos[i]
        if t == " ": out.append(" "); i+=1; continue
        if is_vow(t): 
            L,V,T="ㅇ",t,""; adv=1
            if i+1<n and is_con(jamos[i+1]) and not (i+2<n and is_vow(jamos[i+2])):
                if jamos[i+1] in JONG_INDEX: T=jamos[i+1]; adv=2
            code = 0xAC00 + (CHO_INDEX[L]*21+JUNG_INDEX[V])*28 + JONG_INDEX.get(T,0)
            out.append(chr(code)); i+=adv; continue
        if is_con(t):
            if i+1<n and is_vow(jamos[i+1]):
                L,V,T=t,jamos[i+1],""; adv=2
                if i+2<n and is_con(jamos[i+2]) and not (i+3<n and is_vow(jamos[i+3])):
                    if jamos[i+2] in JONG_INDEX: T=jamos[i+2]; adv=3
                code = 0xAC00 + (CHO_INDEX[L]*21+JUNG_INDEX[V])*28 + JONG_INDEX.get(T,0)
                out.append(chr(code)); i+=adv; continue
        out.append(t); i+=1
    return "".join(out)

# ==========================================
# [3] MediaPipe & Threads
# ==========================================
mp_face_mesh = mp.solutions.face_mesh

def _pt(lm, W, H): return (lm.x * W, lm.y * H)
def iris_center(lms, idxs, W, H):
    pts = np.array([_pt(lms[i], W, H) for i in idxs], dtype=np.float32)
    return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
def norm_iris_pos(lms, W, H):
    L_IRIS = [468, 469, 470, 471, 472]
    R_IRIS = [473, 474, 475, 476, 477]
    lcL=_pt(lms[33],W,H); lcR=_pt(lms[133],W,H)
    lt=_pt(lms[159],W,H); lb=_pt(lms[145],W,H); li=iris_center(lms,L_IRIS,W,H)
    rcL=_pt(lms[362],W,H); rcR=_pt(lms[263],W,H)
    rt=_pt(lms[386],W,H); rb=_pt(lms[374],W,H); ri=iris_center(lms,R_IRIS,W,H)
    def safe_norm(iris, cL, cR, top, bot):
        denx, deny = (cR[0]-cL[0]), (bot[1]-top[1])
        if abs(denx)<1e-6 or abs(deny)<1e-6: return None
        return ((iris[0]-cL[0])/denx, (iris[1]-top[1])/deny)
    ln = safe_norm(li, lcL, lcR, lt, lb)
    rn = safe_norm(ri, rcL, rcR, rt, rb)
    if ln is None or rn is None: return None
    return ((ln[0]+rn[0])/2.0, (ln[1]+rn[1])/2.0)

class EyeMouseThread(QThread):
    gaze_signal = pyqtSignal(float, float)
    mouth_signal = pyqtSignal(bool)

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.running=True; self.screen_w=screen_w; self.screen_h=screen_h
        self.calibrated=False; self.cx_base=0.5; self.cy_base=0.5
        self.sensitivity_x=10.0; self.sensitivity_y=22.0
        self.last_gaze_time=0
        self.history_x = deque(maxlen=7); self.history_y = deque(maxlen=7)

    def run(self):
        # [수정] 라즈베리파이 카메라 호환성 (V4L2)
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # 라즈베리파이에서 안되면 주석 해제

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            cal_x, cal_y, start_t = [], [], time.time()
            while self.running:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                H, W, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    pos = norm_iris_pos(lms, W, H)
                    if pos:
                        if not self.calibrated:
                            cal_x.append(pos[0]); cal_y.append(pos[1])
                            if time.time()-start_t > 3.0:
                                self.cx_base=np.mean(cal_x); self.cy_base=np.mean(cal_y); self.calibrated=True
                        elif time.time()-self.last_gaze_time >= 0.05:
                            dx, dy = pos[0]-self.cx_base, pos[1]-self.cy_base
                            tx = (self.screen_w/2) + (dx*self.sensitivity_x*self.screen_w)
                            ty = (self.screen_h/2) + (dy*self.sensitivity_y*self.screen_h)
                            self.history_x.append(tx); self.history_y.append(ty)
                            smooth_x = sum(self.history_x) / len(self.history_x)
                            smooth_y = sum(self.history_y) / len(self.history_y)
                            self.gaze_signal.emit(smooth_x, smooth_y)
                            self.last_gaze_time=time.time()
                    
                    # [수정] 입 벌림 감도 0.25 (인식 잘 되게 완화)
                    up = lms[13]; down = lms[14]
                    left = lms[61]; right = lms[291]
                    v_dist = np.linalg.norm([up.x - down.x, up.y - down.y])
                    h_dist = np.linalg.norm([left.x - right.x, left.y - right.y])
                    
                    if h_dist > 0:
                        mar = v_dist / h_dist
                        if mar > 0.25: self.mouth_signal.emit(True) 
                        else: self.mouth_signal.emit(False)
                    else:
                        if v_dist > 15: self.mouth_signal.emit(True)
                        else: self.mouth_signal.emit(False)

                time.sleep(0.01)
        cap.release()
    def stop(self): self.running=False; self.wait()

class ChatbotThread(QThread):
    frame_signal = pyqtSignal(QImage)
    def __init__(self, cx_base, cy_base):
        super().__init__()
        self.running = True; self.cx_base = cx_base; self.cy_base = cy_base
        self.user_question = ""; self.ai_answer = ""
        self.th_x = 0.05; self.th_y = 0.06
        self.x_ema = None; self.y_ema = None; self.alpha = 0.25
        self.candidate_dir = "NEUTRAL"; self.candidate_since = 0; self.cooldown_until = 0; self.HOLD_SEC = 0.45
        self.seq = ""; self.raw_jamos = []
        
        # DB 통신 변수
        self.req_id = None
        self.is_waiting_for_response = False
        self.last_sent_text = ""

    def classify_dir(self, x, y):
        dx, dy = x - self.cx_base, y - self.cy_base
        if dy < -self.th_y: return "↑"
        if dx < -self.th_x: return "←"
        if dx > self.th_x:  return "→"
        return "NEUTRAL"

    def run(self):
        # [수정] 라즈베리파이 카메라 호환성
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            while self.running:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                H, W, _ = frame.shape
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                now = time.time(); curr_dir = "NEUTRAL"
                
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    pos = norm_iris_pos(lms, W, H)
                    if pos:
                        if self.x_ema is None: self.x_ema, self.y_ema = pos
                        else: self.x_ema = (1-self.alpha)*self.x_ema + self.alpha*pos[0]; self.y_ema = (1-self.alpha)*self.y_ema + self.alpha*pos[1]
                        curr_dir = self.classify_dir(self.x_ema, self.y_ema)
                
                if now >= self.cooldown_until:
                    if curr_dir != self.candidate_dir: self.candidate_dir = curr_dir; self.candidate_since = now
                    elif curr_dir != "NEUTRAL":
                        if (now - self.candidate_since) >= self.HOLD_SEC:
                            final_action = curr_dir; self.cooldown_until = now + 0.6; self.candidate_dir = "NEUTRAL"
                            self.seq += final_action
                            if len(self.seq) == 4:
                                token = decode_token(self.seq); self.seq = ""
                                if token == "전송": 
                                    text = compose_hangul(self.raw_jamos)
                                    if text:
                                        self.user_question = text
                                        self.raw_jamos = [] 
                                        self.send_to_ai_db(text) # DB 전송
                                    
                                elif token == "띄어쓰기": self.raw_jamos.append(" ")
                                elif token == "지우기": 
                                    if self.raw_jamos: self.raw_jamos.pop()
                                elif token and token != "치트": self.raw_jamos.append(token)
                
                # [화면 그리기] 검은 배경
                display_img = np.zeros((H, W, 3), dtype=np.uint8)
                pil_img = Image.fromarray(display_img)
                draw = ImageDraw.Draw(pil_img)
                try: font = ImageFont.truetype("fonts/malgun.ttf", 30)
                except: font = ImageFont.load_default()
                
                arrow_text = curr_dir if curr_dir != "NEUTRAL" else "중앙"
                draw.text((50, 30), f"눈동자: {arrow_text}", font=font, fill=(0, 255, 255))
                
                if curr_dir != "NEUTRAL" and now >= self.cooldown_until:
                    progress = min(1.0, (now - self.candidate_since) / self.HOLD_SEC)
                    bar_w = int(300 * progress)
                    draw.rectangle([(50, 80), (50 + bar_w, 100)], fill=(0, 255, 0))
                
                draw.text((50, 130), f"입력 대기: {self.seq}", font=font, fill=(255, 255, 0))
                final_text = compose_hangul(self.raw_jamos)
                draw.text((50, 180), f"현재 입력값: {final_text}", font=font, fill=(255, 255, 255))
                draw.text((50, 250), f"나: {self.user_question}", font=font, fill=(200, 200, 255))
                draw.text((50, 320), "AI:", font=font, fill=(100, 255, 100))
                
                lines = textwrap.wrap(self.ai_answer, width=30)
                y_text = 360
                for line in lines:
                    draw.text((80, y_text), line, font=font, fill=(255, 255, 255))
                    y_text += 40

                final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qt_img = QImage(frame_rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.frame_signal.emit(qt_img)
                
                # 응답 대기 확인
                if self.is_waiting_for_response:
                    self.check_ai_response()
                    
                time.sleep(0.03)
        cap.release()

    def send_to_ai_db(self, text):
        self.ai_answer = "답변 생성 중... (DB 전송)"
        self.last_sent_text = text
        self.is_waiting_for_response = True
        
        if insert_request:
            try:
                # DB에 질문 등록 (ID 받아오기 시도)
                res = insert_request(text, "default_user", "mirror_device")
                found_id = None
                if res and isinstance(res, list) and len(res) > 0:
                    found_id = res[0].get('id')
                
                self.req_id = found_id
                print(f">> [DB] 질문 전송 완료. ID: {found_id}")
            except Exception as e:
                print(f">> [DB] 전송 오류: {e}")
                self.ai_answer = "오류: DB 연결 실패"
                self.is_waiting_for_response = False
        else:
            self.ai_answer = "데모: 백엔드 모듈 없음"
            self.is_waiting_for_response = False

    def check_ai_response(self):
        if not fetch_rows: return
        try:
            rows = []
            if self.req_id: 
                rows = fetch_rows("requests", "status, result_text", {"id": f"eq.{self.req_id}"})
            else: 
                # ID를 못 받았으면 텍스트로 조회
                rows = fetch_rows("requests", "status, result_text", {"phrase_text": f"eq.{self.last_sent_text}"})
            
            if rows:
                for row in rows:
                    if row.get('status') == 'done' and row.get('result_text'):
                        self.ai_answer = row.get('result_text')
                        self.is_waiting_for_response = False
                        self.req_id = None
                        print(f">> [DB] 답변 수신 완료: {self.ai_answer[:20]}...")
                        break
        except: pass

    def stop(self): self.running = False; self.wait()


# ==========================================
# [4] 메인 윈도우 클래스
# ==========================================
class SmartMirror(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.gaze_offset_x = 0; self.gaze_offset_y = -120
        self.hovered_btn = None; self.dwell_start_time = 0; self.DWELL_THRESHOLD = 1.5
        self.original_styles = {}

        ui_path = os.path.join(CURRENT_DIR, "mirror_design.ui")
        if not os.path.exists(ui_path): sys.exit()
        loadUi(ui_path, self)
        self.setWindowTitle("Smart Mirror Final")

        # 체크박스 스타일 (초록색 체크)
        self.setStyleSheet("""
            QCheckBox::indicator:checked {
                background-color: #00FF00;
                border: 2px solid white;
            }
        """)

        self.save_all_button_styles()

        self.csv_path = os.path.join(WEB_DIR, "body_data.csv")
        self.ex_csv_path = os.path.join(WEB_DIR, "exercise_data.csv")
        self.init_csv_files()
        
        self.last_body_change_time_str = "" 
        self.body_timer_seconds = 0; self.BODY_LIMIT_SECONDS = 7200 
        self.daily_body_count = 0; self.daily_breath_count = 0
        self.exercise_state = ""; self.exercise_cycle = 0; self.exercise_timer_count = 0
        self.sos_timer_count = 5; self.is_mouth_open = False; self.mouth_open_frames = 0; self.total_frames_in_step = 0

        self.body_checkboxes = []
        for i in range(1, 9):
            chk = getattr(self, f"chk_body_{i}", None)
            if chk: self.body_checkboxes.append(chk)
        self.body_part_names = ["머리", "왼쪽 어깨", "오른쪽 어깨", "등", "엉덩이", "왼쪽 팔꿈치", "오른쪽 팔꿈치", "발뒤꿈치"]

        if hasattr(self, 'layout_report_graph'):
            self.fig = plt.Figure(figsize=(5, 4), dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.layout_report_graph.addWidget(self.canvas)
            self.ax = self.fig.add_subplot(111)
            self.fig.patch.set_facecolor('none'); self.ax.set_facecolor('none')
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white') 
            self.ax.spines['right'].set_color('white')
            self.ax.spines['left'].set_color('white')

        style = "QTableWidget { background: transparent; color: white; border: 2px solid white; } QHeaderView::section { background: #444; color: white; font-weight: bold; }"
        if hasattr(self, 'table_report_log'): self.table_report_log.setStyleSheet(style)
        if hasattr(self, 'table_mouth_report'): self.table_mouth_report.setStyleSheet(style)

        def safe_con(name, func):
            if hasattr(self, name): getattr(self, name).clicked.connect(func)

        safe_con("btn_wake_up", lambda: self.go_to_page(1))            
        safe_con("btn_menu", lambda: self.go_to_page(2))                
        safe_con("btn_back_home_1", lambda: self.go_to_page(1))        
        safe_con("btn_back_home_2", self.stop_exercise)
        safe_con("btn_back_home_3", lambda: self.go_to_page(1))        
        safe_con("btn_back_home_4", self.stop_chatbot)                  
        safe_con("btn_back_home_guardian", lambda: self.go_to_page(1)) 
        
        safe_con("btn_bedsores", self.reset_body_timer)
        safe_con("btn_start_body_check", self.start_body_check)        
        safe_con("btn_save_body_data", self.save_body_data)            
        safe_con("btn_start_mouth", self.start_mouth_exercise)          
        safe_con("btn_start_breath", self.start_breath_exercise)        
        safe_con("btn_start_chatbot", self.start_chatbot)                
        safe_con("btn_emergency_call", self.start_sos_sequence)        
        safe_con("btn_sos_cancel", self.cancel_sos)                    
        safe_con("btn_guardian_menu", lambda: self.go_to_page(7))        
        safe_con("btn_view_report", self.show_report_page)
        safe_con("btn_back_report", lambda: self.go_to_page(7))        
        safe_con("btn_go_mouth_report", self.show_mouth_report_page)
        safe_con("btn_back_mouth_report", lambda: self.go_to_page(7))

        self.timer_exercise = QTimer(self); self.timer_exercise.timeout.connect(self.run_exercise_logic)
        self.timer_sos = QTimer(self); self.timer_sos.timeout.connect(self.update_sos_countdown)
        self.timer_body = QTimer(self); self.timer_body.timeout.connect(self.update_body_timer)
        self.timer_body.start(1000)

        self.eye_thread = EyeMouseThread(1024, 600)
        self.eye_thread.gaze_signal.connect(self.update_gaze_point)
        self.eye_thread.mouth_signal.connect(self.set_mouth_status) 
        self.eye_thread.start()
        
        self.cursor_label = QLabel(self)
        self.cursor_label.resize(20, 20)
        self.cursor_label.setStyleSheet("background-color: red; border-radius: 10px; border: 2px solid white;")
        self.cursor_label.hide()
        
        self.chatbot_thread = None
        self.stackedWidget.setCurrentIndex(0) 
        QTimer.singleShot(1000, lambda: QMessageBox.information(self, "안내", "3초간 화면 중앙을 응시하여 보정하세요."))

    def save_all_button_styles(self):
        all_buttons = self.findChildren(QPushButton)
        for btn in all_buttons:
            key = btn.objectName() if btn.objectName() else str(id(btn))
            self.original_styles[key] = btn.styleSheet()

    def update_body_timer(self):
        self.body_timer_seconds += 1
        h = self.body_timer_seconds // 3600
        m = (self.body_timer_seconds % 3600) // 60
        s = self.body_timer_seconds % 60
        time_str = f"{h:02d}:{m:02d}:{s:02d}"
        
        if hasattr(self, 'btn_bedsores'):
            if self.last_body_change_time_str:
                self.btn_bedsores.setText(f"체위 변경 {self.last_body_change_time_str}\n{time_str} 경과")
            else:
                self.btn_bedsores.setText(f"체위 변경\n{time_str} 경과")
            
            if self.body_timer_seconds > self.BODY_LIMIT_SECONDS:
                self.btn_bedsores.setStyleSheet("background-color: red; color: white; font-size: 27px; border-radius: 10px; font-weight: bold;")
        
        now = QTime.currentTime()
        if now.hour() == 0 and now.minute() == 0 and now.second() == 0:
            self.reset_daily_stats()

    def reset_body_timer(self):
        self.body_timer_seconds = 0
        current_time_str = QTime.currentTime().toString("HH:mm:ss")
        self.last_body_change_time_str = current_time_str
        
        if hasattr(self, 'btn_bedsores'):
            key = "btn_bedsores"
            if key in self.original_styles:
                self.btn_bedsores.setStyleSheet(self.original_styles[key])
            else:
                self.btn_bedsores.setStyleSheet("background-color: #333; color: white; font-size: 27px; border-radius: 10px;")
            self.btn_bedsores.setText(f"체위 변경 {self.last_body_change_time_str}\n00:00:00 경과")
        
        self.daily_body_count += 1
        if hasattr(self, 'lbl_check_bed'):
            self.lbl_check_bed.setText(f"체위 변경 ({self.daily_body_count}/10)")
        
        if self.daily_body_count >= 10:
            if hasattr(self, 'chk_bed'): self.chk_bed.setChecked(True)

    def reset_daily_stats(self):
        self.daily_body_count = 0
        self.daily_breath_count = 0
        self.last_body_change_time_str = "" 
        
        if hasattr(self, 'lbl_check_bed'): self.lbl_check_bed.setText("체위 변경 (0/10)")
        if hasattr(self, 'lbl_check_breath'): self.lbl_check_breath.setText("호흡 운동 (0/2)")
        if hasattr(self, 'chk_bed'): self.chk_bed.setChecked(False)
        if hasattr(self, 'chk_breath'): self.chk_breath.setChecked(False)
        if hasattr(self, 'chk_mouth'): self.chk_mouth.setChecked(False)

    def go_to_page(self, idx): self.stackedWidget.setCurrentIndex(idx)
    
    def update_gaze_point(self, x, y):
        if self.stackedWidget.currentIndex() == 5: return 
        final_x = x + self.gaze_offset_x
        final_y = y + self.gaze_offset_y
        self.cursor_label.move(int(final_x), int(final_y)); self.cursor_label.show()
        self.process_dwell_click(final_x, final_y)

    def process_dwell_click(self, x, y):
        current_page = self.stackedWidget.currentWidget()
        found_btn = None
        for widget in current_page.findChildren(QPushButton):
            if widget.isVisible() and widget.geometry().contains(int(x), int(y)):
                found_btn = widget; break
        
        if found_btn:
            if self.hovered_btn != found_btn:
                self.reset_button_style()
                self.hovered_btn = found_btn
                self.dwell_start_time = time.time()
                orig = found_btn.styleSheet()
                found_btn.setStyleSheet(orig + "; border: 5px solid #00FF00;")
            else:
                if (time.time() - self.dwell_start_time) >= self.DWELL_THRESHOLD:
                    found_btn.animateClick()
                    self.reset_button_style()
                    self.hovered_btn = None
                    self.dwell_start_time = 0
        else:
            if self.hovered_btn:
                self.reset_button_style()
                self.hovered_btn = None
                self.dwell_start_time = 0

    def reset_button_style(self):
        if self.hovered_btn:
            key = self.hovered_btn.objectName() if self.hovered_btn.objectName() else str(id(self.hovered_btn))
            if key == "btn_bedsores" and self.body_timer_seconds > self.BODY_LIMIT_SECONDS:
                 self.hovered_btn.setStyleSheet("background-color: red; color: white; font-size: 27px; border-radius: 10px; font-weight: bold;")
            elif key in self.original_styles:
                self.hovered_btn.setStyleSheet(self.original_styles[key])
            else:
                self.hovered_btn.setStyleSheet("") 

    def set_mouth_status(self, is_open): self.is_mouth_open = is_open

    def init_csv_files(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding='utf-8-sig') as f:
                csv.writer(f).writerow(["Date", "Time", "Parts", "Percent"])
        if not os.path.exists(self.ex_csv_path):
            with open(self.ex_csv_path, "w", newline="", encoding='utf-8-sig') as f:
                csv.writer(f).writerow(["Date", "Time", "Type", "Result", "Count", "SuccessRate"])

    def save_exercise_data(self, ex_type, result, count, success_rate=0.0):
        now = datetime.datetime.now()
        try:
            with open(self.ex_csv_path, "a", newline="", encoding='utf-8-sig') as f:
                csv.writer(f).writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), ex_type, result, count, f"{success_rate:.1f}"])
            print(f"운동 저장 완료: {ex_type}, {result}, {success_rate}%")
        except: pass
        
        # DB 저장 (에러나도 프로그램 안 멈춤)
        if insert_exercise_log:
            try:
                insert_exercise_log(ex_type, result, count)
            except: pass

    # 🔴 [수정됨] 욕창 그래프 표시 (꺾은선 그래프)
    def show_report_page(self):
        self.go_to_page(8) # Page 9 (Index 8)
        
        rows = []
        if fetch_report_data:
            try: rows = fetch_report_data("body_check_logs", limit=20)
            except: pass
        
        # DB 실패 시 CSV 백업 사용
        if not rows and os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                df_recent = df.tail(10).iloc[::-1]
                rows = df_recent.to_dict('records')
            except: pass

        # 1. 표(Table) 채우기
        if hasattr(self, 'table_report_log'):
            self.table_report_log.setColumnCount(3)
            self.table_report_log.setHorizontalHeaderLabels(["시간", "부위", "진행률(%)"])
            self.table_report_log.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table_report_log.setRowCount(len(rows))
            
            dates = []
            percents = []

            for i, row in enumerate(rows):
                if isinstance(row, dict): 
                    t_str = row.get('created_at', row.get('Date', '') + " " + row.get('Time', ''))[:16].replace('T', ' ')
                    parts = str(row.get('parts_text', row.get('Parts', '-')))
                    percent_val = float(row.get('percent', row.get('Percent', '0')))
                    
                    self.table_report_log.setItem(i, 0, QTableWidgetItem(t_str))
                    self.table_report_log.setItem(i, 1, QTableWidgetItem(parts))
                    self.table_report_log.setItem(i, 2, QTableWidgetItem(f"{percent_val}%"))
                    
                    # 그래프용 데이터 수집
                    try:
                        dt = datetime.datetime.strptime(t_str, "%Y-%m-%d %H:%M")
                        dates.append(dt)
                        percents.append(percent_val)
                    except: pass
                else: continue 

            # 2. 꺾은선 그래프 그리기 (Line Plot)
            self.ax.clear()
            if dates:
                # 시간 순 정렬
                sorted_pairs = sorted(zip(dates, percents))
                s_dates, s_percents = zip(*sorted_pairs)
                
                # 날짜 포맷 간단히 (월-일 시:분)
                date_labels = [d.strftime("%m-%d %H:%M") for d in s_dates]
                
                self.ax.plot(date_labels, s_percents, marker='o', linestyle='-', color='lime', linewidth=2)
                self.ax.set_title("체위 변경 진행률 변화", fontsize=12, color='white')
                self.ax.set_ylabel("진행률 (%)", color='white')
                self.ax.grid(True, linestyle='--', alpha=0.3)
                
                # X축 라벨 겹침 방지
                plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right")
                self.fig.tight_layout()
            else:
                self.ax.text(0.5, 0.5, "데이터 없음", ha='center', va='center', color='white')
            
            self.canvas.draw()

    # 🔴 [추가됨] 구강 운동 리포트 (표 + 골든타임 분석)
    def show_mouth_report_page(self):
        self.go_to_page(9) # Page 10 (Index 9)
        
        # 데이터 가져오기 (DB -> CSV)
        rows = []
        if fetch_report_data:
            try: rows = fetch_report_data("exercise_logs", limit=30)
            except: pass
        
        if not rows and os.path.exists(self.ex_csv_path):
            try:
                df = pd.read_csv(self.ex_csv_path)
                df_recent = df.tail(20).iloc[::-1]
                rows = df_recent.to_dict('records')
            except: pass

        # 1. 표 채우기
        if hasattr(self, 'table_mouth_report'):
            self.table_mouth_report.setColumnCount(4)
            self.table_mouth_report.setHorizontalHeaderLabels(["시간", "종류", "결과", "성공률"])
            self.table_mouth_report.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table_mouth_report.setRowCount(len(rows))
            
            time_success_map = {"오전": [], "오후": [], "저녁": [], "밤": []}

            for i, row in enumerate(rows):
                if isinstance(row, dict):
                    t_str = row.get('created_at', row.get('Date', '') + " " + row.get('Time', ''))[:16].replace('T', ' ')
                    ex_type = "입운동" if row.get('ex_type') == 'mouth' or row.get('Type') == 'mouth' else "호흡"
                    res = row.get('result', row.get('Result', '-'))
                    
                    # DB엔 success_rate 없을 수도 있어서 처리
                    rate_val = row.get('success_rate', row.get('SuccessRate', 0))
                    if rate_val is None: rate_val = 0
                    rate_str = f"{float(rate_val):.0f}%"

                    self.table_mouth_report.setItem(i, 0, QTableWidgetItem(t_str))
                    self.table_mouth_report.setItem(i, 1, QTableWidgetItem(ex_type))
                    self.table_mouth_report.setItem(i, 2, QTableWidgetItem(res))
                    self.table_mouth_report.setItem(i, 3, QTableWidgetItem(rate_str))
                    
                    # 골든 타임 분석용 데이터 수집
                    try:
                        dt = datetime.datetime.strptime(t_str, "%Y-%m-%d %H:%M")
                        hour = dt.hour
                        period = "밤"
                        if 6 <= hour < 12: period = "오전"
                        elif 12 <= hour < 18: period = "오후"
                        elif 18 <= hour < 24: period = "저녁"
                        
                        is_success = 1 if res == "Success" else 0
                        time_success_map[period].append(is_success)
                    except: pass

        # 2. 골든 타임 분석 및 표시
        best_period = "분석 중..."
        best_rate = 0
        
        for period, results in time_success_map.items():
            if len(results) > 0:
                rate = (sum(results) / len(results)) * 100
                if rate >= best_rate:
                    best_rate = rate
                    best_period = period
        
        msg = "아직 데이터가 충분하지 않습니다."
        if best_rate > 0:
            msg = f"💡 분석 결과: 환자분의 운동 골든 타임은 '{best_period}' 입니다. (성공률 {best_rate:.0f}%)"
        
        if hasattr(self, 'lbl_active_analysis'): 
            self.lbl_active_analysis.setText(msg)

    def start_body_check(self):
        self.go_to_page(3) 
        paths = [os.path.join(WEB_DIR, "body_new.png"), os.path.join(WEB_DIR, "body.png"),
                 os.path.join(CURRENT_DIR, "body_new.png"), os.path.join(CURRENT_DIR, "body.png")]
        found = False
        for p in paths:
            if os.path.exists(p):
                if hasattr(self, 'label'): self.label.setPixmap(QPixmap(p)); self.label.setScaledContents(True)
                elif hasattr(self, 'lbl_body_map'): self.lbl_body_map.setPixmap(QPixmap(p)); self.lbl_body_map.setScaledContents(True)
                found = True; break
        if not found and hasattr(self, 'label'): self.label.setText("이미지 파일 없음")
        for chk in self.body_checkboxes: chk.setChecked(False)

    def save_body_data(self):
        checked_names = [self.body_part_names[i] for i, chk in enumerate(self.body_checkboxes) if chk.isChecked()]
        parts_str = ", ".join(checked_names) if checked_names else "없음"
        percent = (len(checked_names) / 8) * 100
        now = datetime.datetime.now()
        
        try:
            with open(self.csv_path, "a", newline="", encoding='utf-8-sig') as f:
                csv.writer(f).writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), parts_str, percent])
        except: pass

        if insert_body_log:
            try:
                insert_body_log(parts_str, percent)
                print(">> DB: 욕창 기록 저장 완료")
            except Exception as e:
                print(f">> DB Error: {e}")

        QMessageBox.information(self, "저장 완료", f"저장되었습니다.\n진행률: {percent:.1f}%")
        self.go_to_page(1) 

    def start_mouth_exercise(self):
        self.exercise_state = "mouth"; self.exercise_cycle = 0; self.exercise_timer_count = -3
        self.mouth_open_frames = 0; self.total_frames_in_step = 0
        if hasattr(self, 'lbl_func_title'): self.lbl_func_title.setText("👄 구강 근육 운동")
        self.go_to_page(4) 
        if hasattr(self, 'lbl_guide_text'): self.lbl_guide_text.setText("운동 준비 중...")
        if hasattr(self, 'lbl_visual'): self.lbl_visual.setText("준비"); self.lbl_visual.setStyleSheet("background-color: black; color: white; font-size: 80px; border: 3px solid yellow;")
        self.timer_exercise.start(1000)

    def start_breath_exercise(self):
        self.exercise_state = "breath"; self.exercise_cycle = 0; self.exercise_timer_count = -3 
        if hasattr(self, 'lbl_func_title'): self.lbl_func_title.setText("🫁 호흡 운동")
        self.go_to_page(4)
        if hasattr(self, 'lbl_guide_text'): self.lbl_guide_text.setText("운동 준비 중...")
        if hasattr(self, 'lbl_visual'): self.lbl_visual.setText("준비"); self.lbl_visual.setStyleSheet("background-color: black; color: white; font-size: 80px; border: 3px solid cyan;")
        self.timer_exercise.start(1000)

    def run_exercise_logic(self):
        if self.exercise_timer_count < 0:
            count = abs(self.exercise_timer_count)
            if hasattr(self, 'lbl_visual'): self.lbl_visual.setText(str(count))
            self.exercise_timer_count += 1; return

        if self.exercise_timer_count == 0:
             if hasattr(self, 'lbl_visual'): self.lbl_visual.setText("START!"); self.lbl_visual.setStyleSheet("background-color: red; color: white;")
             self.exercise_timer_count += 1; return

        if self.exercise_state == "mouth":
            chars = ["아!", "에!", "이!", "오!", "우!"]
            real_count = self.exercise_timer_count - 1
            idx = (real_count // 2) % 5 
            if real_count > 0 and real_count % 10 == 0: self.exercise_cycle += 1
            if self.exercise_cycle >= 3: self.finish_exercise(); return
            if chars[idx] == "아!":
                self.total_frames_in_step += 1
                if self.is_mouth_open: self.mouth_open_frames += 1
            if hasattr(self, 'lbl_visual'): 
                self.lbl_visual.setText(chars[idx]); self.lbl_visual.setStyleSheet("background-color: rgba(0,0,0,200); color: yellow; font-size: 100px;")
            if hasattr(self, 'lbl_guide_text'): self.lbl_guide_text.setText(f"따라해보세요 ({self.exercise_cycle+1}세트)")
            self.exercise_timer_count += 1

        elif self.exercise_state == "breath":
            real_count = self.exercise_timer_count - 1
            cycle_t = real_count % 19 
            if cycle_t < 4: 
                if hasattr(self, 'lbl_visual'): self.lbl_visual.setStyleSheet("background-color: #3498db; border-radius: 100px;")
                if hasattr(self, 'lbl_guide_text'): self.lbl_guide_text.setText(f"🌿 들이마시세요... ({cycle_t+1}초)")
            elif cycle_t < 11: 
                if hasattr(self, 'lbl_visual'): self.lbl_visual.setStyleSheet("background-color: #f1c40f; border-radius: 100px;")
                if hasattr(self, 'lbl_guide_text'): self.lbl_guide_text.setText(f"✋ 참으세요! ({cycle_t-3}초)")
            elif cycle_t < 19: 
                if hasattr(self, 'lbl_visual'): self.lbl_visual.setStyleSheet("background-color: #2ecc71; border-radius: 100px;")
                if hasattr(self, 'lbl_guide_text'): self.lbl_guide_text.setText(f"💨 뱉으세요... ({cycle_t-10}초)")
            if cycle_t == 18:
                self.exercise_cycle += 1
                if self.exercise_cycle >= 2: self.finish_exercise(); return
            self.exercise_timer_count += 1

    def finish_exercise(self):
        self.timer_exercise.stop()
        msg = "완료"; res = "Success"; rate = 0.0
        
        if self.exercise_state == "mouth":
            if self.total_frames_in_step > 0: rate = (self.mouth_open_frames / self.total_frames_in_step) * 100
            if rate < 15: res = "Fail"
            else: res = "Success"
            msg = f"{res}! (정확도 {rate:.0f}%)"
            
            if res == "Success":
                if hasattr(self, 'chk_mouth'): 
                    self.chk_mouth.setChecked(True)
                    self.chk_mouth.repaint() 
                
        elif self.exercise_state == "breath":
            rate = 100.0; msg = "성공!"
            self.daily_breath_count += 1
            if hasattr(self, 'lbl_check_breath'): self.lbl_check_breath.setText(f"호흡 운동 ({self.daily_breath_count}/2)")
            if self.daily_breath_count >= 2:
                if hasattr(self, 'chk_breath'): self.chk_breath.setChecked(True)
            
        self.save_exercise_data(self.exercise_state, res, self.exercise_cycle, success_rate=rate)
        if hasattr(self, 'lbl_visual'): self.lbl_visual.setText("끝")
        if hasattr(self, 'lbl_guide_text'): self.lbl_guide_text.setText(f"운동 종료!\n{msg}\n데이터 저장됨.")
        QTimer.singleShot(3000, lambda: self.go_to_page(1))

    def stop_exercise(self):
        self.timer_exercise.stop()
        self.save_exercise_data(self.exercise_state, "Fail", self.exercise_cycle, success_rate=0.0)
        self.go_to_page(1)

    def start_chatbot(self):
        self.eye_thread.stop(); self.cursor_label.hide()
        self.chatbot_thread = ChatbotThread(self.eye_thread.cx_base, self.eye_thread.cy_base)
        self.chatbot_thread.frame_signal.connect(self.update_chatbot_ui)
        self.chatbot_thread.start()
        self.go_to_page(5) 
        
    def update_chatbot_ui(self, qt_img):
        if hasattr(self, 'lbl_chatbot_frame'): self.lbl_chatbot_frame.setPixmap(QPixmap.fromImage(qt_img))
    
    def stop_chatbot(self):
        if self.chatbot_thread: self.chatbot_thread.stop()
        self.eye_thread.start(); self.cursor_label.show(); 
        self.go_to_page(1) 

    def start_sos_sequence(self):
        self.sos_timer_count = 5
        if hasattr(self, 'lbl_sos_countdown'): self.lbl_sos_countdown.setText(f"긴급 전화를 연결합니다...\n{self.sos_timer_count}초 남음")
        self.go_to_page(6); self.timer_sos.start(1000)

    def update_sos_countdown(self):
        self.sos_timer_count -= 1
        if hasattr(self, 'lbl_sos_countdown'): self.lbl_sos_countdown.setText(f"긴급 전화를 연결합니다...\n{self.sos_timer_count}초 남음")
        if self.sos_timer_count <= 0:
            self.timer_sos.stop(); self.send_sos_message()
            QMessageBox.critical(self, "SOS 전송", "보호자에게 긴급 호출(전화)을 요청했습니다!")
            self.go_to_page(1)

    def cancel_sos(self):
        self.timer_sos.stop(); self.go_to_page(1)

    # 🔴 [성공한 단독 코드 로직 이식]
    def send_sos_message(self):
        print(">> [SOS] 1. 보호자 호출 프로세스 시작 (독립 실행 모드)")
        
        # 1. 여기서 라이브러리를 직접 import하여 전역 namespace 오염 방지
        try:
            from solapi.services.message_service import SolapiMessageService
            from solapi.model.request.send_message_request import SendMessageRequest
            from solapi.model.request.message import Message
            
            # 2. 인증키 설정 (성공 코드와 동일)
            api_key = 'NCSPGFN72DWVC9WE'
            api_secret = 'NRN6VWYLA4MAN0DLJBWZ2YV21XUQDTLP'
            
            # 3. 서비스 객체 새로 생성 (기존 self.message_service 무시)
            local_service = SolapiMessageService(api_key, api_secret)
            
            # 4. 메시지 생성
            msg = Message(
                to='01043994541',
                from_='01098113416', 
                type='VOICE',         
                text='긴급호출! 긴급호출! 환자가 보호자를 긴급호출 하였습니다.'
            )
            print(">> [SOS] 2. 메시지 객체 생성 완료")

            # 5. 전송 (성공 코드 방식)
            response = local_service.send(messages=[msg])
            
            print(f">> [SOS] 4. 발송 성공! 결과 ID: {response}")

        except Exception as e:
            print(f">> [SOS] 실패 발생: {e}")
            print(f">> [SOS] 상세 에러: {str(e)}")
            # 실패 시 사용자에게 알림
            QMessageBox.warning(self, "전송 실패", f"긴급 호출 실패: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SmartMirror()
    window.show()
    sys.exit(app.exec_())