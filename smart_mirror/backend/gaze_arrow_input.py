# -*- coding: utf-8 -*-
"""
gaze_arrow_input.py (젯슨 나노 최적화 버전)

[개선 사항]
1. Kalman Filter 적용: 눈동자 떨림 보정 (OpenCV 내장 기능 사용 -> 가볍고 빠름)
2. Threading 적용: Supabase 전송 시 화면 멈춤(렉) 현상 제거
3. UI Caching: 고정된 도움말 패널을 미리 그려두어 CPU 사용량 절감
4. 충돌 방지: 카메라 락 및 예외 처리 강화
"""

import os
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from dotenv import load_dotenv

from supabase_rest import insert_row

# -----------------------------
# ENV (session/device)
# -----------------------------
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DEVICE_ID = (os.getenv("DEVICE_ID") or os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or "device").strip()
SESSION_ID = (os.getenv("SESSION_ID") or datetime.now().strftime("S%Y%m%d_%H%M%S")).strip()

# -----------------------------
# ✅ 1. 카메라 락 (충돌 방지)
# -----------------------------
def acquire_camera_lock():
    lock_dir = os.path.join(BACKEND_DIR, "logs")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, "camera.lock")
    f = open(lock_path, "a+")

    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        try: f.close()
        except: pass
        return None
    return f

def release_camera_lock(f):
    if not f: return
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except: pass
    try: f.close()
    except: pass

# -----------------------------
# ✅ 2. 칼만 필터 (Kalman Filter) 클래스
# -----------------------------
class GazeKalman:
    """OpenCV 내장 KalmanFilter를 사용한 저지연 떨림 보정"""
    def __init__(self):
        # 상태 변수 4개 (x, y, dx, dy), 측정 변수 2개 (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        # 노이즈 공분산 행렬 튜닝 (반응속도 vs 부드러움 조절)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.first_run = True

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if self.first_run:
            # 초기값 설정
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.first_run = False
        
        self.kf.correct(meas)
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def reset(self):
        self.first_run = True

# -----------------------------
# 설정 값 (튜닝)
# -----------------------------
HOLD_SEC = 0.42
CENTER_RELEASE_SEC = 0.22
SEQ_LEN = 4

# 칼만 필터가 있으므로 EMA Alpha는 조금 더 반응성 위주로 설정
DEADZONE_SCALE = 0.62
CONFIRM_RATIO = 1.15
COOLDOWN_SEC = 0.35

CAL_CENTER_SEC = 2.0
CAL_DIR_SEC = 1.8

ARM_TIMEOUT_SEC = 2.2
ARM_REQUIRED_REPEAT = 2

mp_face_mesh = mp.solutions.face_mesh
L_IRIS = [468, 469, 470, 471, 472]
R_IRIS = [473, 474, 475, 476, 477]

DIR2ARROW = {"UP": "↑", "LEFT": "←", "RIGHT": "→"}
ARROWS3 = ["↑", "←", "→"]

CONSONANTS_19 = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
VOWELS_21 = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
CHO_LIST = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
JUNG_LIST = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
JONG_LIST = [""] + ["ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

CHO_INDEX  = {c: i for i, c in enumerate(CHO_LIST)}
JUNG_INDEX = {v: i for i, v in enumerate(JUNG_LIST)}
JONG_INDEX = {c: i for i, c in enumerate(JONG_LIST)}

def compose_hangul(jamos: list[str]) -> str:
    out = []
    i = 0
    n = len(jamos)
    def is_con(x): return x in CHO_INDEX
    def is_vow(x): return x in JUNG_INDEX

    while i < n:
        t = jamos[i]
        if t == " ":
            out.append(" ")
            i += 1
            continue
        if is_vow(t):
            L, V, T = "ㅇ", t, ""
            adv = 1
            if i+1 < n and is_con(jamos[i+1]) and not (i+2 < n and is_vow(jamos[i+2])):
                if jamos[i+1] in JONG_INDEX:
                    T = jamos[i+1]
                    adv = 2
            code = 0xAC00 + (CHO_INDEX[L]*21 + JUNG_INDEX[V])*28 + (JONG_INDEX.get(T, 0))
            out.append(chr(code))
            i += adv
            continue
        if is_con(t):
            if i+1 < n and is_vow(jamos[i+1]):
                L, V, T = t, jamos[i+1], ""
                adv = 2
                if i+2 < n and is_con(jamos[i+2]) and not (i+3 < n and is_vow(jamos[i+3])):
                    if jamos[i+2] in JONG_INDEX:
                        T = jamos[i+2]
                        adv = 3
                code = 0xAC00 + (CHO_INDEX[L]*21 + JUNG_INDEX[V])*28 + (JONG_INDEX.get(T, 0))
                out.append(chr(code))
                i += adv
                continue
        out.append(t)
        i += 1
    return "".join(out)

def find_font_path():
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        r"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None
FONT_PATH = find_font_path()

def get_font(size: int):
    try: return ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default()
    except: return ImageFont.load_default()

def draw_text(img_bgr, x, y, text, size=24, color=(255,255,255)):
    if not text: return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    dr = ImageDraw.Draw(pil)
    dr.text((x, y), text, font=get_font(size), fill=(int(color[2]), int(color[1]), int(color[0])))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_box(img, x, y, w, h, color=(0,0,0), alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img

def _pt(lm, W, H): return (lm.x * W, lm.y * H)
def iris_center(lms, idxs, W, H):
    pts = np.array([_pt(lms[i], W, H) for i in idxs], dtype=np.float32)
    return (float(np.mean(pts[:,0])), float(np.mean(pts[:,1])))

def norm_iris_pos(lms, W, H):
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

def calibrate(cap, face_mesh):
    # 캘리브레이션은 정확도가 중요하므로 EMA/Kalman 없이 Raw 데이터 수집
    def collect(sec, guide_text):
        xs, ys = [], []
        t0 = time.time()
        while time.time() - t0 < sec:
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            
            canvas = frame.copy()
            canvas = draw_box(canvas, 0, 0, W, 120, (0,0,0), 0.75)
            canvas = draw_text(canvas, 18, 16, guide_text, 32, (255,255,255))
            remain = sec - (time.time()-t0)
            canvas = draw_text(canvas, 18, 52, f"남은 시간: {max(0, remain):.1f}s", 26, (255,220,180))
            
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                p = norm_iris_pos(lms, W, H)
                if p:
                    xs.append(p[0]); ys.append(p[1])
                    canvas = draw_text(canvas, 18, 86, f"iris x={p[0]:.3f} y={p[1]:.3f}", 24, (220,220,220))
            cv2.imshow("gaze_arrow_input", canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"): break
        return xs, ys

    cx, cy = collect(CAL_CENTER_SEC, "캘리브 1/4: 정면(가운데) 보기")
    lx, ly = collect(CAL_DIR_SEC,    "캘리브 2/4: 왼쪽 보기")
    rx, ry = collect(CAL_DIR_SEC,    "캘리브 3/4: 오른쪽 보기")
    uy, uy2= collect(CAL_DIR_SEC,    "캘리브 4/4: 위 보기")

    def mean_or(arr, default): return float(np.mean(arr)) if len(arr)>=8 else default
    cxx = mean_or(cx, 0.50); cyy = mean_or(cy, 0.50)
    lxx = mean_or(lx, cxx-0.12); rxx = mean_or(rx, cxx+0.12); uyy = mean_or(uy, cyy-0.10)
    
    span_x = max(0.08, min(rxx-cxx, cxx-lxx))
    span_y = max(0.06, cyy-uyy)
    th_x = float(np.clip(span_x * DEADZONE_SCALE, 0.040, 0.16))
    th_y = float(np.clip(span_y * DEADZONE_SCALE, 0.035, 0.14))
    return (cxx, cyy, th_x, th_y)

def classify_dir_3way(x, y, cx, cy, th_x, th_y):
    dx, dy = x - cx, y - cy
    if dy > th_y * 1.10: return "NEUTRAL" # 아래보기 무시
    if abs(dx) < th_x and abs(dy) < th_y: return "NEUTRAL"
    if dy < -th_y: return "UP"
    if dx > th_x: return "RIGHT"
    if dx < -th_x: return "LEFT"
    return "NEUTRAL"

def strength_3way(x, y, cx, cy, th_x, th_y, d):
    dx, dy = x - cx, y - cy
    if d == "UP": return abs(dy)/max(th_y, 1e-6)
    if d in ("LEFT", "RIGHT"): return abs(dx)/max(th_x, 1e-6)
    return 0.0

SUFFIX27 = [a+b+c for a in ARROWS3 for b in ARROWS3 for c in ARROWS3]
SUFFIX_TO_INDEX = {s: i for i, s in enumerate(SUFFIX27)}
MODE_ARROW_TO_MODE = {"↑": "CONS", "←": "VOW", "→": "CMD"}
CMD_MAP = {0:"SEND_ARM", 1:"SPACE", 2:"BACKSPACE_ARM", 3:"CHEAT", 4:"RECAL"}

def decode_token(seq4: str):
    if len(seq4)!=4: return None
    mode = MODE_ARROW_TO_MODE.get(seq4[0])
    idx = SUFFIX_TO_INDEX.get(seq4[1:])
    if mode is None or idx is None: return None
    if mode == "CONS": return CONSONANTS_19[idx] if idx < len(CONSONANTS_19) else "NOP"
    if mode == "VOW": return VOWELS_21[idx] if idx < len(VOWELS_21) else "NOP"
    if mode == "CMD": return CMD_MAP.get(idx, "NOP")
    return None

def mapping_lines():
    return [
        "3방향 입력 체계(↑/←/→ 전용)", "",
        "코드 길이: 4개 화살표",
        "  1번째(모드):  ↑=자음  ←=모음  →=기능",
        "  2~4번째(선택): ↑/←/→ 조합(27개) 중 하나", "",
        "※ 아래(↓) 시선은 입력에 사용하지 않음", "",
        "[기능] (→ + suffix3)",
        f"  SEND(전송)  = →{SUFFIX27[0]}",
        f"  SPACE(공백) = →{SUFFIX27[1]}",
        f"  삭제(BKSP)  = →{SUFFIX27[2]}",
        f"  패널ON/OFF  = →{SUFFIX27[3]}",
        f"  재캘리브    = →{SUFFIX27[4]}", "",
        "[자음(19)] (↑ + ...)", f"  ㄱ=↑{SUFFIX27[0]} ...", "",
        "[모음(21)] (← + ...)", f"  ㅏ=←{SUFFIX27[0]} ..."
    ]

# -----------------------------
# ✅ 3. 비동기 전송 (Threading)
# -----------------------------
def send_question_to_supabase_async(text: str, callback_success, callback_fail):
    text = (text or "").strip()
    if not text: return
    
    def _task():
        try:
            res = insert_row("requests", {
                "phrase_text": text, "status": "pending",
                "session_id": SESSION_ID, "device_id": DEVICE_ID
            })
            if callback_success: callback_success(text, res)
        except Exception as e:
            if callback_fail: callback_fail(e)

    # 데몬 쓰레드로 실행 (메인 종료 시 같이 종료)
    t = threading.Thread(target=_task, daemon=True)
    t.start()

# -----------------------------
# 메인 실행
# -----------------------------
def open_camera_any(max_index=5):
    for idx in range(max_index + 1):
        if os.name == "nt": cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else: cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) # 젯슨 V4L2 필수
        if cap.isOpened(): return cap, idx
        cap.release()
    return None, None

def run():
    lock_f = acquire_camera_lock()
    if lock_f is None:
        print("카메라 Lock 실패 (이미 실행 중?)")
        return

    cap, cam_idx = open_camera_any()
    if cap is None:
        print("카메라 열기 실패")
        return

    # 젯슨 나노 성능 고려 해상도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

    right_w, text_h, top_h = 760, 170, 220
    
    # 초기 캘리브레이션
    cx, cy, th_x, th_y = calibrate(cap, face_mesh)
    
    # ✅ Kalman Filter 초기화
    gaze_kf = GazeKalman()

    seq = ""
    jamo_stream = []
    popup, popup_until = "", 0.0
    cheat_on = True
    last_sent = ""
    
    allow_input = False
    neutral_since = None
    candidate_dir = "NEUTRAL"
    candidate_since = time.time()
    cooldown_until = 0.0

    armed_action, armed_until, armed_count = None, 0.0, 0

    # UI Caching
    cached_cheat_panel = None

    def set_popup(text, sec=0.9):
        nonlocal popup, popup_until
        popup = text; popup_until = time.time() + sec

    # --- 내부 로직 함수들 (ARM, Send 등) ---
    def arm(action_name):
        nonlocal armed_action, armed_until, armed_count
        armed_action = action_name; armed_until = time.time() + ARM_TIMEOUT_SEC
        armed_count = 1
        set_popup(f"{action_name} 준비(ARM) 1/{ARM_REQUIRED_REPEAT}", 1.0)

    def confirm_arm(action_name):
        nonlocal armed_action, armed_until, armed_count
        now = time.time()
        if armed_action != action_name or now > armed_until:
            arm(action_name); return False
        armed_count += 1
        if armed_count < ARM_REQUIRED_REPEAT:
            set_popup(f"{action_name} 준비 {armed_count}/{ARM_REQUIRED_REPEAT}", 1.0)
            armed_until = now + ARM_TIMEOUT_SEC
            return False
        armed_action = None; armed_until = 0.0; armed_count = 0
        return True

    def on_send_success(text, res):
        nonlocal last_sent, seq
        last_sent = text
        jamo_stream.clear(); seq = ""
        # 쓰레드에서 실행되므로 print만 (UI 팝업은 메인루프에서 상태값 보고 처리하면 더 좋지만 단순화함)
        print(f"[전송성공] {text}")
    
    def on_send_fail(e):
        print(f"[전송실패] {e}")

    def do_send():
        phrase = compose_hangul(jamo_stream).strip()
        if not phrase: set_popup("내용 없음", 1.0); return
        set_popup("전송 중...🚀", 1.5)
        # ✅ 비동기 전송 호출
        send_question_to_supabase_async(phrase, on_send_success, on_send_fail)

    def apply_token(tok):
        nonlocal cheat_on, seq, cx, cy, th_x, th_y, cached_cheat_panel
        if tok=="SPACE": jamo_stream.append(" "); set_popup("띄어쓰기", 0.7)
        elif tok=="CHEAT": cheat_on = not cheat_on; set_popup(f"패널 {cheat_on}", 0.8)
        elif tok=="RECAL":
            jamo_stream.clear(); seq=""
            cx, cy, th_x, th_y = calibrate(cap, face_mesh)
            gaze_kf.reset() # 칼만 필터도 리셋
            set_popup("재캘리브 완료", 0.9)
        elif tok=="SEND_ARM":
            if confirm_arm("전송"): do_send()
        elif tok=="BACKSPACE_ARM":
            if confirm_arm("삭제"): 
                if jamo_stream: jamo_stream.pop(); set_popup("삭제됨", 0.7)
        elif tok and tok!="NOP":
            jamo_stream.append(tok); set_popup(f"{tok} 입력", 0.55)

    # --- 메인 루프 ---
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            # 1. 시선 좌표 추출 및 보정
            curr_x, curr_y = None, None
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                raw = norm_iris_pos(lms, W, H)
                if raw:
                    # ✅ Kalman Filter Update
                    curr_x, curr_y = gaze_kf.update(raw[0], raw[1])
            
            now = time.time()
            if armed_action and now > armed_until:
                armed_action = None; set_popup("ARM 해제", 0.7)

            # 2. 방향 판정
            new_dir = "NEUTRAL"
            if curr_x is not None:
                new_dir = classify_dir_3way(curr_x, curr_y, cx, cy, th_x, th_y)
            else:
                neutral_since = now # 인식 실패 시 중립 취급
            
            # 3. 입력 로직 (Dwell Time)
            if new_dir == "NEUTRAL":
                if neutral_since is None: neutral_since = now
                if (now - neutral_since) >= CENTER_RELEASE_SEC and now >= cooldown_until:
                    allow_input = True
            else:
                neutral_since = None

            if new_dir != candidate_dir:
                candidate_dir = new_dir
                candidate_since = now

            if allow_input and candidate_dir in DIR2ARROW and now >= cooldown_until:
                s = strength_3way(curr_x, curr_y, cx, cy, th_x, th_y, candidate_dir)
                if (now - candidate_since) >= HOLD_SEC and s >= CONFIRM_RATIO:
                    arrow = DIR2ARROW[candidate_dir]
                    seq += arrow
                    set_popup(f"입력: {arrow}", 0.35)
                    allow_input = False
                    cooldown_until = now + COOLDOWN_SEC
                    neutral_since = None
                    
                    if len(seq) >= SEQ_LEN:
                        tok = decode_token(seq)
                        apply_token(tok)
                        seq = "" # 토큰 처리 후 리셋
                    candidate_dir = "NEUTRAL"
            
            # 4. 그리기 (UI)
            # 캔버스 초기화
            cam_h = H; canvas_h = cam_h + text_h; canvas_w = W + right_w
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:cam_h, :W] = frame # 카메라 영상

            # 상단 정보바
            canvas = draw_box(canvas, 0, 0, W, top_h, (0,0,0), 0.72)
            canvas = draw_text(canvas, 18, 12, "Gaze Arrow (젯슨 최적화)", 26)
            canvas = draw_text(canvas, 18, 46, f"DIR={new_dir} SEQ={seq}", 23, (230,230,230))
            if curr_x: canvas = draw_text(canvas, 18, 74, f"Kalman({curr_x:.3f}, {curr_y:.3f})", 21, (200,200,200))
            
            # 하단 텍스트바
            canvas = draw_box(canvas, 0, cam_h, W, text_h, (0,0,0), 0.84)
            composed = compose_hangul(jamo_stream).strip()
            show_text = composed[-18:] if len(composed)>18 else composed
            canvas = draw_text(canvas, 18, cam_h+16, "입력 문장:", 26, (210,210,255))
            canvas = draw_text(canvas, 18, cam_h+68, show_text, 64)

            # 팝업
            if popup and now <= popup_until:
                pw, ph = 560, 96
                px, py = W//2 - pw//2, cam_h//2 - ph//2
                canvas = draw_box(canvas, px, py, pw, ph, (0,0,0), 0.82)
                cv2.rectangle(canvas, (px,py), (px+pw, py+ph), (80,200,255), 2)
                canvas = draw_text(canvas, px+18, py+22, popup, 36, (80,200,255))

            # ✅ 5. 우측 패널 (캐싱 최적화)
            if cheat_on:
                if cached_cheat_panel is None:
                    # 패널을 처음 한 번만 그림
                    p = np.zeros((canvas_h, right_w, 3), dtype=np.uint8)
                    p[:] = (12,12,12)
                    y_pos = 12
                    for line in mapping_lines():
                        p = draw_text(p, 12, y_pos, line, 22, (240,240,240))
                        y_pos += 28
                    cached_cheat_panel = p # 저장
                
                # 캐시된 이미지 복사 (CPU 절약)
                canvas[:, W:W+right_w] = cached_cheat_panel
                # 동적 정보만 덧그리기
                canvas = draw_text(canvas, W+12, canvas_h-86, f"SEQ: {seq}", 24, (180,220,180))

            cv2.imshow("gaze_arrow_input", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord("r"):
                jamo_stream.clear(); seq = ""
                cx, cy, th_x, th_y = calibrate(cap, face_mesh)
                gaze_kf.reset()
                set_popup("재캘리브 완료", 0.9)
            elif key == ord("t"):
                cheat_on = not cheat_on

    finally:
        if cap: cap.release()
        cv2.destroyAllWindows()
        if face_mesh: face_mesh.close()
        release_camera_lock(lock_f)

if __name__ == "__main__":
    run()