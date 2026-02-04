import sys
import os
import cv2
import time
from dotenv import load_dotenv

# ----------------------------------------------------------------
# 1. 경로 설정 (모듈을 못 찾는 에러 방지)
# ----------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # src 폴더
project_root = os.path.dirname(current_dir)               # smart_mirror1 폴더
gaze_dir = os.path.join(project_root, "GazeTracking")     # GazeTracking 폴더

sys.path.append(project_root)
sys.path.append(gaze_dir)

# ----------------------------------------------------------------
# 2. 모듈 임포트
# ----------------------------------------------------------------
try:
    from gaze_tracking import GazeTracking
    # src 폴더 내부의 모듈들을 가져옵니다.
    from minimax_client import ask_minimax
    from supabase_rest import insert_request, update_by_id
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("💡 팁: 가상환경(venv)이 켜져 있는지, 경로가 맞는지 확인하세요.")
    sys.exit(1)

# 환경 변수 로드 (.env)
load_dotenv()

def main():
    # ----------------------------------------------------------------
    # 3. 초기화
    # ----------------------------------------------------------------
    print("==================================================")
    print("🚀 RPi 5 Pipeline Test Started (WSL Optimized)")
    print(" [왼쪽 보기] = 질문: '안녕?'")
    print(" [오른쪽 보기] = 질문: '오늘 날씨 어때?'")
    print(" [눈 깜빡임] = 전송 (SEND)")
    print(" [q 키]      = 종료")
    print("==================================================")

    gaze = GazeTracking()
    
    # ----------------------------------------------------------------
    # 4. 카메라 설정 (WSL 타임아웃 해결을 위한 핵심 파트 ⭐)
    # ----------------------------------------------------------------
    # 0번 카메라를 리눅스 전용 모드(V4L2)로 엽니다.
    webcam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # 👇 [중요] 데이터 양을 줄여서 WSL 타임아웃을 막는 설정 (MJPEG)
    webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_FPS, 30)

    # 카메라가 안 열리면 에러 메시지 출력
    if not webcam.isOpened():
        print("⚠️ 카메라(0번)를 열 수 없습니다.")
        print("💡 팁: 코드를 수정하여 cv2.VideoCapture(1, ...) 로 변경해보세요.")
        return

    # 변수 초기화
    text = ""
    current_question = None  # 현재 선택된 질문

    while True:
        # ----------------------------------------------------------------
        # 5. 영상 읽기 및 시선 추적
        # ----------------------------------------------------------------
        ret, frame = webcam.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다. (카메라 연결 끊김?)")
            break

        # GazeTracking 분석
        gaze.refresh(frame)
        frame = gaze.annotated_frame() # 눈동자 표시된 화면

        # ----------------------------------------------------------------
        # 6. 로직: 눈 방향 판단
        # ----------------------------------------------------------------
        if gaze.is_blinking():
            text = "Blinking (SENDING...)"
            
            # 질문이 선택된 상태라면 -> 전송 시작
            if current_question:
                print(f"\n🚀 전송 중... 질문: {current_question}")
                
                # 1) DB에 'sending' 상태로 저장
                req_id = insert_request(current_question)
                
                if req_id:
                    # 2) AI에게 질문
                    ai_response = ask_minimax(current_question)
                    print(f"🤖 AI 응답: {ai_response}")
                    
                    # 3) DB 업데이트 (완료)
                    update_by_id(req_id, ai_response)
                    print("✅ DB 저장 완료!")
                
                # 전송 후 질문 초기화 (중복 전송 방지)
                current_question = None
                time.sleep(1) # 너무 빠른 연속 전송 방지

        elif gaze.is_right():
            text = "Looking Right -> 'Weather'"
            current_question = "오늘 서울 날씨 어때?"
        
        elif gaze.is_left():
            text = "Looking Left -> 'Hello'"
            current_question = "안녕? 너는 누구니?"
        
        elif gaze.is_center():
            text = "Looking Center"
            # 중앙을 보면 질문 선택을 취소하고 싶으면 아래 주석 해제
            # current_question = None 

        # ----------------------------------------------------------------
        # 7. 화면 출력
        # ----------------------------------------------------------------
        # 화면에 현재 상태 텍스트 표시
        cv2.putText(frame, text, (90, 60), cv2.VideoWriter_fourcc(*'XVID') if 0 else cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        # 선택된 질문이 있으면 화면 아래에 표시
        if current_question:
            cv2.putText(frame, f"Selected: {current_question}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Smart Mirror Test", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) == ord('q'):
            print("👋 프로그램을 종료합니다.")
            break
    
    # ----------------------------------------------------------------
    # 8. 종료 처리
    # ----------------------------------------------------------------
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()