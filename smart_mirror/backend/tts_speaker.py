from gtts import gTTS
import pygame
import os
import time

def speak_text(text, lang='ko'):
    """
    텍스트를 입력받아 음성(MP3)으로 변환 후 재생 (pygame 사용)
    WSL 환경에서는 pulseaudio 또는 WSLg 오디오가 작동해야 소리가 납니다.
    """
    if not text:
        return

    try:
        # 1. gTTS로 음성 파일 생성
        tts = gTTS(text=text, lang=lang)
        filename = "temp_voice.mp3"
        
        # 파일이 이미 있으면 삭제 (권한 에러 방지)
        if os.path.exists(filename):
            os.remove(filename)
            
        tts.save(filename)

        # 2. Pygame으로 재생
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # 재생이 끝날 때까지 대기 (비동기로 하려면 이 루프를 제거하거나 스레드 사용)
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # 3. 파일 정리 (재생 후 삭제하려면 mixer를 멈추고 unload 해야 함)
        pygame.mixer.music.unload()
        # os.remove(filename) # 디버깅을 위해 파일 남겨둠, 필요시 주석 해제

    except Exception as e:
        print(f"[TTS Error] {e}")

if __name__ == "__main__":
    speak_text("안녕하세요. 스마트 미러입니다.")