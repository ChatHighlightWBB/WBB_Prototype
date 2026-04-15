# services/video_service.py
import os
import uuid
import shutil
from fastapi import UploadFile
import cv2
import numpy as np
import librosa
import json
from services import ai_engine

UPLOAD_DIR = "uploads"

def save_upload_file(file: UploadFile):
    analysis_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, analysis_id)
    os.makedirs(session_dir, exist_ok=True)
    
    file_path = os.path.join(session_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return analysis_id, file_path

def detect_chat_roi(video_path):
    """영상의 특정 지점을 분석하여 채팅창 좌표(x, y, w, h)를 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
    ret, frame = cap.read()
    cap.release()

    if not ret: return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    max_area = 0
    height, width = frame.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > (height * width * 0.05) and h > w:
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)

    if not best_rect:
        # 기본값: 우측 30%, 상단 50%만 (하단 게임화면 제외)
        return (int(width*0.7), 0, int(width*0.3), int(height*0.5))
        
    return best_rect

def analyze_audio(video_path: str):
    """
    Librosa로 오디오 분석.
    1초 단위로 RMS(음량)와 주파수 중심(spectral centroid) 계산.
    반환: [{"second": 0, "rms": 0.12, "spectral_centroid": 2300.5}, ...]
    """
    print(f"--- [WBB] 오디오 분석 시작 ---")
    
    try:
        # librosa로 오디오 로드 (mono=True: 스테레오를 모노로 변환)
        y, sr = librosa.load(video_path, mono=True, sr=None)
        
        # 1초 단위 hop_length 계산
        hop_length = sr  # 1초 = 샘플레이트만큼의 샘플
        
        # RMS (소리 에너지) 계산 - 값이 클수록 소리가 큼
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Spectral Centroid (주파수 중심) - 값이 클수록 고음 성분 많음
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        audio_results = []
        for i, (r, sc) in enumerate(zip(rms, spectral_centroid)):
            audio_results.append({
                "second": i,
                "rms": round(float(r), 4),
                "spectral_centroid": round(float(sc), 2)
            })
        
        print(f"--- [WBB] 오디오 분석 완료 (총 {len(audio_results)}초) ---")
        return audio_results
        
    except Exception as e:
        print(f"--- [WBB] 오디오 분석 실패: {str(e)} ---")
        return []

def analyze_pixel_change(video_path: str):
    """
    OpenCV로 프레임 간 픽셀 변화량 분석.
    1초 단위로 이전 프레임과의 차이값 계산.
    반환: [{"second": 0, "pixel_diff": 1234.5}, ...]
    """
    print(f"--- [WBB] 픽셀 변화량 분석 시작 ---")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30
    
    pixel_results = []
    prev_frame = None
    second = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1초마다 분석
        if count % int(fps) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # 이전 프레임과 현재 프레임의 절대 차이값 평균
                diff = cv2.absdiff(prev_frame, gray)
                pixel_diff = float(np.mean(diff))
            else:
                pixel_diff = 0.0
            
            pixel_results.append({
                "second": second,
                "pixel_diff": round(pixel_diff, 4)
            })
            
            prev_frame = gray
            second += 1
        
        count += 1

    cap.release()
    print(f"--- [WBB] 픽셀 변화량 분석 완료 (총 {len(pixel_results)}초) ---")
    return pixel_results

def extract_chat_frames(analysis_id: str, video_path: str):
    session_dir = os.path.dirname(video_path)
    processed_dir = os.path.join(session_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # 1. 채팅창 ROI 탐지
    roi = detect_chat_roi(video_path)
    x, y, w, h = roi
    print(f"--- [WBB] 탐지된 채팅창 좌표: x={x}, y={y}, w={w}, h={h} ---")

    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30
    
    count = 0
    saved_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        if count % int(fps) == 0:
            chat_area = frame[y:y+h, x:x+w]
            frame_path = os.path.join(processed_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, chat_area)
            saved_count += 1
            
        count += 1

    cam.release()
    print(f"--- [WBB] 캡처 완료 (총 {saved_count}장) ---")

    # 2. OCR 엔진 가동
    ai_engine.run_ocr_on_frames(analysis_id)

    # 3. 오디오 분석
    audio_data = analyze_audio(video_path)
    audio_path = os.path.join(session_dir, "audio_data.json")
    with open(audio_path, "w", encoding="utf-8") as f:
        json.dump(audio_data, f, ensure_ascii=False, indent=4)
    print(f"--- [WBB] 오디오 데이터 저장 완료: {audio_path} ---")

    # 4. 픽셀 변화량 분석
    pixel_data = analyze_pixel_change(video_path)
    pixel_path = os.path.join(session_dir, "pixel_data.json")
    with open(pixel_path, "w", encoding="utf-8") as f:
        json.dump(pixel_data, f, ensure_ascii=False, indent=4)
    print(f"--- [WBB] 픽셀 데이터 저장 완료: {pixel_path} ---")

    return saved_count