# services/video_service.py
import os
import uuid
import shutil
from fastapi import UploadFile
import cv2
import numpy as np
import librosa
import json
import yt_dlp
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

def download_from_url(url: str):
    """
    yt-dlp로 URL에서 영상 다운로드.
    유튜브, 치지직 등 지원.
    """
    analysis_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, analysis_id)
    os.makedirs(session_dir, exist_ok=True)

    print(f"--- [WBB] URL 다운로드 시작: {url} ---")

    try:
        ydl_opts = {
            # 최대 1080p 영상 다운로드
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            # 저장 경로 및 파일명 설정
            'outtmpl': os.path.join(session_dir, 'video.%(ext)s'),
            # 진행상황 출력 끔 (서버 로그 정리용)
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = info.get('ext', 'mp4')
            saved_path = os.path.join(session_dir, f'video.{ext}')

        print(f"--- [WBB] 다운로드 완료: {saved_path} ---")
        return analysis_id, saved_path

    except Exception as e:
        print(f"--- [WBB] 다운로드 실패: {str(e)} ---")
        return analysis_id, None

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
        return (int(width*0.7), 0, int(width*0.3), int(height*0.5))
        
    return best_rect

def analyze_audio(video_path: str):
    print(f"--- [WBB] 오디오 분석 시작 ---")
    
    try:
        y, sr = librosa.load(video_path, mono=True, sr=None)
        hop_length = sr
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
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
        
        if count % int(fps) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
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

    # OCR 엔진 가동
    ai_engine.run_ocr_on_frames(analysis_id)

    # 오디오 분석
    audio_data = analyze_audio(video_path)
    audio_path = os.path.join(session_dir, "audio_data.json")
    with open(audio_path, "w", encoding="utf-8") as f:
        json.dump(audio_data, f, ensure_ascii=False, indent=4)
    print(f"--- [WBB] 오디오 데이터 저장 완료: {audio_path} ---")

    # 픽셀 변화량 분석
    pixel_data = analyze_pixel_change(video_path)
    pixel_path = os.path.join(session_dir, "pixel_data.json")
    with open(pixel_path, "w", encoding="utf-8") as f:
        json.dump(pixel_data, f, ensure_ascii=False, indent=4)
    print(f"--- [WBB] 픽셀 데이터 저장 완료: {pixel_path} ---")

    # 5. 하이라이트 파이프라인 실행
    from services import highlight_service
    highlight_service.run_highlight_pipeline(analysis_id, video_path)

    return saved_count