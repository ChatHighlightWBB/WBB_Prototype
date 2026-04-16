# api/v1/analysis.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from services import video_service

router = APIRouter()

# URL 입력용 데이터 모델
class VideoURL(BaseModel):
    url: str

@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """파일 업로드로 영상 분석 시작"""
    
    analysis_id, saved_path = video_service.save_upload_file(file)
    background_tasks.add_task(video_service.extract_chat_frames, analysis_id, saved_path)
    
    return {
        "status": "success",
        "analysis_id": analysis_id,
        "filename": file.filename,
        "message": "와바바(WBB) 엔진이 영상 분석을 시작했습니다."
    }

@router.post("/upload-url")
async def upload_video_url(background_tasks: BackgroundTasks, body: VideoURL):
    """URL 입력으로 영상 분석 시작 (yt-dlp 사용)"""
    
    analysis_id, saved_path = video_service.download_from_url(body.url)
    
    if not saved_path:
        return {
            "status": "error",
            "message": "URL에서 영상을 가져오지 못했습니다. URL을 확인해주세요."
        }
    
    background_tasks.add_task(video_service.extract_chat_frames, analysis_id, saved_path)
    
    return {
        "status": "success",
        "analysis_id": analysis_id,
        "url": body.url,
        "message": "와바바(WBB) 엔진이 영상 분석을 시작했습니다."
    }