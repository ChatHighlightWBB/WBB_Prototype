from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from services import video_service

# 와바바(WBB) 분석 라우터 설정
router = APIRouter()

@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    영상을 업로드받아 서버에 저장하고, 백그라운드에서 채팅창 프레임 추출을 시작합니다.
    """
    
    # 1. 비디오 서비스 호출: 파일을 uploads/{UUID} 폴더에 저장하고 ID와 경로를 받아옴
    analysis_id, saved_path = video_service.save_upload_file(file)
    
    # 2. 백그라운드 작업 등록: 
    # 영상 캡처 작업은 시간이 오래 걸리므로, 클라이언트에게 응답을 먼저 보낸 뒤 서버 뒷편에서 작업을 계속합니다.
    background_tasks.add_task(video_service.extract_chat_frames, analysis_id, saved_path)
    
    # 3. 사용자에게 즉시 응답 반환
    return {
        "status": "success",
        "analysis_id": analysis_id,
        "filename": file.filename,
        "saved_path": saved_path,
        "message": "와바바(WBB) 엔진이 영상 분석(캡처)을 시작했습니다. 잠시만 기다려 주세요!"
    }