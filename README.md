# Capstone_WBB
스트리밍 영상 내 채팅 데이터를 AI로 분석하여 하이라이트 구간을 자동으로 추출하는 시스템의 백엔드 프로토타입

기술 스택 (Tech Stack)

Framework: FastAPI

Computer Vision: OpenCV (cv2)

AI/OCR: EasyOCR -> 교체예정 PaddleOCR (PP-OCRv3)

Language: Python 3.9+

GPU: NVIDIA CUDA 12.4 (RTX 계열 권장)

# 실행 방법
1. 브랜치 이동\
git checkout prototype/chatlight-backend

2. 가상환경 설정 및 패키지 설치\
python -m venv venv\
source venv/Scripts/activate  # Windows 기준\
pip install -r requirements.txt\
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

3. 서버 실행\
uvicorn main:wbb --reload

`venv/`, `uploads/`, `models/` 폴더는 Git에 올리지 않음\
PyTorch는 requirements.txt와 별도로 설치해야 함\
GPU 없는 환경에서는 `ai_engine.py`의 `gpu=True` → `gpu=False` 로 변경
## 📁 Project Structure

```text
WBB_backend/
├── ai/             # KoBERT 감성 분석 모델 관련 로직 및 전처리
├── api/            # FastAPI 엔드포인트 및 라우터 설정 (API 경로)
├── data/           # 샘플 데이터 및 결과물 저장소
├── docs/           # API 명세서 및 프로젝트 설계 문서
├── models/         # 학습된 모델 가중치(.pt, .bin) 파일 보관
├── services/       # 비즈니스 로직 (OCR 엔진, 비디오 처리 핵심 기능)
├── uploads/        # 업로드된 원본 영상 및 프레임 추출 이미지 (Git 제외 권장)
├── main.py         # 백엔드 서버 실행 엔트리 포인트
└── requirements.txt # 설치가 필요한 라이브러리 목록
