# main.py
from fastapi import FastAPI
from api.v1 import analysis

# 프로젝트 이름을 WBB로 업데이트!
wbb = FastAPI(title="WBB - 와바바 백엔드 엔진")

wbb.include_router(analysis.router, prefix="/api/v1")

@wbb.get("/")
def root():
    return {"message": "와바바(WBB) 서버에 접속하신 것을 환영합니다! 분석을 시작해볼까요?"}