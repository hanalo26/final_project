# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import json
import asyncio
import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from pydantic_ai import Agent, ModelRetry, BinaryContent

from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

import sys
# works/Haeun/video_analysis/ 에서 프로젝트 루트(final_project/)까지 3단계 위
sys.path.append(str(Path(__file__).resolve().parents[3]))

# ========================================
# 2. 환경변수 로드 및 API 연결 확인
# ========================================
load_dotenv(dotenv_path=".env", override=True)

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", '')

provider = GoogleProvider(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_REGION,
)
model_id = GoogleModel(GEMINI_MODEL, provider=provider)

print(f"Google Cloud Project: {'✅' if GOOGLE_CLOUD_PROJECT else '❌'} ({GOOGLE_CLOUD_PROJECT})")
print(f"사용할 모델: {GEMINI_MODEL} / 지역: {GOOGLE_CLOUD_REGION}")

# ========================================
# 3. argparse 정의
# ========================================
# TODO: 집계 결과 파일 경로 입력받기
# TODO: output_dir 지정

# ========================================
# 4. 응답 스키마 정의
# ========================================
# TODO: 보고서 구조 확정 후 작성

# ========================================
# 5. 체크포인트 함수 정의
# ========================================
# TODO: argparse 완성 후 경로 설정

# ========================================
# 6. 보고서 생성 함수 정의
# ========================================
# TODO: 보고서 내용 확정 후 작성
# TODO: matplotlib/seaborn으로 그래프 직접 생성 (Gemini 아님)
#       → 수치 데이터는 Python 코드로 직접 시각화
# TODO: Gemini로 텍스트 인사이트 작성
# TODO: 그래프 + 텍스트 합쳐서 보고서 저장
# TODO: 마크다운 저장 방식 (수업 save_markdown_report 재사용 예정)

# ========================================
# 7. 지수 백오프 및 병렬 처리 함수 정의
# ========================================
# TODO: analyze_one 구조 재사용

# ========================================
# 8. 시스템 프롬프트
# ========================================
# TODO: 보고서 생성용 시스템 프롬프트 작성
# TODO: 보고서 내용 확정 후 작성

# ========================================
# 9. Agent 실행 함수
# ========================================
# TODO: run_agent 구조 재사용

# ========================================
# 10. main 실행 블록
# ========================================
# TODO: main 구조 재사용
async def main():
    pass

# 해당 파일을 직접 호출해서 실행하면 실행되도록 하는 함수
if __name__ == '__main__':
    asyncio.run(main())