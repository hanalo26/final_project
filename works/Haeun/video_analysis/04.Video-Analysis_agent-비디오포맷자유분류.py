# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import json
import asyncio
import argparse
import cv2                   # OpenCV - 정량 분석 (밝기, 색온도, 비율 등)
from yt_dlp import YoutubeDL # yt-dlp - 유튜브 영상 다운로드용

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
# → 루트를 import 경로에 추가해서 utils.py 등 공용 파일을 어디서든 불러올 수 있게 함
sys.path.append(str(Path(__file__).resolve().parents[3]))

# ========================================
# 2. 환경변수 로드 및 API 연결 확인
# ========================================

load_dotenv(dotenv_path=".env", override=True)

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", '')

# Vertex AI 유효성 검사
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

# 데이터 파일을 터미널에서 변수처럼 전달해주기 위해 필요한 변수 정의
parser = argparse.ArgumentParser()

# 필수 위치 인자: 터미널에서 첫 번째로 입력한 값이 csv_path에 저장됨
parser.add_argument('csv_path')

# 옵션 인자
parser.add_argument("--concurrent", type=int, default=3) # 동시 실행 수 (기본값 3)
parser.add_argument("--delay", type=int, default=10)     # 기본 대기 시간 (기본값 10초)
parser.add_argument("--video_dir", type=str,             # 영상 저장 폴더 위치 지정 (args.video_dir로 접근 가능)
    default="works/Haeun/video_analysis/videos")
parser.add_argument("--output_dir", type=str,            # 결과 저장 폴더 위치 지정 (args.output_dir로 접근 가능)
    default="works/Haeun/video_analysis/results")

args = parser.parse_args()

# CSV 파일명 기반으로 체크포인트 파일명 동적 생성 (확장자는 제거)
csv_stem = Path(args.csv_path).stem

# ========================================
# 4. 응답 스키마 정의
# TODO: video_format 두 가지 버전 테스트 예정
    # 방식 A. 자유 분류 (Gemini가 알아서 분류) <- 현재 파일
    # video_format: str

    # 방식 B. 합의된 범주로 분류 (합의된 영상 포맷 범주로 채우기)
    # video_format: Literal[...]
# ========================================

"""
응답 스키마 구조
====================================
VideoAnalysis (BaseModel)
├── video_id            (str)   : 영상 고유 ID (식별자)
├── production_quality  (str)   : 영상 제작 수준 (저예산 / 일반 / 고퀄리티 / 프로페셔널)
├── lighting_style      (str)   : 조명 스타일 (자연광 / 인공조명 / 역광 / 저조도 / 혼합)
├── color_mood          (str)   : 색감 분위기 (따뜻함 / 차가움 / 중립 / 비비드 / 무채색)
├── editing_pace        (str)   : 편집 속도 (매우 느림 / 느림 / 보통 / 빠름 / 매우 빠름)
├── motion_graphic      (str)   : 모션그래픽 사용 (없음 / 보조적 / 핵심요소)
├── video_format        (str)   : 영상 포맷 및 스토리텔링 방식
├── first_3sec          (str)   : 영상이 시작되고 3초 동안의 장면 구성 요소 (텍스트 / 인물 / 제품 / 장면 / 기업 로고 / 음식)
├── background_style    (str)   : 배경 스타일 (실내 / 실외 / 스튜디오 / 혼합)
├── top_colors          (list)  : 주요 컬러 팔레트 (상위 3개 색상)
├── person_ratio        (float) : 인물 등장 비율 (0.0~1.0)
├── face_ratio          (float) : 얼굴 등장 비율 (0.0~1.0)
├── text_ratio          (float) : 텍스트/자막 출현 비율 (0.0~1.0)
└── reason              (str)   : 위 항목들로 분석한 이유 (10자 이상)
"""

class VideoAnalysis(BaseModel):
    video_id: str = Field(
        description="영상의 고유 ID. 입력 데이터에서 그대로 가져온다."
    )
    
    production_quality: Literal[
        "저예산",       # 스마트폰 촬영, 조명/편집 거의 없음
        "일반",         # 기본 장비, 간단한 편집
        "고퀄리티",     # 전문 장비, 컬러그레이딩, 음향
        "프로페셔널",   # 광고 제작사 수준, 완성도 매우 높음
    ] = Field(description="영상의 전반적인 제작 수준을 평가한다.")
    
    lighting_style: Literal[
        "자연광",    # 햇빛 등 자연 광원 사용
        "인공조명",  # 인공 조명 장비로 통제된 환경
        "역광",     # 빛이 피사체 뒤에서 들어오는 형태
        "저조도",   # 전반적으로 어둡고 빛이 부족한 환경
        "혼합",     # 여러 조명 방식이 혼합됨
    ] = Field(description="영상에서 사용된 조명 스타일을 고른다.")

    color_mood: Literal[
        "따뜻함",  # 붉은/노란 계열의 따뜻한 색감
        "차가움",  # 푸른/회색 계열의 차가운 색감
        "중립",    # 특정 색감으로 치우치는 경향이 없는 중립적 색감
        "비비드",  # 채도가 높고 선명한 색감
        "무채색",  # 흑백 또는 채도가 매우 낮은 색감
    ] = Field(description="영상의 전반적인 색감 분위기를 고른다.")

    editing_pace: Literal[
        "매우 느림",  # 컷 전환이 거의 없고 여유로운 편집
        "느림",      # 컷 전환이 적고 여유로운 편집
        "보통",      # 일반적인 속도의 편집
        "빠름",      # 컷 전환이 잦고 역동적인 편집
        "매우 빠름", # 컷 전환이 매우 잦고 강렬한 편집
    ] = Field(description="영상의 편집 속도를 고른다.")

    motion_graphic: Literal[
        "없음",     # 모션그래픽 요소 전혀 없음
        "보조적",   # 자막/텍스트 애니메이션을 보조 수단으로 사용
        "핵심요소", # 모션그래픽이 영상의 주된 표현 수단
    ] = Field(description="영상 내 모션그래픽 사용 정도를 고른다.")
    
    video_format: str = Field(
        description="영상의 포맷 및 스토리텔링 방식을 자유롭게 분류한다."
    )
    
    first_3sec: Literal[
        "텍스트",   # 첫 3초에 텍스트/자막이 주를 이룸
        "인물",     # 첫 3초에 인물 등장이 주를 이룸
        "제품",     # 첫 3초에 제품이 주를 이룸
        "장면",     # 첫 3초에 배경/장면이 주를 이룸
        "기업 로고", # 첫 3초에 기업 로고가 주를 이룸
        "음식",     # 첫 3초에 음식 클로즈업이 주를 이룸
    ] = Field(description="영상이 시작되고 3초 동안의 장면 구성 요소를 고른다.")

    background_style: Literal[
        "실내",     # 건물 내부 공간
        "실외",     # 야외 공간
        "스튜디오",  # 스튜디오 세트장
        "혼합",     # 실내/실외/스튜디오가 혼합됨
    ] = Field(description="영상의 주된 배경 스타일을 고른다.")

    top_colors: list[str] = Field(
        description="영상에서 가장 많이 등장하는 상위 3개 색상을 색상명으로 표현한다. 예: ['파란색', '흰색', '회색']"
    )

    person_ratio: float = Field(
        description="전체 프레임 중 인물이 등장하는 프레임 비율. 반드시 0.0~1.0 사이의 소수로 반환한다. 예: 0.75",
        ge=0.0, le=1.0
    )

    face_ratio: float = Field(
        description="전체 프레임 중 얼굴이 감지되는 프레임 비율. 반드시 0.0~1.0 사이의 소수로 반환한다. 예: 0.45",
        ge=0.0, le=1.0
    )

    text_ratio: float = Field(
        description="전체 프레임 중 텍스트/자막이 출현하는 프레임 비율. 반드시 0.0~1.0 사이의 소수로 반환한다. 예: 0.60",
        ge=0.0, le=1.0
    )

    reason: str = Field(
        description="위 항목들을 이렇게 분석한 이유를 간단히 설명한다. (10자 이상)",
        min_length=10
    )

# ========================================
# 5. 체크포인트 함수 정의
# ========================================

# 체크포인트 파일 경로 (CSV 파일명 기반으로 동적 생성)
CHECKPOINT_FILE = Path(f"works/Haeun/video_analysis/checkpoints/checkpoint_{csv_stem}.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

CHECKPOINT_EVERY = 5          # N건마다 중간 저장
MAX_RETRIES = 4               # 최대 재시도 횟수
BASE_DELAY = args.delay       # 기본 대기 시간 (초)
MAX_DELAY = 60                # 최대 대기 시간 (초)
MAX_CONCURRENT = args.concurrent  # 동시 호출 수

sem = asyncio.Semaphore(MAX_CONCURRENT)

def save_checkpoint(results: list, path: Path = CHECKPOINT_FILE) -> None:
    """중간 결과를 JSON 파일로 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

def load_checkpoint(path: Path = CHECKPOINT_FILE) -> list:
    """이전 중간 저장 결과를 복원 — 없으면 빈 리스트 반환"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"📂 체크포인트 복원: {len(results)}건 (이미 처리된 항목을 건너뜁니다)")
        return results
    return []

# ========================================
# 6. yt-dlp을 사용한 영상 다운로드 함수 정의
# ========================================
def download_video(url: str, video_id: str, video_dir: str) -> str | None:
    """
    유튜브 URL을 받아서 영상을 다운로드하고 저장 경로를 반환

    Args:
        url      : 유튜브 숏츠 URL
        video_id : 영상 고유 ID (파일명으로 사용 예정)
        video_dir: 영상 저장 폴더 경로 (args.video_dir에서 받아옴)

    Returns:
        저장된 영상 파일 경로 (실패 시 None 반환)
    """
    output_path = str(Path(video_dir) / f"{video_id}.mp4")  # 파일을 저장할 때는 확장자 필요

    # 이미 다운로드된 영상이면 스킵
    if Path(output_path).exists():
        print(f"⏭️ [{video_id}] 이미 다운로드되었으므로, 건너뜀")
        return output_path

    ydl_opts = {
        "outtmpl": output_path,          # 저장 경로 및 파일명
        "format": "bestvideo+bestaudio/best",  # 최고 화질 비디오 + 오디오 병합
        "merge_output_format": "mp4",    # 병합 후 mp4로 저장
        "noplaylist": True,              # 재생목록이 아닌 단일 영상만 다운로드
        "quiet": True,                   # 터미널에 불필요한 로그 출력 X
        "no_warnings": True,             # 터미널에 경고 메시지 X
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ [{video_id}] 다운로드 완료")
        return output_path
    except Exception as e:
        print(f"❌ [{video_id}] 다운로드 실패: {str(e)[:100]}")
        return None
    
# ========================================
# 7. OpenCV 정량 분석 함수 정의 (6번 결과를 받아옴)
# ========================================
def analyze_quantitative(video_path: str, video_id: str) -> dict | None:
    """
    OpenCV로 영상을 분석하여 정량적 수치를 추출

    Args:
        video_path: 다운로드된 영상 파일 경로 (download_video()에서 받아옴)
        video_id  : 영상 고유 ID (로그 출력용)

    Returns:
        정량 분석 결과 딕셔너리 (실패 시 None)
        {
            "avg_brightness" : 평균 밝기 (0~255)
            "avg_blue"       : 평균 파란색 값 (0~255)
            "avg_green"      : 평균 초록색 값 (0~255)
            "avg_red"        : 평균 빨간색 값 (0~255)
        }
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ [{video_id}] 영상 열기 실패")
        return None

    # 분석에 사용할 변수 초기화
    total_frames = 0
    brightness_sum = 0.0
    all_colors = []

    while True:
        # cap.read(): 영상에서 프레임을 하나씩 꺼내옴
        # ret   → 프레임을 성공적으로 읽었는지 (True/False)
        # frame → 읽어온 프레임 이미지 데이터
        ret, frame = cap.read()
        if ret == False:
            break

        total_frames += 1

        # 평균 밝기 계산 (BGR → 그레이스케일 변환 후 평균)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_sum += gray.mean()

        # 색온도 계산용 BGR 값 추출
        # 프레임을 축소해서 대표 색상 추출 (크기에 따라 정확도와 속도 트레이드오프)
        # (1, 1)     → 가장 빠름, 가장 부정확
        # (10, 10)   → 빠름, 어느 정도 정확
        # (50, 50)   → 보통 속도, 더 정확 ← 현재 선택
        # (100, 100) → 느림, 가장 정확
        small = cv2.resize(frame, (50, 50))
        all_colors.append(small.mean(axis=(0, 1)).tolist())  # BGR 평균값 저장

    # cap.release(): 영상 파일을 닫아줌
    cap.release()

    if total_frames == 0:
        print(f"❌ [{video_id}] 프레임 추출 실패")
        return None

    # 평균 밝기
    avg_brightness = round(brightness_sum / total_frames, 2)

    # BGR 평균값 계산 (수치로만 반환, 판단 없음)
    avg_color = [sum(c[i] for c in all_colors) / len(all_colors) for i in range(3)]
    avg_blue, avg_green, avg_red = avg_color

    print(f"✅ [{video_id}] 정량 분석 완료 (총 {total_frames}프레임)")

    return {
        "avg_brightness" : round(avg_brightness, 2),
        "avg_blue"       : round(avg_blue, 2),
        "avg_green"      : round(avg_green, 2),
        "avg_red"        : round(avg_red, 2),
    }

# ========================================
# 8. 지수 백오프 및 병렬 처리 함수 정의 
# ========================================
async def analyze_one(
    agent,
    row,
    settings,
    stats,
    all_results,
    total_tokens,
    pbar
):
    """
    Semaphore로 동시 실행을 제한하면서 영상 1개를 처리하고,
    실패 시 지수 백오프로 재시도
    
    - 전체 작업 순서
      (1) yt-dlp 라이브러리를 사용해 유튜브 숏츠 영상 다운로드
      (2) OpenCV 라이브러리를 사용해 정량 분석 수행
      (3) Gemini를 사용하여 정성 분석 수행
      (4) 두 분석 결과를 통합하여 저장
    """
    async with sem:
        video_id = row['video_id']

        # Step 1. 영상 다운로드 (6번 함수 호출)
        video_path = download_video(row['final_url'], video_id, args.video_dir)
        if video_path is None:
            stats['fail'] += 1
            pbar.update(1)
            return

        # Step 2. OpenCV 정량 분석 (7번 함수 호출)
        quant_result = analyze_quantitative(video_path, video_id)
        if quant_result is None:
            stats['fail'] += 1
            pbar.update(1)
            return

        # Step 3. Gemini 정성 분석 (지수 백오프 적용)
        with open(video_path, 'rb') as f: # rb : 0,1로 이루어진 바이너리 형식으로 파일 읽기
            video_bytes = f.read()
        
        # ⚠️ 샘플링 규모가 커지면 GCS URI 방식으로 전환 검토 필요
        prompt = [
            BinaryContent(data=video_bytes, media_type="video/mp4"),
            f"video_id: {video_id}"
        ]
                
        for attempt in range(MAX_RETRIES):
            try:
                result = await agent.run(prompt, model_settings=settings)

                usage = result.usage()
                input_tok  = usage.input_tokens or 0
                output_tok = usage.output_tokens or 0
                total_tokens['input']  += input_tok
                total_tokens['output'] += output_tok
                
                # Step 4. 정량 + 정성 결과 통합
                combined = result.output.model_dump() # 정성분석 결과 저장
                combined.update(quant_result)         # 정량분석 결과 합치기
                combined['channel_title'] = row['channel_title']
                combined['domain'] = row['domain']
                
                all_results.append(combined)
                stats['success'] += 1

                pbar.update(1)
                pbar.set_postfix(
                    성공=stats['success'],
                    실패=stats['fail'],
                    입력토큰=total_tokens['input'],
                    출력토큰=total_tokens['output']
                )

                if stats['success'] % CHECKPOINT_EVERY == 0:
                    save_checkpoint(all_results)
                    print(f"💾 체크포인트 저장: {stats['success']}건 완료")

                await asyncio.sleep(BASE_DELAY)
                return

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = '429' in error_msg or 'rate' in error_msg.lower()

                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    if is_rate_limit:
                        delay = min(delay * 2, MAX_DELAY)
                    print(f"⚠️ [{row['video_id']}] 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기)")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ [최종 실패] {row['video_id']} | {error_msg[:100]}")
                    stats['fail'] += 1
                    pbar.update(1)
                    pbar.set_postfix(성공=stats['success'], 실패=stats['fail'])

# ========================================
# 9. 시스템 프롬프트
# ========================================
system_prompt = """
너는 기업 유튜브 숏츠 영상을 분석하는 비주얼 마케팅 전문가야.
입력된 영상을 직접 보고 아래 규칙에 따라 각 항목을 분석해.

[분석 규칙]

1. production_quality
- 스마트폰 촬영, 조명/편집 거의 없으면 → 저예산
- 기본 장비와 간단한 편집이 적용됐으면 → 일반
- 전문 장비, 컬러그레이딩, 음향이 느껴지면 → 고퀄리티
- 방송/광고 제작사 수준의 완성도가 느껴지면 → 프로페셔널

2. lighting_style
- 햇빛 등 자연 광원이 주를 이루면 → 자연광
- 인공 조명으로 통제된 환경이면 → 인공조명
- 빛이 피사체 뒤에서 들어오는 형태면 → 역광
- 전반적으로 어둡고 빛이 부족하면 → 저조도
- 여러 조명 방식이 섞여 있으면 → 혼합

3. color_mood
- 붉은/노란 계열이 주를 이루면 → 따뜻함
- 푸른/회색 계열이 주를 이루면 → 차가움
- 특정 색감 경향이 없으면 → 중립
- 채도가 높고 선명하면 → 비비드
- 흑백 또는 채도가 매우 낮으면 → 무채색

4. editing_pace (숏츠 기준)
- 숏츠치고 컷 전환이 거의 없고 장면이 길게 유지되면 → 매우 느림
- 컷 전환이 적고 여유로운 편집이면 → 느림
- 일반적인 숏츠 수준의 편집 속도면 → 보통
- 컷 전환이 잦고 역동적인 편집이면 → 빠름
- 컷 전환이 매우 잦고 강렬하면 → 매우 빠름

5. motion_graphic
- 모션그래픽 요소가 전혀 없으면 → 없음
- 자막, 로고, 간단한 텍스트 애니메이션 정도면 → 보조적
- 인포그래픽, 2D/3D 애니메이션, 데이터 시각화 등이 영상의 주된 표현 수단이면 → 핵심요소

6. video_format
- 영상의 포맷과 스토리텔링 방식을 자유롭게 판단해서 분류해

7. first_3sec
- 첫 3초에 가장 주를 이루는 단일 요소를 고른다
- 두 요소가 동시에 나오면 더 많은 화면을 차지하는 것을 선택
- 첫 3초에 텍스트/자막이 주를 이루면 → 텍스트
- 첫 3초에 인물이 주를 이루면 → 인물
- 첫 3초에 제품이 주를 이루면 → 제품
- 첫 3초에 배경/장면이 주를 이루면 → 장면
- 첫 3초에 기업 로고가 주를 이루면 → 기업 로고
- 첫 3초에 음식 클로즈업이 주를 이루면 → 음식

8. background_style
- 건물 내부 공간이 주배경이면 → 실내
- 야외 공간이 주배경이면 → 실외
- 스튜디오 세트장이면 → 스튜디오
- 여러 배경이 섞여 있으면 → 혼합

9. person_ratio
- 화면에 사람의 신체 일부(얼굴, 몸통, 손 등)가 보이면 인물 등장으로 판단
- 실루엣이나 그림자는 제외
- 0.0~1.0 사이의 소수로 반환

10. face_ratio
- 화면에 사람의 얼굴이 보이면 등장으로 판단
- 뒷모습, 옆모습도 포함
- 0.0~1.0 사이의 소수로 반환

11. text_ratio
- 자막, 제목 텍스트, 오버레이 글자, 이모지가 보이면 등장으로 판단
- 배경에 자연스럽게 있는 간판/포스터는 제외
- 0.0~1.0 사이의 소수로 반환

12. top_colors
- 영상 전체에서 가장 많이 등장하는 색상 상위 3개
- 배경색 포함
- 색상명은 한국어로 표현

[중요]
- 반드시 영상을 직접 보고 분석해
- 추측하지 말고 영상에서 직접 보이는 것을 근거로 판단해
- reason에는 위 항목들을 이렇게 분류한 이유를 간단히 설명해
"""

# ========================================
# 10. Agent 실행 함수
# ========================================
async def run_agent(df: pd.DataFrame) -> tuple:
    # 체크포인트 복원
    all_results = load_checkpoint()
    processed_ids = {r['video_id'] for r in all_results}

    # 미처리 영상 필터링
    pending = [row for _, row in df.iterrows() if row['video_id'] not in processed_ids]

    if processed_ids:
        print(f"스킵: {len(processed_ids)}개 (이미 처리됨)")
    print(f"처리 대상: {len(pending)}개")
    print(f"처리 방식: 병렬 처리 (최대 {MAX_CONCURRENT}개 동시 실행)")
    print(f"재시도: 최대 {MAX_RETRIES}회 (지수 백오프, 기본 {BASE_DELAY}초)")
    print("=" * 60)

    stats = {'success': len(all_results), 'fail': 0}
    total_tokens = {'input': 0, 'output': 0}

    # Agent 초기화
    agent = Agent(
        model_id,
        output_type=VideoAnalysis,
        system_prompt=system_prompt,
        retries=3
    )

    settings = GoogleModelSettings(temperature=0.3)
    pbar = tqdm(total=len(pending), desc="영상 분석")
    
    # 병렬 처리
    tasks = [
        analyze_one(agent, row, settings, stats, all_results, total_tokens, pbar)
        for row in pending
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    # 최종 체크포인트 저장
    save_checkpoint(all_results)

    return all_results, stats, total_tokens

# ========================================
# 11. main 실행 블록
# ========================================

async def main():
    # args.csv_path로 터미널에서 전달받은 경로의 CSV 파일 불러오기
    df = pd.read_csv(args.csv_path, encoding="utf-8")

    print(f"전체 영상 수: {len(df)}개")
    print("=" * 60)

    # Agent 호출
    all_results, stats, total_tokens = await run_agent(df)

    # 처리 결과 요약
    print()
    print("=" * 60)
    print("배치 처리 완료 (병렬 처리 방식)")
    print(f"  성공: {stats['success']}개")
    print(f"  실패: {stats['fail']}개")
    print(f"  성공률: {stats['success'] / max(stats['success'] + stats['fail'], 1) * 100:.1f}%")
    print(f"  체크포인트: {CHECKPOINT_FILE}")
    print("=" * 60)

    # 정성 + 정량 분석 결과 통합 CSV 저장
    df_result = pd.DataFrame(all_results)
    output_path = Path(args.output_dir) / f"result_{csv_stem}.csv"
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n분석 결과 저장 완료: {output_path}")

    print(f"  입력 토큰: {total_tokens['input']:,} / 출력 토큰: {total_tokens['output']:,}")
    if stats['success'] > 0:
        print(f"  1건 평균: 입력 {total_tokens['input'] / stats['success']:.0f} / 출력 {total_tokens['output'] / stats['success']:.0f} tokens")


# 해당 파일을 직접 호출해서 실행하면 실행되도록 하는 함수
if __name__ == '__main__':
    asyncio.run(main())