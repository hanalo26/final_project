# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import json
import asyncio
import argparse
import cv2        # OpenCV - 정량 분석 (밝기, 채도, RGB 등)
import requests

import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, List
from pydantic_ai import Agent, BinaryContent

from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

import sys
# works/Haeun/thumbnail_analysis/ 에서 프로젝트 루트(final_project/)까지 3단계 위
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
parser.add_argument("--thumbnail_dir", type=str,         # 썸네일 저장 폴더 위치 지정 (args.thumbnail_dir로 접근 가능)
    default="works/Haeun/thumbnail_analysis/thumbnails")
parser.add_argument("--output_dir", type=str,            # 결과 저장 폴더 위치 지정 (args.output_dir로 접근 가능)
    default="works/Haeun/thumbnail_analysis/results")

args = parser.parse_args()

# CSV 파일명 기반으로 체크포인트 파일명 동적 생성 (확장자는 제거)
csv_stem = Path(args.csv_path).stem

# ========================================
# 4. 응답 스키마 정의
# ========================================
"""
응답 스키마 구조
====================================
ThumbnailAnalysis (BaseModel)
├── video_id                    (str)   : 영상 고유 ID
├── visual_summary              (str)   : 썸네일 전체 장면 요약 (한국어 10자 이상)
├── main_objects                (list)  : 썸네일에 존재하는 주요 객체 리스트 (사람, 제품, 글자, 브랜드로고 등)
├── thumbnail_category          (str)   : 썸네일 카테고리 (정보 전달형 / 제품 홍보형 / 브랜드 이미지형 / 이벤트 홍보형 / 인터뷰/인물형 / 예능/콘텐츠형 / 리뷰/비교형 / 기타)
├── has_person                  (bool)  : 인물 등장 여부
├── person_count                (int)   : 등장 인물 수
├── has_face                    (bool)  : 얼굴 등장 여부
├── has_text                    (bool)  : 텍스트 존재 여부
├── text_on_thumbnail           (str)   : 썸네일 내 텍스트 문구 (없으면 '-' 반환)
├── text_language               (str)   : 텍스트 언어 (한국어 / 영어 / 2가지 이상의 언어가 혼용됨 / 기타)
├── text_size_level             (str)   : 텍스트 크기 수준 (none / small / medium / large)
├── brand_name_visible          (bool)  : 브랜드명 노출 여부
├── dominant_colors             (list)  : 주요 색상 리스트 (파란색, 흰색, 검정색 등)
├── color_tone                  (str)   : 색감 톤 (warm / cool / neutral)
├── background_complexity_level (str)   : 배경 복잡도 (low / medium / high)
├── composition_style           (str)   : 구도 스타일 (인물 중심 / 제품 중심 / 텍스트 중심 / 혼합 / 기타)
├── attention_hook              (str)   : 클릭 유도 핵심 시각 요소 (한국어 10자 이상)
├── visual_hook_level           (int)   : 시각적 자극성 정도 (0~5 정수)
├── design_quality_level        (int)   : 디자인 완성도 (0~5 정수)
└── reason                      (str)   : 분석 이유 (10자 이상)
"""

class ThumbnailAnalysis(BaseModel):
    video_id: str = Field(description="영상의 고유 ID. 입력 데이터에서 그대로 가져온다.")

    visual_summary: str = Field(
        description="썸네일의 전체 장면을 10자 이상의 한국어 문장으로 요약한다.",
        min_length=10
    )

    main_objects: List[str] = Field(
        description="썸네일에 존재하는 주요 객체 리스트를 한국어로 반환한다. "
                    "예: 사람, 제품, 글자, 브랜드로고"
    )

    thumbnail_category: Literal[
        "정보 전달형",
        "제품 홍보형",
        "브랜드 이미지형",
        "이벤트 홍보형",
        "인터뷰/인물형",
        "예능/콘텐츠형",
        "리뷰/비교형",
        "기타",
    ] = Field(description="썸네일의 전반적인 카테고리를 고른다.")

    has_person: bool = Field(description="썸네일에 인물이 등장하는지 여부를 반환한다.")

    person_count: int = Field(
        description="썸네일에 등장하는 인물 수를 반환한다."
    )

    has_face: bool = Field(description="썸네일에 얼굴이 등장하는지 여부를 반환한다.")

    has_text: bool = Field(description="썸네일에 텍스트가 존재하는지 여부를 반환한다.")

    text_on_thumbnail: str = Field(
        description="썸네일에 존재하는 텍스트를 반환한다. "
                    "텍스트가 존재하지 않으면 '-'을 반환한다."
    )

    text_language: Literal[
        '한국어',
        '영어',
        '2가지 이상의 언어가 혼용됨',
        '기타'
    ] = Field(description="썸네일에 포함된 텍스트의 언어를 반환한다.")

    text_size_level: Literal[
        "none",     # 텍스트가 없음
        "small",    # 작은 글씨
        "medium",   # 중간 글씨
        "large"     # 큰 글씨
    ] = Field(description="썸네일 내 텍스트의 크기 수준을 고른다.")

    brand_name_visible: bool = Field(description="썸네일에 브랜드명이 노출되는지 여부를 반환한다.")

    dominant_colors: List[str] = Field(
        description="썸네일에서 가장 많이 쓰인 색상 3가지를 한국어로 반환한다. "
                    "예: 파란색, 흰색, 검정색"
    )

    color_tone: Literal["warm", "cool", "neutral"] = Field(description="썸네일의 전반적인 색감 톤을 고른다.")

    background_complexity_level: Literal["low", "medium", "high"] = Field(description="썸네일 배경의 복잡도를 고른다.")

    composition_style: Literal[
        "인물 중심",
        "제품 중심",
        "텍스트 중심",
        "혼합",
        "기타",
    ] = Field(description="썸네일의 주된 구도 스타일을 고른다.")

    attention_hook: str = Field(
        description="시청자의 클릭을 유도하는 핵심 시각 요소를 10자 이상의 한국어 문장으로 반환한다.",
        min_length=10
    )

    visual_hook_level: int = Field(
        description="썸네일의 시각적 자극성 정도를 0~5 정수로 평가한다.",
        ge=0, le=5
    )

    design_quality_level: int = Field(
        description="썸네일의 전반적인 디자인 완성도를 0~5 정수로 평가한다.",
        ge=0, le=5
    )

    reason: str = Field(
        description="위 항목들을 이렇게 분석한 이유를 10자 이상의 한국어 문장으로 설명한다.",
        min_length=10
    )

# ========================================
# 5. 체크포인트 함수 정의
# ========================================

# 체크포인트 파일 경로 (CSV 파일명 기반으로 동적 생성)
CHECKPOINT_FILE = Path(f"works/Haeun/thumbnail_analysis/checkpoints/checkpoint_{csv_stem}.json")
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
# 6. 썸네일 다운로드 함수 정의
# ========================================
def download_thumbnail(video_id: str, thumbnail_dir: str) -> str | None:
    """
    video_id를 받아서 유튜브 썸네일 이미지를 다운로드하고 저장 경로를 반환
    maxresdefault(1280x720) 우선 시도 → 실패 시 hqdefault(480x360)로 폴백

    Args:
        video_id     : 영상 고유 ID (파일명 및 URL 생성에 사용)
        thumbnail_dir: 썸네일 저장 폴더 경로 (args.thumbnail_dir에서 받아옴)

    Returns:
        저장된 썸네일 파일 경로 (실패 시 None 반환)
    """

    # 썸네일 저장 폴더가 없으면 자동 생성
    Path(thumbnail_dir).mkdir(parents=True, exist_ok=True)
    
    # 저장할 파일 경로
    output_path = Path(thumbnail_dir) / f"{video_id}.jpg"

    # 이미 다운로드된 썸네일이면 스킵 (재실행 시 중복 다운로드 방지)
    if output_path.exists():
        print(f"⏭️ [{video_id}] 이미 다운로드되었으므로, 건너뜀")
        return str(output_path)

    # 시도할 URL 목록: 고화질 순으로 정렬
    # maxresdefault: 1280x720 고화질 — 모든 영상에 존재하지 않을 수 있음
    # hqdefault    : 480x360  — 모든 영상에 항상 존재하는 썸네일 해상도
    urls = [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",  # 1280x720 (없을 수도 있음)
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",      # 480x360  (항상 존재)
    ]

    for url in urls:
        try:
            res = requests.get(url, timeout=10)

            if res.status_code == 200:
                # 받아온 바이트 데이터를 numpy 배열로 변환 후 OpenCV로 해상도 확인
                # → 파일 저장 없이 메모리에서 바로 유효성 판별
                img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR) # .imdecode(): 메모리의 바이트 데이터를 numpy 배열로 변환

                # 유튜브가 썸네일이 없을 때 반환하는 기본 대체 이미지는 120x90px
                # → width가 200px 미만이면 실제 썸네일이 아닌 것으로 판별
                if img is not None and img.shape[1] >= 200:
                    output_path.write_bytes(res.content)
                    print(f"✅ [{video_id}] 썸네일 다운로드 완료")
                    return str(output_path)

        except Exception as e:
            print(f"⚠️ [{video_id}] URL 시도 실패 ({url}): {str(e)[:100]}")
            continue
        
    # 모든 URL 시도 후 실패한 video_id를 출력하고 None 반환
    print(f"❌ [{video_id}] 썸네일 다운로드 실패 (모든 URL 시도 후)")
    return None

# ========================================
# 7. OpenCV 정량 분석 함수 정의 (6번 결과를 받아옴)
# ========================================
def analyze_quantitative(thumbnail_path: str, video_id: str) -> dict | None:
    """
    OpenCV로 썸네일 이미지를 분석하여 정량적 수치를 추출

    Args:
        thumbnail_path: 다운로드된 썸네일 파일 경로 (download_thumbnail()에서 받아옴)
        video_id      : 영상 고유 ID (로그 출력용)

    Returns:
        정량 분석 결과 딕셔너리 (실패 시 None)
        {
            "brightness_mean" : 평균 밝기 (0~255)
            "saturation_mean" : 평균 채도 (0~255)
            "contrast_std"    : 밝기 표준편차 (색의 대비를 의미)
            "avg_blue"        : 평균 파란색 값 (0~255)
            "avg_green"       : 평균 초록색 값 (0~255)
            "avg_red"         : 평균 빨간색 값 (0~255)
        }
    """
    # cv2.imread(): 디스크에 저장된 파일 경로를 받아 numpy 배열로 변환
    img = cv2.imread(thumbnail_path)

    if img is None:
        print(f"❌ [{video_id}] 이미지 열기 실패")
        return None

    # BGR → HSV 변환 (밝기/채도 추출에 더 직관적인 색공간)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # BGR → 그레이스케일 변환 (대비 계산용)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지를 50x50으로 축소해서 BGR 대표 색상 추출
    small = cv2.resize(img, (50, 50))
    avg_blue, avg_green, avg_red = small.mean(axis=(0, 1))  # (height, width) 축 기준 BGR 평균

    print(f"✅ [{video_id}] 정량 분석 완료")

    return {
        "brightness_mean" : round(float(np.mean(hsv[:, :, 2])), 2),  # HSV의 V채널 = 밝기
        "saturation_mean" : round(float(np.mean(hsv[:, :, 1])), 2),  # HSV의 S채널 = 채도
        "contrast_std" : round(float(np.std(gray)), 2),           # 표준편차가 클수록 대비가 강함
        "avg_blue" : round(float(avg_blue), 2),
        "avg_green" : round(float(avg_green), 2),
        "avg_red" : round(float(avg_red), 2),
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
    Semaphore로 동시 실행을 제한하면서 썸네일 1개를 처리하고,
    실패 시 지수 백오프로 재시도

    - 전체 작업 순서
      (1) video_id 기반으로 유튜브 썸네일 이미지 다운로드
      (2) OpenCV 라이브러리를 사용해 정량 분석 수행
      (3) Gemini를 사용하여 정성 분석 수행
      (4) 두 분석 결과를 통합하여 저장
        - 분석에 사용한 썸네일 이미지는 삭제.
    """
    async with sem:
        video_id = row['video_id']

        # Step 1. 썸네일 다운로드 (6번 함수 호출)
        thumbnail_path = download_thumbnail(video_id, args.thumbnail_dir)
        if thumbnail_path is None:
            stats['fail'] += 1
            pbar.update(1)
            return

        # Step 2. OpenCV 정량 분석 (7번 함수 호출)
        quant_result = analyze_quantitative(thumbnail_path, video_id)
        if quant_result is None:
            stats['fail'] += 1
            pbar.update(1)
            return

        # Step 3. Gemini 정성 분석 (지수 백오프 적용)
        with open(thumbnail_path, 'rb') as f:  # rb: 0,1로 이루어진 바이너리 형식으로 파일 읽기
            image_bytes = f.read()

        prompt = [
            BinaryContent(data=image_bytes, media_type="image/jpeg"),  # 이미지(image/jpeg) 파일 전달
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
                combined = result.output.model_dump()  # 정성 분석 결과 저장
                combined.update(quant_result)          # 정량 분석 결과 합치기
                combined['channel_title'] = row['channel_title']
                combined['domain'] = row['domain']

                all_results.append(combined)

                # Step 5. 로컬 파일 삭제 (분석 완료 후 불필요)
                try:
                    Path(thumbnail_path).unlink()  # Path.unlink(): 파일 삭제
                    print(f"🗑️ [{video_id}] 로컬 파일 삭제 완료")
                except Exception as e:
                    print(f"⚠️ [{video_id}] 로컬 파일 삭제 실패: {str(e)[:100]}")

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
                    print(f"⚠️ [{video_id}] 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기)")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ [최종 실패] {video_id} | {error_msg[:100]}")
                    stats['fail'] += 1
                    pbar.update(1)
                    pbar.set_postfix(성공=stats['success'], 실패=stats['fail'])
                    
# ========================================
# 9. 시스템 프롬프트
# ========================================
system_prompt = """
너는 기업 유튜브 썸네일 분석 전문가야.
입력된 유튜브 썸네일 이미지를 보고 기업 유튜브 관점에서 시각적 특징을 분석해.

[작업 순서]
1. 썸네일 전체 장면을 파악해.
2. 인물, 얼굴, 인물 수를 확인해.
3. 텍스트 존재 여부, 문구, 언어, 크기를 확인해.
4. 브랜드명 노출 여부를 확인해.
5. 색감, 배경 복잡도, 구도 스타일을 파악해.
6. 클릭 유도 요소(attention_hook)를 파악해.
7. 썸네일 카테고리, 시각적 자극성, 디자인 완성도를 종합적으로 평가해.
8. 위 분석을 바탕으로 reason을 작성해.

[분석 원칙]
1. 기업 공식 채널 또는 브랜드 채널에 업로드된 영상의 썸네일이야.
2. 썸네일의 인물, 제품, 텍스트, 브랜드 이름 노출 여부, 색감, 구도, 클릭 유도 요소를 분석해.
3. 불확실한 내용은 과도하게 단정하지 말고, reason에서 반드시 언급해.

[반환값 처리 원칙]
1. 값이 존재하지 않거나 확인이 불가능한 경우 '-'을 반환해.
2. 리스트 필드(dominant_colors, main_objects)는 확인된 값만 반환해.
   - 값이 없으면 빈 리스트([])를 반환해.
   - dominant_colors는 최대 3가지를 반환하되, 3가지 미만이면 있는 만큼만 반환해.
3. person_count는 has_person=False이면 0을, has_person=True이면 반드시 1 이상을 반환해.

[평가 기준]
1. thumbnail_category는 반드시 아래 중 하나로 분류해.
   - 정보 전달형
   - 제품 홍보형
   - 브랜드 이미지형
   - 이벤트 홍보형
   - 인터뷰/인물형
   - 예능/콘텐츠형
   - 리뷰/비교형
   - 기타

2. visual_hook_level — 썸네일의 시각적 자극성 정도를 0~5 정수로 평가해.
   - 0: 자극적 요소 없음
   - 1: 매우 약한 자극 (밝은 색상 정도)
   - 2: 약한 자극 (강조 텍스트 또는 클로즈업)
   - 3: 보통 (과장된 표정 또는 강한 색상 대비)
   - 4: 강한 자극 (자극적 텍스트 + 과장된 표정)
   - 5: 매우 강한 자극 (낚시성 요소가 복합적으로 존재)

3. design_quality_level — 썸네일의 디자인 완성도를 0~5 정수로 평가해.
   - 0: 매우 낮음 (구도/색감/텍스트 모두 조잡함)
   - 1: 낮음 (전반적으로 완성도가 떨어짐)
   - 2: 다소 낮음 (일부 요소만 완성도 있음)
   - 3: 보통 (전반적으로 무난한 수준)
   - 4: 높음 (색감, 구도, 텍스트 가독성이 좋음)
   - 5: 매우 높음 (전문 디자이너 수준의 완성도)
"""

# ========================================
# 10. Agent 실행 함수
# ========================================
async def run_agent(df: pd.DataFrame) -> tuple:
    # 체크포인트 복원
    all_results = load_checkpoint()
    processed_ids = {r['video_id'] for r in all_results}

    # 미처리 썸네일 필터링
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
        output_type=ThumbnailAnalysis,  
        system_prompt=system_prompt,
        retries=3
    )

    settings = GoogleModelSettings(temperature=0.3)
    pbar = tqdm(total=len(pending), desc="썸네일 분석")

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

    print(f"전체 썸네일 수: {len(df)}개")
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
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 저장 폴더 없으면 자동 생성
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n분석 결과 저장 완료: {output_path}")

    print(f"  입력 토큰: {total_tokens['input']:,} / 출력 토큰: {total_tokens['output']:,}")
    if stats['success'] > 0:
        print(f"  1건 평균: 입력 {total_tokens['input'] / stats['success']:.0f} / 출력 {total_tokens['output'] / stats['success']:.0f} tokens")


# 해당 파일을 직접 호출해서 실행하면 실행되도록 하는 함수
if __name__ == '__main__':
    asyncio.run(main())