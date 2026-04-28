"""
기업 유튜브 채널 영상의 인스트림 광고 여부를 분류하는 Agent (instream_classifier.py)

[실행 방법]
python instream_classifier.py <csv_path> [--concurrent 3] [--delay 10] [--limit N]

[예시]
python instream_classifier.py ../data/results/filtered_food_with_redirect.csv
python instream_classifier.py ../data/results/filtered_food_with_redirect.csv --concurrent 2 --delay 15 --limit 10

[의존 패키지]
pip install pydantic-ai google-genai youtube-transcript-api httpx pandas tqdm python-dotenv
"""

# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import re
import sys
import json
import asyncio
import argparse
import httpx

import pandas as pd

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Literal

# Pydantic: 데이터 검증 및 스키마 정의
from pydantic import BaseModel, Field

# pydantic_ai: Vertex AI Gemini Agent 프레임워크
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

# YouTube 자막 수집 (v1.2.x 호환)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# Gemini Vision (썸네일 분석용)
import google.genai as genai
from google.genai import types as genai_types


# ========================================
# 2. 환경변수 로드
# ========================================
load_dotenv(dotenv_path=".env", override=True)

GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "global")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

# 환경변수 유효성 검사
if not GCP_PROJECT:
    print("❌ GOOGLE_CLOUD_PROJECT 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

print(f"✅ GCP 프로젝트: {GCP_PROJECT}")
print(f"✅ 사용 모델: {GEMINI_MODEL} / 리전: {GCP_LOCATION}")

# 모듈 레벨에서 Provider/Model 초기화 (run_agent 안에서 매번 생성할 필요 없이 재사용)
provider = GoogleProvider(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)
model_id = GoogleModel(GEMINI_MODEL, provider=provider)


# ========================================
# 3. CLI 인수 파싱
# ========================================
parser = argparse.ArgumentParser(description="유튜브 인스트림 광고영상 소스 분류 Agent")

# 필수 인수: CSV 파일 경로
parser.add_argument("csv_path", help="분석할 CSV 파일 경로")

# 선택 인수: 동시 실행 수 (기본값 3)
parser.add_argument("--concurrent", type=int, default=3,
                    help="동시 API 호출 수 (기본값: 3)")

# 선택 인수: 영상 간 대기 시간 (기본값 10초) - 429 방지
parser.add_argument("--delay", type=int, default=10,
                    help="영상 간 대기 시간(초) (기본값: 10)")

# 선택 인수: 처리할 최대 영상 수
parser.add_argument("--limit", type=int, default=None,
                    help="처리할 최대 영상 수 (기본값: 전체)")

args = parser.parse_args()


# ========================================
# 4. 실행 상수 정의
# ========================================
csv_stem = Path(args.csv_path).stem  # 파일명 (확장자 제외)

# --limit 옵션 사용 시 체크포인트 파일명을 구분해서 전체 실행과 충돌 방지
suffix          = f"_limit{args.limit}" if args.limit else ""
CHECKPOINT_FILE = Path(f"./outputs/checkpoint_{csv_stem}{suffix}.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)  # outputs/ 폴더 없으면 생성

CHECKPOINT_EVERY = 5           # N건마다 중간 저장
MAX_RETRIES      = 4           # 최대 재시도 횟수
BASE_DELAY       = args.delay  # 기본 대기 시간(초)
MAX_DELAY        = 60          # 최대 대기 시간(초) - 지수 백오프 상한
MAX_CONCURRENT   = args.concurrent

# asyncio 객체는 이벤트 루프 안에서 생성해야 함
# 파이썬 3.10+에서 모듈 레벨에서 생성하면 경고 발생 → run_agent() 내부에서 초기화
sem  = None
lock = None


# ========================================
# 5. Pydantic 출력 스키마 정의
# ========================================

"""
응답 스키마 구조
====================================
VideoInstreamClassification (BaseModel)
├── video_id        (str)     : 영상 고유 ID (입력 데이터에서 그대로 가져옴)
├── adverdict       (Literal) : 합산 점수 2점 이상 → High, 1점 이하 → Low
├── instream_type   (Literal) : 광고 유형 분류
│                               "건너뛰기가능" / "건너뛰기불가" / "범퍼광고" / "None"
├── reason          (str)     : 분류 근거 2~3문장 (10자 이상)
└── url             (str)     : get_video_metadata 결과의 url 필드값 그대로 사용
"""

class VideoInstreamClassification(BaseModel):
    """Agent의 최종 판정 결과 스키마"""

    video_id: str = Field(
        description="영상의 고유 ID. 입력 데이터에서 그대로 가져온다."
    )

    adverdict: Literal["High", "Low"] = Field(
        description=(
            "인스트림 광고 가능성 판정 결과. "
            "합산 점수가 2점 이상이면 High, 1점 이하면 Low로 분류한다."
        )
    )

    instream_type: Literal["건너뛰기가능", "건너뛰기불가", "범퍼광고", "None"] = Field(
        description=(
            "어떤 형태의 인스트림 광고로 사용되었는지 나타낸다. "
            "건너뛰기 가능한 인스트림 광고에 부합하면 건너뛰기가능으로 분류한다. "
            "건너뛰기 불가능한 인스트림 광고에 부합하면 건너뛰기불가로 분류한다. "
            "6초 이하의 범퍼 광고에 부합하면 범퍼광고로 분류한다. "
            "인스트림 광고 형태에 적합하지 않으면 None으로 분류한다."
        )
    )

    reason: str = Field(
        description=(
            "분류 근거를 2~3문장으로 요약. "
            "구체적인 시그널(제목, 자막, 썸네일)을 반드시 인용할 것."
        ),
        min_length=10,
    )

    url: str = Field(
        description=(
            "영상의 URL. "
            "get_video_metadata 결과에 포함된 url 필드값을 그대로 사용할 것. "
            "절대로 URL을 직접 조합하거나 추론하지 말 것."
        )
    )


# ========================================
# 6. Tool 함수 정의
# ========================================

# ── Tool 1: 메타데이터 조회 ──────────────────────────────────
# CSV에 이미 메타데이터가 있으므로 YouTube API를 호출하지 않고
# 전역 딕셔너리(VIDEO_METADATA_STORE)에서 바로 반환
VIDEO_METADATA_STORE: dict[str, dict] = {}

def get_video_metadata(video_id: str) -> dict:
    """
    CSV에서 미리 로드한 영상 메타데이터를 반환합니다.
    YouTube API를 호출하지 않고 로컬 데이터를 사용합니다.

    Parameters
    ----------
    video_id : str  YouTube 영상 고유 ID

    Returns
    -------
    dict : 제목, 설명, 길이(초), 태그, 채널명, 썸네일 URL, final_url 등
    """
    data = VIDEO_METADATA_STORE.get(video_id)
    if not data:
        return {"error": f"video_id '{video_id}' 를 CSV에서 찾을 수 없습니다."}
    return data


# ── Tool 2: 자막 수집 ────────────────────────────────────────
# youtube-transcript-api가 동기 라이브러리라 이벤트 루프 안에서 직접 호출하면
# 네트워크 대기 시간 동안 루프 전체가 블로킹됨
# → 내부 로직을 클로저로 감싸 asyncio.to_thread()로 별도 스레드에서 실행
async def get_video_transcript(video_id: str, max_chars: int = 1500) -> dict:
    """
    YouTube 영상 자막을 수집합니다.
    한국어 우선, 없으면 영어, 없으면 자동생성 자막 순으로 시도합니다.
    youtube-transcript-api v1.2.x 호환.

    Parameters
    ----------
    video_id  : str  YouTube 영상 고유 ID
    max_chars : int  반환할 최대 글자 수 (토큰 절약용)

    Returns
    -------
    dict : transcript, language, is_auto_generated, available, error
    """
    def _fetch():
        ytt = YouTubeTranscriptApi()

        # 선호 언어 순서로 시도
        for lang in ["ko", "en"]:
            try:
                fetched   = ytt.fetch(video_id, languages=[lang])
                full_text = " ".join(s.text for s in fetched.snippets)
                return {
                    "transcript"        : full_text[:max_chars],
                    "language"          : fetched.language_code,
                    "is_auto_generated" : fetched.is_generated,
                    "available"         : True,
                    "error"             : None,
                }
            except Exception:
                pass

        # 언어 지정 없이 아무 자막이나 시도
        try:
            fetched   = ytt.fetch(video_id)
            full_text = " ".join(s.text for s in fetched.snippets)
            return {
                "transcript"        : full_text[:max_chars],
                "language"          : fetched.language_code,
                "is_auto_generated" : fetched.is_generated,
                "available"         : True,
                "error"             : None,
            }
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            return {
                "transcript": "", "language": "", "is_auto_generated": False,
                "available": False, "error": "자막 없음",
            }
        except Exception as e:
            return {
                "transcript": "", "language": "", "is_auto_generated": False,
                "available": False, "error": str(e),
            }

    # asyncio.to_thread(): 동기 함수를 별도 스레드에서 실행해 이벤트 루프 블로킹 방지
    return await asyncio.to_thread(_fetch)


# ── Tool 3: 썸네일 Vision 분석 ──────────────────────────────
VISION_PROMPT = """\
이 유튜브 썸네일 이미지를 분석해서 아래 JSON만 출력하세요. 다른 텍스트 없이 JSON만.

{
  "description": "이미지 전반 설명 (2문장 이내)",
  "brand_detected": true 또는 false,
  "product_visible": true 또는 false,
  "price_or_cta": true 또는 false,
  "ad_visual_score": 0~10 정수,
  "reason": "점수 판단 이유 한 문장"
}

판단 기준:
- brand_detected : 브랜드 로고나 기업명이 화면에 강조되어 있으면 true
- product_visible: 제품이 화면 중심을 클로즈업으로 차지하면 true
- price_or_cta   : 가격·할인율·"지금 구매" 같은 행동 유도 문구가 보이면 true
- ad_visual_score: 0(일반 콘텐츠) ~ 10(명확한 광고)\
"""

# genai.Client 싱글턴: 매 호출마다 생성하면 인증·세션 초기화 오버헤드가 누적됨
# 최초 호출 시에만 생성하고 이후 재사용
_genai_client: genai.Client | None = None

def _get_genai_client() -> genai.Client:
    """genai.Client 싱글턴 반환. 최초 호출 시에만 생성합니다."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            http_options=genai_types.HttpOptions(api_version="v1beta1"),
        )
    return _genai_client


# 이미지 다운로드(httpx)와 Vision API 호출 모두 블로킹 I/O 발생
# → 이미지 다운로드는 async httpx.AsyncClient로 대체
# → Vision API 호출은 동기 라이브러리이므로 asyncio.to_thread()로 실행
async def analyze_thumbnail(thumbnail_url: str) -> dict:
    """
    썸네일 이미지를 Gemini Vision으로 분석하여
    광고성 시각 요소(브랜드 로고, 제품 클로즈업, CTA 텍스트 등)를 추출합니다.

    Parameters
    ----------
    thumbnail_url : str  YouTube 썸네일 이미지 URL

    Returns
    -------
    dict : description, brand_detected, product_visible,
           price_or_cta, ad_visual_score, reason, error
    """
    empty = {
        "description": "", "brand_detected": False, "product_visible": False,
        "price_or_cta": False, "ad_visual_score": 0, "reason": "",
    }

    # 이미지 다운로드: 비동기 httpx 사용 (이벤트 루프 블로킹 방지)
    try:
        async with httpx.AsyncClient() as client:
            resp      = await client.get(thumbnail_url, timeout=10, follow_redirects=True)
            img_bytes = resp.content
    except Exception as e:
        return {**empty, "error": f"이미지 다운로드 실패: {e}"}

    # Vision API 호출: google-genai가 동기 라이브러리이므로 to_thread()로 실행
    def _vision():
        c = _get_genai_client()
        return c.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                VISION_PROMPT,
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )

    try:
        resp  = await asyncio.to_thread(_vision)
        raw   = resp.text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {**empty, "error": f"JSON 파싱 실패: {raw[:200]}"}
        return {**json.loads(match.group()), "error": None}
    except Exception as e:
        return {**empty, "error": f"Vision 분석 실패: {e}"}


# ========================================
# 7. 체크포인트 유틸 함수
# ========================================

def save_checkpoint(results: list) -> None:
    """중간 결과를 JSON 파일로 저장합니다."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)


def load_checkpoint() -> list:
    """이전 체크포인트를 복원합니다. 없으면 빈 리스트 반환."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8-sig") as f:
            results = json.load(f)
        print(f"📂 체크포인트 복원: {len(results)}건 (이미 처리된 항목 스킵)")
        return results
    return []


# ========================================
# 8. 시스템 프롬프트 정의
# ========================================
SYSTEM_PROMPT = """\
당신은 기업 유튜브 채널 영상이 인스트림 광고(유튜브 영상 재생 전/중간에 노출되는 광고)로
제작되거나 실제 집행되었는지를 판별하는 전문 분석가입니다.

[인스트림 광고의 정의]
유튜브 영상 재생 전(Pre-roll) 또는 중간(Mid-roll)에 삽입되는 광고 영상.
- 건너뛸 수 있는 광고: 5초 후 스킵 가능, 보통 15초~3분
- 건너뛸 수 없는 광고: 15~60초 고정
- 범퍼 광고: 6초 이하
- 일반 채널 콘텐츠처럼 보이지만 광고로 집행된 경우도 있으니 주의할 것

[분석 절차 - 반드시 이 순서를 따를 것]
Step 1. get_video_metadata(video_id) 호출
        → 제목·설명·길이(duration_seconds)·태그·채널명·썸네일 URL·verdict·url 확인

Step 2. get_video_transcript(video_id) 호출
        → 자막에서 CTA 문구, 광고 고지 멘트, 브랜드 언급 패턴 분석

Step 3. analyze_thumbnail(thumbnail_url) 호출
        → 썸네일 시각 요소 분석 (브랜드 로고, 제품 클로즈업, CTA 텍스트)

Step 4. 모든 시그널 종합 → 최종 판정 출력

[채점 기준]
강한 시그널 (+2점):
    A. 영상 길이(duration_seconds) 180초 이하
    B. description에 "유료광고" / "협찬" / "#광고" / "sponsored" / "광고를 포함" 명시
    C. 자막 앞 20% 구간에 브랜드명 + 제품 소개 집중 등장
    D. "지금 구매" / "링크 클릭" / "아래 링크" / "지금 바로" 등 CTA 문구 존재
    E. 썸네일 ad_visual_score 7점 이상

보통 시그널 (+1점):
    F. 영상 전체가 단일 제품/서비스 홍보에 집중
    G. 할인·쿠폰·이벤트·기간 한정 정보 포함
    H. 자막 또는 description에 가격 정보 포함
    I. 썸네일에 브랜드 로고 또는 제품 클로즈업 존재

감점 시그널 (-1점):
    J. 영상 길이(duration_seconds) 300초(5분) 초과
    K. 브이로그 / 레시피 / 리뷰 / 스토리텔링 형태의 일반 콘텐츠 구성
    L. 조회수 10만 이상이며 반응이 일반 콘텐츠 수준

[최종 판정 기준]
합산 점수 2점 이상 → High (인스트림 광고 가능성 있음)
합산 점수 1점 이하 → Low  (인스트림 광고 가능성 낮음)
"""


# ========================================
# 9. 단건 처리 함수 (지수 백오프 포함)
# ========================================

async def analyze_one(
    agent: Agent,
    row: dict,
    settings: GoogleModelSettings,
    stats: dict,
    all_results: list,
    total_tokens: dict,
    pbar: tqdm,
) -> None:
    """
    Semaphore로 동시 실행을 제한하면서 영상 1개를 분석합니다.
    실패 시 지수 백오프로 최대 MAX_RETRIES회 재시도합니다.

    Parameters
    ----------
    agent        : pydantic_ai Agent 인스턴스
    row          : 영상 메타데이터 딕셔너리 (CSV 1행)
    settings     : Gemini 모델 설정 (temperature 등)
    stats        : 성공/실패 카운터 (공유 상태)
    all_results  : 전체 결과 리스트 (공유 상태)
    total_tokens : 입출력 토큰 집계 (공유 상태)
    pbar         : tqdm 진행률 바
    """
    async with sem:
        # video_id만 전달 → get_video_metadata Tool이 CSV에서 나머지 데이터 조회
        prompt = f"다음 영상을 분석해주세요.\nvideo_id: {row['video_id']}"

        for attempt in range(MAX_RETRIES):
            try:
                result = await agent.run(prompt, model_settings=settings)

                # 결과 추출
                output = result.output.model_dump()

                # Agent가 URL을 잘못 조합하는 경우를 대비해
                # CSV 원본의 final_url로 강제 덮어쓰기
                original_url = VIDEO_METADATA_STORE.get(row["video_id"], {}).get("url", "")
                if original_url:
                    output["url"] = original_url

                # 토큰 집계 및 결과 저장 (Lock으로 동시 접근 방지)
                usage = result.usage()
                async with lock:
                    total_tokens["input"]  += usage.input_tokens  or 0
                    total_tokens["output"] += usage.output_tokens or 0
                    all_results.append(output)
                    stats["success"] += 1

                pbar.update(1)
                pbar.set_postfix(
                    성공=stats["success"],
                    실패=stats["fail"],
                    입력토큰=total_tokens["input"],
                    출력토큰=total_tokens["output"],
                )

                # N건마다 체크포인트 저장
                if stats["success"] % CHECKPOINT_EVERY == 0:
                    save_checkpoint(all_results)
                    print(f"\n💾 체크포인트 저장: {stats['success']}건 완료")

                # 429 방지: 영상 처리 후 대기
                await asyncio.sleep(BASE_DELAY)
                return

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "429" in error_msg or "rate" in error_msg.lower()

                if attempt < MAX_RETRIES - 1:
                    # 지수 백오프: 2^attempt 배수로 대기 시간 증가
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    if is_rate_limit:
                        delay = min(delay * 2, MAX_DELAY)  # Rate limit이면 2배 더 대기
                    print(f"\n⚠️ [{row['video_id']}] 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기) | {error_msg[:80]}")
                    await asyncio.sleep(delay)
                else:
                    print(f"\n❌ [최종 실패] {row['video_id']} | {error_msg[:100]}")
                    async with lock:
                        stats["fail"] += 1
                    pbar.update(1)
                    pbar.set_postfix(성공=stats["success"], 실패=stats["fail"])


# ========================================
# 10. 배치 실행 함수
# ========================================

async def run_agent(df: pd.DataFrame) -> tuple[list, dict, dict]:
    """
    전체 영상을 비동기 병렬로 처리합니다.
    체크포인트가 있으면 이미 처리된 항목을 스킵합니다.

    Parameters
    ----------
    df : pd.DataFrame  CSV에서 로드한 전체 영상 데이터

    Returns
    -------
    tuple : (all_results, stats, total_tokens)
    """
    # asyncio 객체는 이벤트 루프 안에서 생성해야 하므로 여기서 초기화
    global sem, lock
    sem  = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()

    # 체크포인트 복원 (이미 처리된 항목 스킵)
    all_results   = load_checkpoint()
    processed_ids = {r["video_id"] for r in all_results}

    # 미처리 영상만 필터링
    pending = [row for _, row in df.iterrows() if row["video_id"] not in processed_ids]

    # --limit 옵션 적용
    if args.limit:
        pending = pending[:args.limit]

    if processed_ids:
        print(f"⏭️  스킵: {len(processed_ids)}개 (이미 처리됨)")
    print(f"📋 처리 대상: {len(pending)}개")
    print(f"⚡ 동시 실행: 최대 {MAX_CONCURRENT}개 | 재시도: 최대 {MAX_RETRIES}회 | 대기: {BASE_DELAY}초")
    print("=" * 60)

    # success는 새로 처리한 건수만 카운팅 (체크포인트 복원 건수 제외)
    # 기존에 복원 건수를 초기값으로 쓰면 CHECKPOINT_EVERY 배수 시 즉시 저장 트리거 문제 발생
    stats        = {"success": 0, "fail": 0}
    total_tokens = {"input": 0, "output": 0}

    # Agent 초기화
    agent = Agent(
        model_id,
        output_type=VideoInstreamClassification,
        system_prompt=SYSTEM_PROMPT,
        retries=3,
        # Tool 1: CSV 로컬 데이터에서 메타데이터 조회 (YouTube API 호출 없음)
        # Tool 2: youtube-transcript-api로 자막 수집
        # Tool 3: Gemini Vision으로 썸네일 광고 시각 요소 분석
        tools=[get_video_metadata, get_video_transcript, analyze_thumbnail],
    )

    settings = GoogleModelSettings(temperature=0.3)
    pbar     = tqdm(total=len(pending), desc="인스트림 영상 분류")

    # 모든 영상을 비동기 병렬 처리
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
# 11. main 함수
# ========================================

async def main():
    global VIDEO_METADATA_STORE

    # CSV 로드
    df = pd.read_csv(args.csv_path, encoding="utf-8-sig")
    print(f"\n📁 CSV 로드 완료: {len(df)}개 영상")
    print("=" * 60)

    # ── duration 변환 함수 ───────────────────────────────────
    def parse_duration(raw: str) -> int:
        """
        ISO 8601 duration 문자열을 초(int)로 변환합니다.
        예: 'PT3M20S' → 200, 'PT1H2M3S' → 3723
        변환 실패 시 0 반환.
        """
        if not raw:
            return 0
        try:
            match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(raw))
            if not match:
                return 0
            h, m, s = (int(x or 0) for x in match.groups())
            return h * 3600 + m * 60 + s
        except Exception:
            return 0

    # ── VIDEO_METADATA_STORE 구축 ────────────────────────────
    # CSV의 각 행을 video_id 키로 딕셔너리에 저장
    # get_video_metadata Tool이 이 데이터를 직접 조회 (YouTube API 호출 없음)
    # duration은 ISO 8601 → 초 단위로 변환 (채점 기준 A, J가 초 단위 참조)
    # has_paid_product_placement 제거: PPL 포함 여부는 인스트림 광고 집행과 개념이 달라 혼란 방지
    VIDEO_METADATA_STORE = {
        row["video_id"]: {
            "video_id"        : row.get("video_id", ""),
            "title"           : row.get("title", ""),
            "channel_title"   : row.get("channel_title", ""),
            "description"     : str(row.get("description", ""))[:1000],
            "tags"            : row.get("tags", ""),
            "duration_seconds": parse_duration(row.get("duration", "")),
            "view_count"      : row.get("view_count", 0),
            "like_count"      : row.get("like_count", 0),
            "thumbnail_url"   : row.get("thumbnail", ""),
            "upload_date"     : row.get("upload_date", ""),
            "verdict"         : row.get("verdict", ""),  
            "url"             : row.get("final_url", ""), # CSV의 final_url 그대로 저장
        }
        for _, row in df.iterrows()
    }
    print(f"✅ 메타데이터 스토어 구축 완료: {len(VIDEO_METADATA_STORE)}개")

    # Agent 실행
    all_results, stats, total_tokens = await run_agent(df)

    # ── 결과 요약 출력 ────────────────────────────────────────
    print()
    print("=" * 60)
    print("✅ 배치 처리 완료")
    print(f"   성공: {stats['success']}개 | 실패: {stats['fail']}개")
    total = max(stats["success"] + stats["fail"], 1)
    print(f"   성공률: {stats['success'] / total * 100:.1f}%")
    print(f"   체크포인트: {CHECKPOINT_FILE}")
    print("=" * 60)

    # ── 결과 CSV 저장 ─────────────────────────────────────────
    df_result   = pd.DataFrame(all_results)
    output_path = Path(f"./outputs/classified_{csv_stem}{suffix}.csv")
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n💾 저장 완료: {output_path}")
    print(f"   컬럼: {list(df_result.columns)}")
    print(f"   입력 토큰: {total_tokens['input']:,} / 출력 토큰: {total_tokens['output']:,}")
    if stats["success"] > 0:
        print(f"   1건 평균: 입력 {total_tokens['input'] / stats['success']:.0f} / "
              f"출력 {total_tokens['output'] / stats['success']:.0f} tokens")

    # ── adverdict 분포 출력 ───────────────────────────────────
    if not df_result.empty and "adverdict" in df_result.columns:
        print(f"\n[판정 결과 분포]")
        for verdict, count in df_result["adverdict"].value_counts().items():
            pct = count / len(df_result) * 100
            print(f"   {verdict}: {count}개 ({pct:.1f}%)")

    # ── 상세 결과 출력 (url, instream_type 포함) ─────────────
    print(f"\n[개별 결과]")
    for i, video in enumerate(all_results, start=1):
        print(f"{i:3d}. [{video['adverdict']}] [{video.get('instream_type', '-')}] {video['video_id']}")
        print(f"      URL: {video.get('url', '')}")
        print(f"      {video['reason'][:120]}")


# ========================================
# 12. 진입점
# ========================================
if __name__ == "__main__":
    # asyncio.run(): 이벤트 루프 생성 → main() 실행 → 루프 종료
    # 프로그램 전체에서 딱 한 번만 호출
    asyncio.run(main())
