"""
유튜브 댓글 스팸/홍보성 필터링 Agent + 이벤트 분류 + CTA 분석 (comment_filter_agent_v0.3.py)

[실행 방법] #상대경로
python comment_filter_agent.py <csv_path> [--concurrent 3] [--delay 5] [--limit N]

[예시]
python comment_filter_agent.py ../data/comments.csv
python comment_filter_agent.py ../data/comments.csv --concurrent 2 --delay 8 --limit 50

[의존 패키지]
pip install pydantic-ai google-genai pandas tqdm python-dotenv

[✅ 주요 개선사항 - v0.3]
────────────────────────────────────────────
1. is_event 필드 추가 (이벤트 참여 분류)
2. 이벤트 패턴 감지 함수 추가
3. 시스템 프롬프트에 이벤트 분류 가이드 추가
"""

# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import sys
import json
import asyncio
import argparse
import re
from collections import Counter

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


# ========================================
# 2. 환경변수 로드
# ========================================
load_dotenv(dotenv_path=".env", override=True)

GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "global")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

if not GCP_PROJECT:
    print("❌ GOOGLE_CLOUD_PROJECT 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

print(f"✅ GCP 프로젝트: {GCP_PROJECT}")
print(f"✅ 사용 모델: {GEMINI_MODEL} / 리전: {GCP_LOCATION}")

provider = GoogleProvider(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)
model_id = GoogleModel(GEMINI_MODEL, provider=provider)


# ========================================
# 3. CLI 인수 파싱
# ========================================
parser = argparse.ArgumentParser(description="유튜브 댓글 스팸/홍보성 필터링 + 이벤트 분류 + CTA 분석 Agent")

parser.add_argument("csv_path",         help="분석할 CSV 파일 경로")
parser.add_argument("--concurrent",     type=int, default=3,
                    help="동시 API 호출 수 (기본값: 3)")
parser.add_argument("--delay",          type=int, default=5,
                    help="댓글 간 대기 시간(초) (기본값: 5)")
parser.add_argument("--limit",          type=int, default=None,
                    help="처리할 최대 댓글 수 (기본값: 전체)")
parser.add_argument("--batch-size",     type=int, default=10,
                    help="한 번에 묶어서 보낼 댓글 수 (기본값: 10, 비용 절감)")

args = parser.parse_args()


# ========================================
# 4. 실행 상수 정의
# ========================================
csv_stem = Path(args.csv_path).stem

suffix          = f"_limit{args.limit}" if args.limit else ""
CHECKPOINT_FILE = Path(f"./outputs/checkpoint_{csv_stem}{suffix}.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

CHECKPOINT_EVERY = 5
MAX_RETRIES      = 4
BASE_DELAY       = args.delay
MAX_DELAY        = 60
MAX_CONCURRENT   = args.concurrent
BATCH_SIZE       = args.batch_size   # 댓글은 묶음 처리 비용 효율적

sem  = None
lock = None


# ========================================
# 5. Pydantic 출력 스키마 정의
# ========================================

"""
응답 스키마 구조 
====================================
CommentFilterResult (BaseModel)
├── comment_id      (str)     : 댓글 고유 ID
├── video_id        (str)     : 영상 ID (입력값 그대로)
├── label           (Literal) : 분류 레이블
│                               "Spam"          - 스팸/홍보성 댓글
│                               "Inappropriate" - 음란성/부적절 댓글 
│                               "Normal"        - 일반 댓글
├── spam_score      (int)     : 스팸 가능성 점수 0~10
├── spam_type       (Literal) : 스팸 세부 유형
│                               "홍보/광고"
│                               "도배"
│                               "어뷰징링크"
│                               "무관한내용"
│                               "음란성"        
│                               "부적절_성적표현" 
│                               "성적_암시"     
│                               "None"
├── detected_signals (List[str]) : 탐지된 신호들 
│                                  예) ["성인물_언급", "도배_패턴", "부적절_표현"]
├── reason          (str)     : 분류 근거 1~2문장
├── confidence      (float)   : 판정 신뢰도 0.0~1.0
│                               0.0 = 확신 없음, 1.0 = 매우 확신
├── recommend_action (Literal) : 권장 조치 
│                                 "BLOCK"  - 즉시 차단 (음란성 우선)
│                                 "FILTER" - 필터링 (일반 스팸)
│                                 "ALLOW"  - 허용 (정상)
└── is_event       (bool)     : 이벤트 참여 댓글 여부
                                 True  = 우리 채널 공식 이벤트 참여
                                 False = 일반 댓글 또는 외부 이벤트
"""

class CommentFilterResult(BaseModel):
    """Agent의 최종 댓글 판정 결과 스키마"""

    comment_id: str = Field(
        description="댓글 고유 ID. 입력 데이터에서 그대로 가져온다."
    )

    video_id: str = Field(
        description="댓글이 달린 영상 ID. 입력 데이터에서 그대로 가져온다."
    )

    label: Literal["Spam", "Inappropriate", "Normal"] = Field(  
        description=(
            "댓글 분류 레이블. "
            "스팸 점수 5~7점이면 Spam, 8~10점이면 Inappropriate, 4점 이하면 Normal로 분류한다. "  
            "음란성/부적절 신호 2개 이상이면 우선적으로 Inappropriate으로 판정한다."  
        )
    )

    spam_score: int = Field(
        description=(
            "스팸/홍보성/부적절성 가능성 점수. 0(완전한 일반 댓글) ~ 10(명확한 부적절/스팸). "  
            "5~7점이면 Spam, 8~10점이면 Inappropriate, 4점 이하면 Normal으로 판정한다." 
        ),
        ge=0,
        le=10,
    )

    spam_type: Literal[
        "홍보/광고", 
        "도배", 
        "어뷰징링크", 
        "무관한내용",
        "음란성", 
        "부적절_성적표현",  
        "성적_암시",  
        "None"
    ] = Field(
        description=(
            "스팸/부적절 댓글의 세부 유형. "
            "제품·서비스 홍보나 광고성 문구가 있으면 홍보/광고, "
            "동일하거나 유사한 내용을 반복하면 도배, "
            "bit.ly 등 단축 URL이나 외부 링크를 포함하면 어뷰징링크, "
            "영상과 전혀 무관한 내용이면 무관한내용, "
            "성인물이나 노출 콘텐츠를 직접 언급하면 음란성, "  
            "부적절한 성적 표현이 있으면 부적절_성적표현, " 
            "성적 암시나 은유 표현이 있으면 성적_암시, "  
            "해당 없으면 None."
        )
    )

    detected_signals: list[str] = Field(  
        description=(
            "탐지된 스팸/부적절 신호 리스트. "
            "어떤 신호들이 감지되었는지 추적 가능하게 함. "
            "예) ['성인물_언급', '도배_패턴', '영상무관_성적표현']"
        ),
        default_factory=list,
    )

    reason: str = Field(
        description=(
            "분류 근거를 1~2문장으로 요약. "
            "댓글 텍스트의 구체적인 특성과 판정 이유를 명시할 것."  
        ),
        min_length=5,
    )
    
    confidence: float = Field(
        description=(
            "판정 신뢰도. 0.0(확신 없음) ~ 1.0(매우 확신). "
            "명확한 스팸/부적절 신호 2개 이상이면 0.85 이상, "  
            "신호 1개이면 0.5~0.75, "  
            "애매한 경우는 0.4~0.6으로 설정한다." 
        ),
        ge=0.0,
        le=1.0,
    )

    recommend_action: Literal["BLOCK", "FILTER", "ALLOW"] = Field( 
        description=(
            "권장 조치. "
            "음란성/부적절 댓글이면 BLOCK(즉시 차단), "
            "일반 스팸이면 FILTER(필터링), "
            "정상 댓글이면 ALLOW(허용)."
        )
    )

    # 이벤트 참여 댓글 분류
    is_event: bool = Field(
        description=(
            "이벤트 참여 댓글 여부 (우리 채널 공식 이벤트만). "
            "True: 우리 채널의 공식 이벤트 참여 댓글 (응모, 참여, 추첨, 경품, 상품 등) "
            "False: 일반 댓글 또는 외부 이벤트, 자체 프로모션 댓글. "
            "다음 패턴을 포함하면 True로 분류: "
            "'응모', '참여', '이벤트', '추첨', '상품', '경품', '당첨', '선착순', '공식 이벤트' 등의 명확한 이벤트 언어. "
            "다른 채널의 이벤트, 자체 광고, 링크 있는 홍보는 False로 분류."
        ),
        default=False,
    )


class BatchFilterResult(BaseModel):
    """배치 처리 시 여러 댓글의 판정 결과를 묶는 컨테이너"""
    results: list[CommentFilterResult]


# ========================================
# 6. 헬퍼 함수 - 이벤트 패턴 감지
# ========================================

# 이벤트 패턴 감지 함수
def detect_event_pattern(text: str) -> dict:
    """
    이벤트 참여 댓글 패턴을 미리 감지합니다.
    Agent의 판정을 위한 보조 정보로 사용됩니다.
    
    Parameters
    ----------
    text : str  댓글 텍스트
    
    Returns
    -------
    dict : {
        "is_likely_event": bool,          # 이벤트 참여 댓글 가능성
        "detected_keywords": list[str],   # 탐지된 이벤트 키워드
        "confidence": float               # 신뢰도 (0.0~1.0)
    }
    
    판정 기준
    ─────────────
    높음 신뢰도 (0.8+):
    - "응모", "참여", "이벤트", "추첨", "상품", "경품", "선착순" 포함
    - "댓글로 참여", "댓글 이벤트" 정확 매칭
    
    중간 신뢰도 (0.5~0.8):
    - "당첨", "공식 이벤트" 포함
    - 여러 개의 약한 신호 조합
    
    낮음 신뢰도 (0.3~0.5):
    - "구독", "좋아요" 같은 약한 신호
    - 문맥상 이벤트일 수 있지만 명확하지 않음
    """
    text_lower = text.lower()
    
    # ▶ 강한 이벤트 키워드 (높은 신뢰도)
    strong_keywords = [
        "응모", "참여", "이벤트", "추첨", "경품", "상품",
        "당첨", "선착순", "공식 이벤트", "우리의 이벤트",
        "댓글 이벤트", "댓글로 참여"
    ]
    
    # ▶ 구체적 이벤트 패턴 (정확 매칭)
    event_patterns = [
        r"댓글로\s?참여",
        r"댓글\s?이벤트",
        r"댓글\s?응모",
        r"이벤트\s?참여",
        r"응모\s?합니다",
        r"참여\s?합니다",
    ]
    
    # ▶ 약한 신호 (낮은 신뢰도)
    weak_keywords = ["구독", "좋아요", "알림", "구독 부탁"]
    
    detected_keywords = []
    
    # 강한 키워드 검사
    for keyword in strong_keywords:
        if keyword in text_lower:
            detected_keywords.append(keyword)
    
    # 구체적 패턴 검사
    for pattern in event_patterns:
        if re.search(pattern, text_lower):
            match = re.findall(pattern, text_lower)[0]
            detected_keywords.append(match)
    
    # 약한 신호 검사
    weak_detected = []
    for keyword in weak_keywords:
        if keyword in text_lower:
            weak_detected.append(keyword)
    
    # ── 신뢰도 계산 ──────────────────────────────────
    if len(detected_keywords) >= 2:
        confidence = 0.9
    elif len(detected_keywords) == 1:
        confidence = 0.75
    elif len(weak_detected) >= 2:
        confidence = 0.5
    elif len(weak_detected) == 1:
        confidence = 0.3
    else:
        confidence = 0.0
    
    # ── 제외 패턴 (이벤트가 아닌 경우) ──────────────
    # 홍보/광고성 링크, 외부 이벤트 언급 등
    exclude_patterns = [
        r"http[s]?://",  # URL 포함
        r"bit\.ly",       # 단축 URL
        r"tinyurl",       # 단축 URL
        r"외부\s?이벤트", # 외부 이벤트
        r"다른\s?채널",   # 다른 채널
        r"다른\s?이벤트", # 다른 이벤트
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, text_lower):
            confidence = max(0, confidence - 0.3)
    
    is_likely_event = len(detected_keywords) > 0 and confidence >= 0.5
    
    return {
        "is_likely_event": is_likely_event,
        "detected_keywords": detected_keywords,
        "confidence": confidence
    }


# ========================================
# 7. Tool 함수 정의
# ========================================

# ── 댓글 메타데이터 스토어 ─────────────────────────────────
# CSV에서 미리 로드, Tool을 통해 Agent가 조회
COMMENT_STORE: dict[str, dict] = {}

# 원본 댓글 데이터 병합 함수
def merge_original_comments(df_result: pd.DataFrame, comment_store: dict) -> pd.DataFrame:
    """
    분석 결과와 원본 댓글 정보를 병합합니다.
    
    Parameters
    ----------
    df_result : pd.DataFrame  분석 결과 데이터프레임
    comment_store : dict      원본 댓글 저장소
    
    Returns
    -------
    pd.DataFrame  원본 정보가 추가된 결과 데이터프레임
    """
    original_data = []
    for _, row in df_result.iterrows():
        comment_id = row["comment_id"]
        original_info = comment_store.get(comment_id, {})
        
        # 원본 정보 추가
        row_dict = row.to_dict()
        row_dict["comment_text"] = original_info.get("comment_text", "")
        row_dict["author_name"] = original_info.get("author_name", "")
        row_dict["like_count"] = original_info.get("like_count", 0)
        row_dict["published_at"] = original_info.get("published_at", "")
        
        original_data.append(row_dict)
    
    return pd.DataFrame(original_data)


def get_comment_context(comment_id: str) -> dict:
    """
    CSV에서 미리 로드한 댓글 데이터를 반환합니다.

    Parameters
    ----------
    comment_id : str  댓글 고유 ID

    Returns
    -------
    dict : comment_id, video_id, comment_text, author_name,
           like_count, reply_count, published_at
    """
    data = COMMENT_STORE.get(comment_id)
    if not data:
        return {"error": f"comment_id '{comment_id}'를 CSV에서 찾을 수 없습니다."}
    return data


def get_video_comment_stats(video_id: str) -> dict:
    """
    특정 영상의 전체 댓글 통계를 반환합니다.
    도배 패턴 탐지(동일 저자의 반복 댓글)에 활용합니다.

    Parameters
    ----------
    video_id : str  영상 고유 ID

    Returns
    -------
    dict : total_comments, top_authors (댓글 수 상위 5명), avg_like_count
    """
    video_comments = [
        c for c in COMMENT_STORE.values()
        if c.get("video_id") == video_id
    ]

    if not video_comments:
        return {"error": f"video_id '{video_id}'에 해당하는 댓글이 없습니다."}

    # 저자별 댓글 수 집계
    author_counts: dict[str, int] = {}
    total_likes = 0
    for c in video_comments:
        author = c.get("author_name", "unknown")
        author_counts[author] = author_counts.get(author, 0) + 1
        total_likes += int(c.get("like_count", 0))

    top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_comments" : len(video_comments),
        "top_authors"    : [{"author": a, "count": n} for a, n in top_authors],
        "avg_like_count" : round(total_likes / len(video_comments), 2),
    }


# ========================================
# 8. 체크포인트 유틸 함수
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
# 9. 시스템 프롬프트 정의
# ========================================
SYSTEM_PROMPT = """\
당신은 유튜브 채널 댓글에서 스팸·홍보성 댓글을 탐지하는 전문 필터링 시스템입니다.

[3가지 주요 탐지 카테고리] 
1. 스팸/홍보성 댓글 (Spam) - 외부 링크, 광고성 문구
2. 음란성/부적절한 댓글 (Inappropriate) - 성인물 유도, 부적절한 성적 표현
3. 정상 댓글 (Normal) - 건전한 의견, 질문, 감상

[스팸/홍보성 댓글의 정의]
- 특정 타제품·서비스를 홍보하거나 외부 링크 유도를 목적으로 작성된 댓글 (단, 채널주/공식 계정 제외)
- 동일하거나 유사한 내용을 여러 영상에 반복 게시하는 도배성 댓글
- bit.ly, tinyurl 등 단축 URL 또는 의심스러운 외부 링크를 포함한 댓글
- 영상 내용과 전혀 무관한 홍보 문구, 연락처, SNS 계정 유도 댓글

[제외 대상]
다음 댓글은 분석하지 말고 반드시 Normal로 분류하세요:
- 채널주(영상 업로더)가 작성한 댓글
- 채널 공식 운영진/매니저 댓글
- 브랜드 공식 계정(예: LG유플러스 공식 계정)
이들은 링크나 이벤트 공지를 포함해도 광고/스팸이 아님!


[음란성/부적절한 댓글의 정의] 
부적절한 성적 표현이나 음란성 콘텐츠 유도를 목적으로 작성된 댓글입니다:

A. 성인물·노출 콘텐츠 직접 언급 (spam_score +4점) 
   - 성인 콘텐츠, 노출 이미지 등을 직접 언급하는 댓글
   - "야동", "19금", "성인물", "누드" 등 직접적 성인 콘텐츠 언급
   - "얼굴 인증", "신체 인증" 같은 노출 콘텐츠 요청
   - 성인/노출 콘텐츠로 알려진 크리에이터 명시 언급

B. 부적절한 성적 표현 (spam_score +3점) 
   - 성적 수치심을 자극하거나 호기심을 유도하는 표현
   - "수위", "노출", "섹시", "야함" 등 성적 표현으로 콘텐츠 강조
   - "생각하지 말고 봐", "봤으면 후회한다" 같은 신비주의적 유도
   - 명시적 성욕 표현 ("흥분", "도취" 등)

C. 성적 암시·은유 표현 (spam_score +2점) 
   - 일상적 단어를 성적 맥락에서 사용
   - 의료용어를 성적 의도로 오용
   - 신체 부위에 대한 성적 은유 표현
   - "미쳤음", "진짜 미쳤네" + 신체·외모 과도 강조

D. 성적 비교·추천 (spam_score +2점) 
   - 콘텐츠의 성적 매력을 강조하는 비교
   - 성인 콘텐츠를 다른 영상과 비교·추천
   - 타인과 성적 매력도 비교

E. 도배·반복 강화 (spam_score +2~3점 추가) 
   - 같은 저자가 여러 영상에 음란성 댓글 반복 게시
   - 음란성 관련 키워드 도배
   - → 도배 패턴 + 음란성 = 높은 우선순위 차단

F. 영상과 무관한 성적 댓글 (spam_score +2점) 
   - 일반 영상(요리, IT, 음악 등)에 갑자기 부적절한 성적 내용 언급
   - 영상 맥락과 무관한 성인 콘텐츠 유도

[분석 절차 - 반드시 이 순서를 따를 것]
Step 1. get_comment_context(comment_id) 호출
        → 댓글 텍스트, 작성자, 좋아요 수, 게시일 확인

Step 2. get_video_comment_stats(video_id) 호출
        → 동일 저자의 반복 댓글 패턴(도배 가능성) 확인

Step 3. 모든 시그널 종합 → 최종 판정 출력

✅이벤트 참여 댓글 분류 가이드
─────────────────────────────────────────────────────────

[이벤트 참여 댓글의 정의]
우리 채널의 공식 이벤트에 참여하는 댓글입니다:
- 공식 댓글 이벤트 참여 ("이벤트 참여합니다", "응모 완료", "댓글 이벤트 참여")
- 추첨 및 경품 이벤트 ("추첨 부탁드립니다", "경품 응모", "응모합니다")
- 선착순 이벤트 ("선착순 신청합니다", "참여합니다")
- 댓글 기반 이벤트 ("댓글로 참여하겠습니다", "댓글 응모")

[is_event 필드 판별 기준 (필수)]
True로 분류:
  - "이벤트", "응모", "참여", "추첨", "경품", "상품", "선착순", "정답" 등 명확한 이벤트 용어 포함
  - "댓글로 참여", "댓글 이벤트" 같은 구체적 패턴
  - "당첨", "공식 이벤트" 명시
  - 댓글 자체가 우리 채널 공식 이벤트에 대한 응답인 경우

False로 분류:
  - 다른 채널이나 외부 플랫폼의 이벤트 언급
  - "구독 부탁드립니다" 같은 단순 요청 (이벤트 아님)
  - URL이나 링크를 포함한 자체 프로모션
  - 일반적인 칭찬이나 감상 댓글
  - 제품 홍보나 광고성 댓글

[주의사항]
- is_event = True이고 label = "Spam" or "Inappropriate" 가능
  예) "이벤트 참여합니다. 우리 제품도 구매해요 링크: bit.ly/xxx"
  → is_event: True, label: Spam (스팸 신호 우선)
- 외부 이벤트는 False로 분류
- URL 포함 시 신뢰도 감소
- is_event는 label (스팸/부적절/정상)과 독립적으로 판정
"""

# ========================================
# 10. 배치 단위 처리 함수 (지수 백오프 포함)
# ========================================
async def analyze_batch(
    agent: Agent,
    batch: list[dict],
    settings: GoogleModelSettings,
    stats: dict,
    all_results: list,
    total_tokens: dict,
    pbar: tqdm,
) -> None:
    """
    Semaphore로 동시 실행을 제한하면서 댓글 N개를 배치로 분석합니다.
    실패 시 지수 백오프로 최대 MAX_RETRIES회 재시도합니다.

    Parameters
    ----------
    agent        : pydantic_ai Agent 인스턴스
    batch        : 댓글 딕셔너리 리스트 (최대 BATCH_SIZE개)
    settings     : Gemini 모델 설정
    stats        : 성공/실패 카운터 (공유 상태)
    all_results  : 전체 결과 리스트 (공유 상태)
    total_tokens : 입출력 토큰 집계 (공유 상태)
    pbar         : tqdm 진행률 바
    """
    async with sem:
        # 배치 내 댓글 목록을 프롬프트에 나열
        comment_lines = "\n".join(
            f"- comment_id: {c['comment_id']} | video_id: {c['video_id']}"
            for c in batch
        )
        prompt = (
            f"다음 댓글 {len(batch)}개를 순서대로 분석해주세요.\n"
            f"각 댓글에 대해 get_comment_context와 get_video_comment_stats를 호출하고 "
            f"results 리스트에 모든 결과를 담아 반환하세요.\n\n"
            f"{comment_lines}"
        )

        for attempt in range(MAX_RETRIES):
            try:
                result = await agent.run(prompt, model_settings=settings)
                batch_output = result.output  # BatchFilterResult

                usage = result.usage()
                async with lock:
                    total_tokens["input"]  += usage.input_tokens or 0
                    total_tokens["output"] += usage.output_tokens or 0

                    for item in batch_output.results:
                        row = item.model_dump()
                        # CSV 원본 video_id 강제 보정 (Agent가 잘못 추론하는 경우 방지)
                        original = COMMENT_STORE.get(item.comment_id, {})
                        if original.get("video_id"):
                            row["video_id"] = original["video_id"]
                        all_results.append(row)

                    stats["success"] += len(batch_output.results)

                pbar.update(len(batch))
                pbar.set_postfix(
                    성공=stats["success"],
                    실패=stats["fail"],
                    입력토큰=total_tokens["input"],
                    출력토큰=total_tokens["output"],
                )

                # N건마다 체크포인트 저장
                if (stats["success"] // CHECKPOINT_EVERY) > (
                    (stats["success"] - len(batch_output.results)) // CHECKPOINT_EVERY
                ):
                    save_checkpoint(all_results)
                    print(f"\n💾 체크포인트 저장: {stats['success']}건 완료")

                await asyncio.sleep(BASE_DELAY)
                return

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "429" in error_msg or "rate" in error_msg.lower()

                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    if is_rate_limit:
                        delay = min(delay * 2, MAX_DELAY)
                    ids_str = ", ".join(c["comment_id"] for c in batch[:3])
                    print(f"\n⚠️ [{ids_str}...] 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기) | {error_msg[:80]}")
                    await asyncio.sleep(delay)
                else:
                    ids_str = ", ".join(c["comment_id"] for c in batch)
                    print(f"\n❌ [최종 실패] {ids_str} | {error_msg[:100]}")
                    async with lock:
                        stats["fail"] += len(batch)
                    pbar.update(len(batch))
                    pbar.set_postfix(성공=stats["success"], 실패=stats["fail"])


async def run_agent(df: pd.DataFrame) -> tuple:
    """
    Agent를 실행합니다.
    
    Parameters
    ----------
    df : pd.DataFrame  입력 CSV 데이터프레임
    
    Returns
    -------
    tuple : (all_results, stats, total_tokens)
    """
    global sem, lock
    
    # 세마포어 초기화 (동시 요청 수 제한)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()

    # 기존 체크포인트 복원
    all_results = load_checkpoint()
    processed_ids = set(r["comment_id"] for r in all_results)

    # 미처리 댓글
    pending = [
        row.to_dict()
        for _, row in df.iterrows()
        if row["comment_id"] not in processed_ids
    ]

    if args.limit:
        pending = pending[:args.limit]

    if processed_ids:
        print(f"⏭️  스킵: {len(processed_ids)}개 (이미 처리됨)")
    print(f"📋 처리 대상: {len(pending)}개 댓글")
    print(f"⚡ 배치 크기: {BATCH_SIZE}개 | 동시 실행: 최대 {MAX_CONCURRENT}개 | "
          f"재시도: 최대 {MAX_RETRIES}회 | 대기: {BASE_DELAY}초")
    print("=" * 60)

    stats        = {"success": 0, "fail": 0}
    total_tokens = {"input": 0, "output": 0}

    # Agent 초기화
    agent = Agent(
        model_id,
        output_type=BatchFilterResult,
        system_prompt=SYSTEM_PROMPT,
        retries=3,
        # Tool 1: CSV 로컬 데이터에서 댓글 컨텍스트 조회
        # Tool 2: 영상 단위 댓글 통계 (도배 패턴 탐지)
        tools=[get_comment_context, get_video_comment_stats],
    )

    settings = GoogleModelSettings(temperature=0.1)  # 필터링은 일관성이 중요하므로 낮게 설정
    pbar     = tqdm(total=len(pending), desc="댓글 필터링")

    # 댓글을 BATCH_SIZE 단위로 나눠 배치 생성
    batches = [
        pending[i : i + BATCH_SIZE]
        for i in range(0, len(pending), BATCH_SIZE)
    ]

    tasks = [
        analyze_batch(agent, batch, settings, stats, all_results, total_tokens, pbar)
        for batch in batches
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    save_checkpoint(all_results)

    return all_results, stats, total_tokens



# ========================================
# 11. main 함수
# ========================================

async def main():
    global COMMENT_STORE

    # CSV 로드
    df = pd.read_csv(args.csv_path, encoding="utf-8-sig")
    print(f"\n📁 CSV 로드 완료: {len(df)}개 댓글")

    # ── 필수 컬럼 검증 ───────────────────────────────────────
    required_cols = {"video_id", "comment_id", "comment", "author", "likes", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ CSV에 필수 컬럼이 없습니다: {missing}")
        print(f"   현재 컬럼: {list(df.columns)}")
        sys.exit(1)

    # ── CSV 칼럼명 자동 변환 ─────────────────────────────────
    # 너의 CSV 칼럼을 코드가 사용하는 이름으로 변환
    df = df.rename(columns={
        "comment": "comment_text",
        "author": "author_name",
        "likes": "like_count",
        "date": "published_at"
    })

    print("✅ CSV 칼럼명 변환 완료: comment→comment_text, author→author_name, likes→like_count, date→published_at")
    print("=" * 60)

    # ── COMMENT_STORE 구축 ───────────────────────────────────
    # get_comment_context / get_video_comment_stats Tool이 이 데이터를 직접 조회
    COMMENT_STORE = {
        str(row["comment_id"]): {
            "comment_id"  : str(row["comment_id"]),
            "video_id"    : str(row.get("video_id", "")),
            "comment_text": str(row.get("comment_text", ""))[:500],  # 토큰 절약
            "author_name" : str(row.get("author_name", "unknown")),
            "like_count"  : int(row.get("like_count", 0) or 0),
            "reply_count" : int(row.get("reply_count", 0) or 0), 
            "published_at": str(row.get("published_at", "")),
        }
        for _, row in df.iterrows()
    }
    print(f"✅ 댓글 스토어 구축 완료: {len(COMMENT_STORE)}개")

    # ── 영상별 댓글 수 미리보기 ──────────────────────────────
    video_counts = df["video_id"].value_counts()
    print(f"📊 영상 수: {len(video_counts)}개 | "
          f"영상당 평균 댓글: {video_counts.mean():.1f}개 | "
          f"최대: {video_counts.max()}개")

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

    if not all_results:
        print("⚠️ 처리된 결과가 없습니다.")
        return

    # ── 결과 CSV 저장 ─────────────────────────────────────────
    df_result   = pd.DataFrame(all_results)
    
    # 원본 댓글 데이터 추가 병합
    df_result_with_original = merge_original_comments(df_result, COMMENT_STORE)
    
    # Normal 댓글만 따로 저장 (깨끗한 결과)
    normal_df = df_result_with_original[df_result_with_original["label"] == "Normal"]
    normal_output_path = Path(f"./outputs/filtered_cleaned_{csv_stem}{suffix}.csv")
    normal_df.to_csv(normal_output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 정상 댓글 저장: {normal_output_path}")
    print(f"   {len(normal_df)}개")
    
    # Spam + Inappropriate 댓글만 따로 저장 (위험한 댓글)
    filtered_df = df_result_with_original[df_result_with_original["label"].isin(["Spam", "Inappropriate"])]
    filtered_output_path = Path(f"./outputs/filtered_flagged_{csv_stem}{suffix}.csv")
    filtered_df.to_csv(filtered_output_path, index=False, encoding="utf-8-sig")
    print(f"💾 플래그된 댓글 저장: {filtered_output_path}")
    print(f"   {len(filtered_df)}개 (Spam + Inappropriate)")
    
    # 이벤트 참여 댓글만 따로 저장
    if "is_event" in df_result_with_original.columns:
        event_df = df_result_with_original[df_result_with_original["is_event"]]
        if not event_df.empty:
            event_output_path = Path(f"./outputs/filtered_event_participants_{csv_stem}{suffix}.csv")
            event_df.to_csv(event_output_path, index=False, encoding="utf-8-sig")
            print(f"💾 이벤트 참여 댓글 저장: {event_output_path}")
            print(f"   {len(event_df)}개")
    
    # 전체 결과도 저장
    full_output_path = Path(f"./outputs/filtered_all_{csv_stem}{suffix}.csv")
    df_result_with_original.to_csv(full_output_path, index=False, encoding="utf-8-sig")
    print(f"💾 전체 결과 저장: {full_output_path}")
    print(f"   컬럼: {list(df_result_with_original.columns)}")
    print(f"   입력 토큰: {total_tokens['input']:,} / 출력 토큰: {total_tokens['output']:,}")
    if stats["success"] > 0:
        print(f"   댓글 1개 평균: 입력 {total_tokens['input'] / stats['success']:.1f} / "
              f"출력 {total_tokens['output'] / stats['success']:.1f} tokens")

    # ── label 분포 출력 ───────────────────────────────────────
    if "label" in df_result_with_original.columns:
        print(f"\n[판정 결과 분포]")
        for label, count in df_result_with_original["label"].value_counts().items():
            pct = count / len(df_result_with_original) * 100
            print(f"   {label}: {count}개 ({pct:.1f}%)")

    # 이벤트 참여 분포 출력
    if "is_event" in df_result_with_original.columns:
        print(f"\n[이벤트 참여 분류]")
        event_count = df_result_with_original["is_event"].sum()
        non_event_count = len(df_result_with_original) - event_count
        event_pct = event_count/len(df_result_with_original)*100 if len(df_result_with_original) > 0 else 0
        print(f"   이벤트 참여: {event_count}개 ({event_pct:.1f}%)")
        print(f"   일반 댓글: {non_event_count}개 ({100-event_pct:.1f}%)")
        
        # 이벤트 참여 댓글의 label 분포
        event_with_label = df_result_with_original[df_result_with_original["is_event"]]
        if not event_with_label.empty:
            print(f"\n   [이벤트 참여 댓글 중 label 분포]")
            for label, count in event_with_label["label"].value_counts().items():
                pct = count / len(event_with_label) * 100
                print(f"      {label}: {count}개 ({pct:.1f}%)")

    # spam_type 분포 출력 (Spam + Inappropriate 모두 포함)
    filtered_df_stats = df_result_with_original[df_result_with_original["label"].isin(["Spam", "Inappropriate"])]
    if not filtered_df_stats.empty and "spam_type" in filtered_df_stats.columns:
        print(f"\n[플래그된 댓글 유형 분포] (총 {len(filtered_df_stats)}개)")
        for stype, count in filtered_df_stats["spam_type"].value_counts().items():
            pct = count / len(filtered_df_stats) * 100
            print(f"   {stype}: {count}개 ({pct:.1f}%)")

    # Spam 유형만 따로 출력
    spam_df_only = df_result_with_original[df_result_with_original["label"] == "Spam"]
    if not spam_df_only.empty and "spam_type" in spam_df_only.columns:
        print(f"\n[스팸 유형 분포] (총 {len(spam_df_only)}개)")
        for stype, count in spam_df_only["spam_type"].value_counts().items():
            pct = count / len(spam_df_only) * 100
            print(f"   {stype}: {count}개 ({pct:.1f}%)")

    # Inappropriate 유형만 따로 출력
    inappropriate_df_only = df_result_with_original[df_result_with_original["label"] == "Inappropriate"]
    if not inappropriate_df_only.empty and "spam_type" in inappropriate_df_only.columns:
        print(f"\n[부적절 댓글 유형 분포] (총 {len(inappropriate_df_only)}개)")
        for stype, count in inappropriate_df_only["spam_type"].value_counts().items():
            pct = count / len(inappropriate_df_only) * 100
            print(f"   {stype}: {count}개 ({pct:.1f}%)")

    # recommend_action 분포 출력
    if "recommend_action" in df_result_with_original.columns:
        print(f"\n[권장 조치 분포]")
        for action, count in df_result_with_original["recommend_action"].value_counts().items():
            pct = count / len(df_result_with_original) * 100
            print(f"   {action}: {count}개 ({pct:.1f}%)")

    # ── 스팸 점수 통계 출력 ───────────────────────────────────
    if "spam_score" in df_result_with_original.columns:
        print(f"\n[spam_score 통계]")
        print(f"   평균: {df_result_with_original['spam_score'].mean():.2f} | "
              f"중앙값: {df_result_with_original['spam_score'].median():.1f} | "
              f"최대: {df_result_with_original['spam_score'].max()}")

    # confidence 분포 출력
    if "confidence" in df_result_with_original.columns:
        print(f"\n[신뢰도(confidence) 분포]")
        high_conf = len(df_result_with_original[df_result_with_original["confidence"] >= 0.9])
        mid_conf = len(df_result_with_original[(df_result_with_original["confidence"] >= 0.7) & (df_result_with_original["confidence"] < 0.9)])
        low_conf = len(df_result_with_original[df_result_with_original["confidence"] < 0.7])
        print(f"   매우 높음 (≥0.9): {high_conf}개 ({high_conf/len(df_result_with_original)*100:.1f}%)")
        print(f"   중간 (0.7~0.9): {mid_conf}개 ({mid_conf/len(df_result_with_original)*100:.1f}%)")
        print(f"   낮음 (<0.7): {low_conf}개 ({low_conf/len(df_result_with_original)*100:.1f}%)")


    # ── 상위 위험 댓글 출력 (신뢰도 기준 상위 10개 - Spam + Inappropriate) ────────
    # confidence와 label 기준으로 정렬
    if "spam_score" in df_result_with_original.columns:
        top_flagged = filtered_df_stats.nlargest(10, "confidence").sort_values("spam_score", ascending=False)
        print(f"\n[위험 댓글 상위 10개 (신뢰도 기준, Spam + Inappropriate)]")
        for i, row in enumerate(top_flagged.itertuples(), start=1):
            original_text = row.comment_text if hasattr(row, 'comment_text') else ""
            action_emoji = "🚫" if row.label == "Inappropriate" else "⚠️"
            print(f"{i}. {action_emoji} [신뢰도:{row.confidence:.2f}] [점수:{row.spam_score}] [{row.label}] [{row.spam_type}]")
            print(f"   댓글: {original_text[:80]}")
            print(f"   근거: {row.reason[:100]}")
            print()


# ========================================
# 13. 진입점
# ========================================
if __name__ == "__main__":
    asyncio.run(main())
