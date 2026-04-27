# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import json
import asyncio
import pandas as pd
import argparse

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from pydantic_ai import Agent, ModelRetry

# Vertex AI 연결
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

import sys
sys.path.append(str(Path(__file__).resolve().parents[2])) # 프로젝트 루트를 import 경로에 추가

# ========================================
# 2. 환경변수 로드 및 API 연결 확인
# ========================================

# override=True: 시스템 환경변수에 이미 등록된 값이 있어도 .env 파일 값으로 덮어씀 
#              -> 모델명을 바꿔도 .env 파일 다시 안 만들어도 됨
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
    
# 데이터 파일을 터미널에서 변수처럼 전달해주기 위해 필요한 변수 정의
parser = argparse.ArgumentParser()
parser.add_argument('csv_path') # 필수 위치 인자: 터미널에서 첫 번째로 입력한 값이 csv_path에 저장됨

# 옵션 인자 추가
parser.add_argument("--concurrent", type=int, default=3) # 동시 실행 수 (기본값 3, 미입력 시 기본값 사용)
parser.add_argument("--delay", type=int, default=10) # 기본 대기 시간 (기본값 10초, 미입력 시 기본값 사용)

args = parser.parse_args()      # 터미널에서 입력한 인자를 파싱하여 args 객체에 저장

# ========================================
# 3. 응답 스키마 정의
# ========================================

"""
응답 스키마 구조
====================================
VideoClassification (BaseModel)
├── video_id           (str)  : 영상 고유 ID (식별자)
├── domain             (str)  : 영상이 속한 도메인 (IT / F&B)
├── content_type       (str)  : 영상이 어떤 포맷으로 만들어졌는지
├── marketing_purpose  (str)  : 영상을 왜 올렸는지
├── cta_type           (str)  : 영상이 시청자에게 뭘 하라고 하는지
├── is_series          (bool) : 시리즈물인지 아닌지
├── is_collaboration   (bool) : 다른 브랜드나 크리에이터와 함께 만든 영상인지
└── reason             (str)  : 위 항목들로 분류한 이유
"""

class VideoClassification(BaseModel):
    video_id: str = Field(
        description="영상의 고유 ID. 입력 데이터에서 그대로 가져온다."
    )

    domain: Literal[
        "IT",   # 기술, 소프트웨어, 전자기기 관련 기업의 영상
        "F&B",  # 식품, 음료, 외식 관련 기업의 영상
    ] = Field(
        description="이 영상이 속한 산업 도메인. 채널 이름과 카테고리 ID를 우선 참고해서 고른다. "
                    "카테고리 ID 28(Science & Technology)이면 IT, "
                    "카테고리 ID 26(Howto & Style), 24(Entertainment)이면 F&B를 우선 고려한다."
    )

    content_type: Literal[
        # 공통 포맷
        "웹드라마",      # 드라마 형식으로 만든 브랜디드 콘텐츠 영상
        "브이로그",      # 특정 목적 없이 일상이나 현장을 자연스럽게 담은 영상 (시설 소개 목적이면 시설소개)
        "시설소개",      # 회사 사옥, 매장, 데이터센터 등 내/외부 공간이나 시설을 소개하는 영상
        "에피소드소개",  # 캐릭터·애니메이션·편집된 스토리텔링 형식으로 인물의 이야기를 전달하는 영상
        "제품리뷰",      # 제품을 직접 써보며 특징을 소개하는 영상
        "튜토리얼",      # 시청자가 직접 따라할 수 있도록 단계별로 알려주는 영상 (기술 개념 설명 아님)
        "광고/CF",       # TV 광고처럼 짧고 강하게 만든 영상
        "다큐멘터리",    # 특정 주제나 이야기를 깊이 있게 다루는 영상
        "웹예능",        # 게임, 미션, 토크쇼, 리액션 등 오락 목적의 구성이 명확한 영상
        "이벤트/행사",   # 제품 런칭, 기자간담회, 팝업스토어 등 특정 행사 현장을 담은 영상
        "인터뷰",        # 실제 인물이 카메라 앞에서 직접 말하거나 대화하는 형식이 명확한 영상
        "애니메이션",    # 실사 촬영 없이 모션그래픽, 2D/3D 애니메이션으로 만든 영상

        # IT 특화 포맷
        "기술설명",      # 개발 개념, 알고리즘 등 따라하기보다 이해를 목적으로 한 순수 기술 정보 영상

        # F&B 특화 포맷
        "요리/레시피",   # 요리 과정이나 식재료 조합을 보여주는 영상
        "영양정보",      # 영양학적 정보나 건강 관련 정보를 다루는 영상
        "고객후기",      # 고객 인터뷰나 실제 사용 경험을 담은 영상 (인터뷰 형식이어도 고객 경험이 핵심이면 고객후기)

        "기타",          # 위 포맷 중 어디에도 명확히 해당하지 않는 경우에만 선택
    ] = Field(
        description="이 영상이 어떤 형식(포맷)으로 만들어졌는지 고른다. domain을 참고해서 도메인 특화 포맷을 우선 고려한다."
    )

    marketing_purpose: Literal[
        "브랜드캠페인",  # 브랜드 슬로건, 감성 캠페인 등 브랜드 인지 목적 (기업 신뢰·평판과 다름)
        "제품홍보",      # 특정 제품이나 서비스를 알리고 관심을 끌려는 영상
        "고객유입",      # 브랜드를 모르는 잠재 고객의 관심을 끌어오려는 영상 (기존 고객 유지와 다름)
        "고객유지",      # 기존 고객의 이탈을 막고 로열티를 강화하려는 영상 (신규 고객 유입과 다름)
        "기업이미지",    # 수상, ESG, 임직원 스토리 등 브랜드의 신뢰·평판 관리 목적 (브랜드 인지 캠페인과 다름)
        "채용",          # 회사에 지원할 사람을 찾으려고 만든 영상
        "사회공헌/환경", # 환경 보호나 사회적 가치를 전달하려는 영상
        "서비스활용",    # 자사 제품·서비스 사용법·기능·설정이 핵심인 영상 (일반 정보 제공과 다름)
        "정보제공",      # 자사 제품과 직접 관련 없는 일반 정보·지식이 핵심인 영상 (자사 제품 활용법과 다름)
        "기타",          # 위 목적 중 어디에도 명확히 해당하지 않는 경우에만 선택
    ] = Field(description="이 영상을 올린 목적이 무엇인지 고른다.")

    cta_type: Literal[
        "구매유도",    # 설명란에 구매 링크가 있거나 제품 구매를 직접 유도하는 경우
        "구독유도",    # 채널 구독이나 알림 설정을 요청하는 경우
        "이벤트참여",  # 댓글, 응모, 챌린지, VOC 수집 등 참여를 유도하는 경우
        "정보탐색",    # 공식 홈페이지, 상세페이지 링크만 제공하는 경우
        "앱다운로드",  # 앱 스토어 링크로 앱 설치를 유도하는 경우
        "방문유도",    # 오프라인 매장, 팝업스토어 등 현장 방문을 유도하는 경우
        "기타",        # 위 유형 중 어디에도 명확히 해당하지 않거나 명시적인 CTA가 없는 경우
    ] = Field(description="이 영상이 시청자에게 무엇을 하도록 유도하는지 고른다. "
                        "설명란을 먼저 확인하고, 명시적 CTA가 없으면 marketing_purpose를 참고해서 추론한다.")

    is_series: bool = Field(
        description="제목에 EP, 1화, 2화, #1, 시즌, Part, Track, Session, Day, Vol 등 순서를 나타내는 표현이 있으면 True, 없으면 False"
    )

    is_collaboration: bool = Field(
        description="제목이나 설명에 ft., feat., with, X(크로스), 콜라보 등의 표현이 있으면 True. "
                    "단, 자사 제품끼리의 콜라보(예: 갤럭시 X 버즈)는 False. 해당 표현이 없으면 False"
    )

    reason: str = Field(
        description="위 항목들을 이렇게 분류한 이유를 간단히 설명한다.",
        min_length=10
    )

# ========================================
# 4. 체크포인트 및 지수 백오프 함수 정의
# ========================================

# args.csv_path: 터미널에서 전달받은 CSV 파일 경로
# Path().stem: 파일 경로에서 확장자를 제거한 파일명만 추출
csv_stem = Path(args.csv_path).stem

# 체크포인트 파일명을 CSV 파일명 기반으로 동적 생성
CHECKPOINT_FILE = Path(f"./outputs/checkpoint_{csv_stem}.json")

CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
CHECKPOINT_EVERY = 5         # N건마다 중간 저장
MAX_RETRIES = 4              # 최대 재시도 횟수
BASE_DELAY = args.delay      # 기본 대기 시간 (초)

MAX_DELAY = 60               # 최대 대기 시간 (초)
MAX_CONCURRENT = args.concurrent  # 동시 API 호출 수 (Semaphore 제한)

# Semaphore 생성: async with sem 블록에 동시 진입 가능한 수 = MAX_CONCURRENT
sem = asyncio.Semaphore(MAX_CONCURRENT)

# 체크 포인트 저장/복원
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

# 지수 백오프 재시도 함수
async def analyze_one(
    agent,
    row,
    settings,
    stats,
    all_results,
    total_tokens,
    pbar,
):
    """
    Semaphore로 동시 실행을 제한하면서 영상 1개를 처리하고,
    실패 시 지수 백오프로 재시도
    """
    async with sem:
        # LLM의 텍스트 분류 판단의 근거가 되는 입력 데이터
        prompt = f"""
        채널명: {row['channel_title']}
        카테고리 ID: {row['category_id']}
        유료 PPL 여부: {row['has_paid_product_placement']}
        제목: {row['title']}
        설명: {row['description']}
        태그: {row['tags']}
        영상 ID: {row['video_id']}
        """

        for attempt in range(MAX_RETRIES):
            try:
                result = await agent.run(prompt, model_settings=settings)

                usage = result.usage()
                input_tok  = usage.input_tokens or 0
                output_tok = usage.output_tokens or 0
                total_tokens['input']  += input_tok
                total_tokens['output'] += output_tok

                all_results.append(result.output.model_dump())
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
# 5. 시스템 프롬프트 및 Agent 초기화
# ========================================
system_prompt = """
너는 기업 유튜브 영상을 분류하는 마케팅 데이터 전문가야.
입력된 영상 정보를 읽고 아래 규칙에 따라 각 필드를 분류해.

[입력 데이터 활용 규칙]
- 채널명과 카테고리 ID는 domain, content_type 분류 시 가장 먼저 참고해
- 유료 PPL 여부가 True이면, `content_type=광고/CF`, `marketing_purpose=제품홍보`를 우선 고려해
- 제목은 content_type, is_series, is_collaboration 분류의 핵심 근거야
- 설명은 marketing_purpose, cta_type 분류의 핵심 근거야
- 태그는 content_type, domain 분류의 보조 근거야
- video_id는 분류하지 말고 입력값 그대로 출력해

[분류 규칙]
1. domain
- 채널명에 기술/IT/전자 관련 키워드가 있으면 IT
- 채널명에 식품/음료/외식 관련 키워드가 있으면 F&B

2. content_type
- 공통 포맷은 제목·설명·태그를 보고 판단해
- domain=IT일 때 기술설명을 추가로 고려해
- domain=F&B일 때 요리/레시피, 영양정보, 고객후기를 추가로 고려해
- 에피소드소개는 캐릭터·애니메이션·편집된 스토리텔링 형식으로 인물의 이야기를 전달할 때 선택해
  (예: 직원 스토리 시리즈, 캐릭터가 등장하는 브랜드 스토리텔링 영상)
- 인터뷰는 실제 인물이 카메라 앞에서 직접 말하거나 대화하는 형식이 명확할 때 선택해
- 웹예능은 게임, 미션, 토크쇼, 리액션 등 오락 목적의 구성이 명확할 때만 선택해
- 특정 인물이 경험·의견을 말하는 형식이면 웹예능이 아니라 인터뷰 또는 에피소드소개를 고려해
- 회사 내부 공간, 매장, 시설 등을 소개하는 영상이면 시설소개를 고려해
    (예: 매장 펫존 소개, 사옥 투어, 데이터센터 소개)
- 광고, 행사 등의 촬영 현장을 스케치하거나 메이킹 필름 형식이면 브이로그를 고려해
- 제품을 짧고 강하게 보여주거나 캠페인 해시태그 중심의 짧은 영상이면 광고/CF를 고려해
- 제품리뷰는 실제로 제품을 사용하거나 시음하며 특징을 소개하는 형식이 명확할 때만 선택해
  (콜라보 신메뉴 티저, 제품 출시 영상은 광고/CF를 고려해)
- 제목이나 설명에 이벤트, 공모전, 기념일, 행사 관련 내용이 있으면 이벤트/행사를 고려해
- 기타는 위 포맷 중 어디에도 명확히 해당하지 않을 때만 선택해

3. marketing_purpose
- 설명란에 구매 링크, 제품 상세 안내가 있으면 제품홍보를 우선 고려해
- 제목에 특정 제품명이 있으면 브랜드캠페인보다 제품홍보를 우선 고려해
- 브랜드 슬로건, 캠페인 형식이지만 특정 제품명이 없으면 브랜드캠페인을 고려해
- 임직원, 바리스타 등 내부 구성원의 스토리를 통해 브랜드 신뢰를 높이는 영상이면
    브랜드캠페인보다 기업이미지를 고려해
- 수상, ESG, 사회적 활동 내용이면 기업이미지 또는 사회공헌/환경을 고려해
    - 기업 활동·평판 관련이면 기업이미지, 환경·사회적 가치 전달이 핵심이면 사회공헌/환경
- 자사 기술력, 연구 성과, 컨퍼런스 발표 내용을 알리는 목적이면 기업이미지를 고려해
- 자사 서비스 내에서 즐길 수 있는 콘텐츠를 추천하거나 소개하는 영상이면
    정보제공보다 서비스활용을 고려해
    (예: "B tv에서 볼 수 있는 영화 추천", "앱에서 즐기는 이달의 신작")
- 자사와 직접 관련 없는 일반 정보·지식이 핵심이면 정보제공을 고려해
    (예: "겨울철 면역력 높이는 식재료", "AI 트렌드 2024")
- 신규 고객을 끌어오거나 브랜드를 처음 접하는 사람을 대상으로 한 내용이면 고객유입을 고려해
- 기존 고객 재구매, 로열티 강화, 이탈 방지, 기념일·주년 이벤트 관련 내용이면 고객유지를 고려해
- 채용 관련 키워드(입사, 채용, 커리어 등)가 있으면 채용을 고려해
- 자사 제품·서비스 사용법·기능·설정이 핵심이면 서비스활용을 고려해
    (예: "BTV 채널 추가하는 법", "갤럭시 카메라 설정 방법")
- 기타는 위 목적 중 어디에도 명확히 해당하지 않을 때만 선택해

4. cta_type
- 설명란에 구매 링크가 있으면 구매유도
- 설명란에 이벤트, 응모, 챌린지 관련 문구가 있으면 이벤트참여
    (예: 댓글 이벤트, 고객 후기 남기기, VOC 수집 등 참여 유도 포함)
- 설명란에 앱 스토어 링크가 있으면 앱다운로드
- 설명란에 매장 위치, 팝업 일정 등 오프라인 방문 안내가 있으면 방문유도
- 설명란에 해시태그만 있고 링크가 없으면 기타를 선택해
- 설명란에 더 알아보기, 공식 홈페이지 링크만 있으면 정보탐색
- 이벤트가 종료됐더라도 영상 제작 시점의 CTA를 기준으로 판단해
    (예: 공모전 안내 영상이면 현재 종료됐어도 이벤트참여로 분류)
- 설명란에 명시적인 CTA가 없으면 바로 기타를 선택하지 말고, 반드시 marketing_purpose를 먼저 확인해서 추론해
    - marketing_purpose=제품홍보이면 구매유도를 선택해
    - marketing_purpose=서비스활용이면 정보탐색을 고려해
- 위 추론도 어려울 때만 기타를 선택해

[중요]
- 추측하지 말고 입력 데이터에 근거해서 분류해
- 근거가 불충분하면 기타를 선택해
- 기타는 최후의 수단이야. 위 항목 중 명확히 해당하는 게 없을 때만 선택해
"""

async def run_agent(df: pd.DataFrame) -> list:
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
        output_type=VideoClassification,
        system_prompt=system_prompt,
        retries=3
    )

    # output_validator: 논리적 일관성 검증
    @agent.output_validator
    def validate_output(ctx, output: VideoClassification) -> VideoClassification:
        if not output.video_id:
            raise ModelRetry("video_id가 비어있습니다. 입력 데이터의 video_id를 그대로 넣어주세요.")

        it_only  = {"기술설명"}
        fnb_only = {"요리/레시피", "영양정보", "고객후기"}

        if output.domain == "IT" and output.content_type in fnb_only:
            raise ModelRetry(
                f"domain=IT인데 content_type={output.content_type}은 F&B 전용입니다. 다시 분류해주세요."
            )
        if output.domain == "F&B" and output.content_type in it_only:
            raise ModelRetry(
                f"domain=F&B인데 content_type={output.content_type}은 IT 전용입니다. 다시 분류해주세요."
            )
        return output
    
    # 진행률 바 생성
    settings = GoogleModelSettings(temperature=0.3)
    pbar = tqdm(total=len(pending), desc="영상 분류")

    # 병렬 처리
    tasks = [
        analyze_one(agent, row, settings, stats, all_results, total_tokens, pbar)
        for row in pending
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    # 최종 체크포인트 저장
    save_checkpoint(all_results)

    return all_results, stats, total_tokens # 실질적으로는 세 개가 묶여서 튜플로 반환됨
        
# ========================================
# 6. Agent 작업 실행
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

    # AI 분류 결과 저장 (원본 CSV와 별도 저장 → 나중에 따로 merge)
    df_result = pd.DataFrame(all_results)
    output_path = Path(f"./outputs/classified_{csv_stem}.csv")
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료: {output_path}")
    print(f"  컬럼 수: {len(df_result.columns)}개")
    print(f"  입력 토큰: {total_tokens['input']:,} / 출력 토큰: {total_tokens['output']:,}")
    if stats['success'] > 0:
        print(f"  1건 평균: 입력 {total_tokens['input'] / stats['success']:.0f} / 출력 {total_tokens['output'] / stats['success']:.0f} tokens")

    # 결과 요약 출력
    print(f"\n[결과 요약]")
    for i, video in enumerate(all_results, start=1):
        print(f"{i}. {video['video_id']} [{video['domain']}]")
        print(f"   콘텐츠 유형: {video['content_type']} | 마케팅 목적: {video['marketing_purpose']}")
        print(f"   CTA: {video['cta_type']} | 시리즈: {video['is_series']} | 협업: {video['is_collaboration']}")
        print(f"   └── {video['reason']}")

# ========================================
# 7. main 실행 블록
# ========================================
if __name__ == '__main__':
    # asyncio.run()은 프로그램 전체에서 딱 한 번만 호출하는 것이 원칙
    # 이유: asyncio.run()은 새 이벤트 루프 생성 → 코루틴 실행 → 루프 종료를
    #       한 번에 처리하므로, 여러 번 호출하면 루프 충돌이 발생할 수 있음
    # __name__ == '__main__': 터미널에서 이 파일을 직접 실행할 때만 루프가 돌아가도록 함
    asyncio.run(main())  # 비동기 함수의 진입점 - 최초 1회만 실행