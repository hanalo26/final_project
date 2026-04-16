"""
무아튜터님께서 개선을 도와주신 agent로 주요 개선 사항은 다음과 같습니다.

- YouTube 도구 함수 추가 (검색 및 검증을 위해서)

- 시스템 프롬프트 개선 
    - 웹 서치와 유튜브 서치를 병행하여 기업과 관련된 유튜브 채널_id 수집
    - 출력되는 데이터 규칙 명시
    - 응답 시, 제한 사항 추가

- 사용하는 도구의 변화
    - tavily+ mcp-server-fetch 도구 -> tavily + 유튜브 API를 활용한 검색 및 검증 도구
    
- 터미널에서 파일 실행 시, 각 과정마다 사용한 도구를 출력하도록 하는 파일 추가
    - 가장 최상위에 "utils.py"라는 이름으로 존재
"""

# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import json
import asyncio
from pprint import pprint
import pandas as pd
from typing import Optional

import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.tools import Tool                                 # 커스텀 도구 등록용
from pydantic_ai.common_tools.tavily import tavily_search_tool  # Tavily 검색 도구 직접 연결
from pydantic_ai.models.google import GoogleModelSettings       # 모델 설정 (temperature 등)

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))       # 프로젝트 루트를 import 경로에 추가
from utils import print_tool_calls                              # 도구 호출 내역 출력 함수

# ========================================
# 2. 환경변수 로드 및 API 연결 확인
# ========================================

# override=True: 시스템 환경변수에 이미 등록된 값이 있어도 .env 파일 값으로 덮어씀 
#              -> 모델명을 바꿔도 .env 파일 다시 안 만들어도 됨
load_dotenv(dotenv_path=".env", override=True)

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", '')

# PydanticAI는 GEMINI_API_KEY 환경변수를 자동으로 인식
# 모델 ID 형식: 'google-gla:{모델명}'
model_id = f"google-gla:{GEMINI_MODEL}"

# API 키 유효성 검사
api_key_valid = GEMINI_API_KEY and "YOUR_API_KEY" not in GEMINI_API_KEY
print(f"Gemini API 키 설정 확인: {'✅' if api_key_valid else '❌'}")
print(f"사용할 모델: {model_id}")

# Tavily API 연결 확인
try:
    response = requests.post(
        "https://api.tavily.com/search",
        json={"query": "test", "api_key": TAVILY_API_KEY}
    )
    print(f"\nTavily API 연결: {'✅' if response.status_code == 200 else '❌'}")
except Exception as e:
    print(f"\nTavily API 연결: ❌ ({e})")
    
# YouTube Data API 연결 확인
try:
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "snippet", "forHandle": "@YouTube", "key": YOUTUBE_API_KEY}
    )
    print(f"\nYouTube Data API 연결: {'✅' if response.status_code == 200 else '❌'}")
except Exception as e:
    print(f"\nYouTube Data API 연결: ❌ ({e})")

# API 호출 간격 (초)
API_DELAY = 3

# ========================================
# 3. 응답 스키마 정의
# ========================================

"""
응답 스키마 구조
====================================
최상위 구조: Company (BaseModel)
├── company_name (str): 기업명
├── channels (list[Channel]): 찾은 공식 채널 리스트 (0개 이상)
│   ├── channel_id (str): 유튜브 채널 ID (UCxxxxxxx 형식)
│   ├── channel_name (str): 유튜브 채널명
│   ├── channel_handle (str): 유튜브 채널 핸들 (@handle 형식)
│   ├── channel_url (str): 유튜브 채널 URL
│   └── channel_reason (str): 해당 채널을 공식 채널로 판별한 이유
├── has_official (bool): 공식 채널 소유 여부 (channels가 1개 이상이면 True)
└── reason (str): 탐색 과정 요약 또는 공식 채널을 찾지 못한 이유 (최소 10자 이상)

참고:
- 한 기업이 여러 공식 채널을 운영할 수 있음 (예: 삼성전자 → 삼성전자 Korea, Samsung Global 등)
- 공식 채널을 찾지 못한 경우 channels는 빈 리스트 []
"""
class Channel(BaseModel):
    channel_id: str = Field(
        description="유튜브 채널 ID (UCxxxxxxx 형식, verify_youtube_channel 결과에서 가져올 것)"
    )
    channel_name: str = Field(
        description="유튜브 채널명 (verify_youtube_channel 결과의 title)"
    )
    channel_handle: str = Field(
        description="유튜브 채널 핸들 (@handle 형식, verify_youtube_channel 결과의 custom_url)"
    )
    channel_url: str = Field(
        description="유튜브 채널 URL (https://www.youtube.com/@handle 형식)"
    )
    channel_reason: str = Field(
        description="해당 채널을 공식 채널로 판별한 이유",
        min_length=10
    )

class Company(BaseModel):
    company_name: str = Field(
        description="기업명"
    )
    channels: list[Channel] = Field(
        default=[],
        description="찾은 공식 유튜브 채널 리스트 (0개 이상)"
    )
    has_official: bool = Field(
        description="공식 채널 소유 여부 (channels가 1개 이상이면 True)"
    )
    reason: str = Field(
        description="탐색 과정 요약 또는 공식 채널을 찾지 못한 이유",
        min_length=10
    )
# ========================================
# 4. 체크포인트 및 지수 백오프 함수 정의
# ========================================

CHECKPOINT_FILE = Path("./outputs/checkpoint_IT(test용_배치처리).json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
CHECKPOINT_EVERY = 3         # N건마다 중간 저장
MAX_RETRIES = 4              # 최대 재시도 횟수
BASE_DELAY = 10              # 기본 대기 시간 (초)

MAX_DELAY = 60               # 최대 대기 시간 (초)
MAX_CONCURRENT = 3           # 동시 API 호출 수 (Semaphore 제한)

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
    company_name,
    settings,
    stats,
    all_results,
    total_tokens,
    pbar,
):
    """
    Semaphore로 동시 실행을 제한하면서 기업 1개를 처리하고,
    실패 시 지수 백오프로 재시도
    """
    async with sem:  # 슬롯이 가득 차면 대기, 빈 슬롯 생기면 즉시 진입
        prompt = f"{company_name}의 공식 유튜브 채널을 찾아줘."

        for attempt in range(MAX_RETRIES):
            try:
                result = await agent.run(prompt, model_settings=settings)

                usage = result.usage()
                input_tok  = usage.input_tokens or 0
                output_tok = usage.output_tokens or 0
                total_tokens['input']  += input_tok
                total_tokens['output'] += output_tok

                # 도구 호출 내역 출력
                print(f"\n{'─' * 40}")
                print(f"📌 [{company_name}] 도구 호출 내역")
                print(f"{'─' * 40}")
                print_tool_calls(result, detail=True)

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
                    print(f"⚠️ [{company_name}] 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기)")
                    await asyncio.sleep(delay)
                else:
                    print(f" ❌ [최종 실패] {company_name} | {error_msg[:100]}")
                    stats['fail'] += 1
                    pbar.update(1)
                    pbar.set_postfix(성공=stats['success'], 실패=stats['fail'])

# ========================================
# 5. YouTube 도구 함수 정의 (검색 + 검증) <- 모델 개선 시 추가된 도구
# ========================================

def search_youtube_channel(query: str) -> str:
    """
    YouTube Data API의 search.list를 사용하여 유튜브 내에서 채널을 직접 검색합니다.
    Tavily 웹 검색으로 채널을 찾지 못했을 때 보조 수단으로 사용하세요.
    - query: 검색어 (예: "삼성전자", "Samsung Electronics official")
    - 반환값: 검색된 채널 목록 (최대 5개, 각 채널의 channel_id, title, description)
    - quota 비용: 호출당 100 units (일일 한도 10,000 units 중) — 꼭 필요할 때만 사용할 것
    """
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": query,
            "type": "channel",
            "maxResults": 5,
            "key": YOUTUBE_API_KEY,
        }
    )

    if response.status_code != 200:
        return f"API 호출 실패 (status={response.status_code}): {response.text[:200]}"

    data = response.json()
    items = data.get('items', [])

    if not items:
        return f"'{query}' 검색 결과 채널이 없습니다."

    results = []
    for i, item in enumerate(items, start=1):
        snippet = item.get('snippet', {})
        ch_id = item.get('id', {}).get('channelId', 'N/A')
        results.append(
            f"{i}. title: {snippet.get('title', 'N/A')}\n"
            f"   channel_id: {ch_id}\n"
            f"   description: {snippet.get('description', 'N/A')[:200]}"
        )

    return f"[유튜브 채널 검색 결과] {len(items)}건\n" + "\n".join(results)

def verify_youtube_channel(channel_handle_or_id: str) -> str:
    """
    YouTube Data API의 channels.list를 사용하여 채널의 존재 여부를 검증하고 메타데이터를 반환합니다.
    - channel_handle_or_id: 채널 핸들(@handle) 또는 채널 ID(UC로 시작) 둘 다 가능
      예: "@SamsungKorea", "SamsungKorea", "UCxxxxxxx"
    - 반환값: 채널 메타데이터 문자열 (channel_id, title, description, subscriber_count, custom_url)
    - quota 비용: 호출당 1-3 units (일일 한도 10,000 units 중)
    """
    value = channel_handle_or_id.lstrip('@')

    # UC로 시작하면 channel_id로 조회, 아니면 handle로 조회
    if value.startswith("UC"):
        params = {"part": "snippet,statistics", "id": value, "key": YOUTUBE_API_KEY}
    else:
        params = {"part": "snippet,statistics", "forHandle": value, "key": YOUTUBE_API_KEY}

    response = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params=params
    )

    if response.status_code != 200:
        return f"API 호출 실패 (status={response.status_code}): {response.text[:200]}"

    data = response.json()

    if not data.get('items'):
        return f"'{channel_handle_or_id}'에 해당하는 유튜브 채널이 존재하지 않습니다."

    item = data['items'][0]
    snippet = item.get('snippet', {})
    stats = item.get('statistics', {})

    return (
        f"[검증 결과] 채널이 존재합니다.\n"
        f"- channel_id: {item['id']}\n"
        f"- title: {snippet.get('title', 'N/A')}\n"
        f"- description: {snippet.get('description', 'N/A')[:300]}\n"
        f"- custom_url: {snippet.get('customUrl', 'N/A')}\n"
        f"- subscriber_count: {stats.get('subscriberCount', 'N/A')}\n"
        f"- video_count: {stats.get('videoCount', 'N/A')}"
    )

# ========================================
# 6. 시스템 프롬프트 및 Agent 초기화
# ========================================

it_system_prompt = """
너는 기업의 공식 유튜브 채널을 찾는 데이터 전문가야.
한 기업이 공식 채널을 여러 개 운영할 수 있으므로 (예: 국내용, 글로벌용, 채용 채널 등),
찾을 수 있는 공식 채널을 모두 찾아줘.

[사용 가능한 도구 3가지]
1. tavily_search: 웹 검색으로 기업의 유튜브 채널 정보를 찾는다 (1순위 탐색 도구)
2. search_youtube_channel: 유튜브 내에서 직접 채널을 검색한다 (2순위 보조 탐색 도구, quota 높으므로 필요할 때만)
3. verify_youtube_channel: 핸들(@handle) 또는 채널 ID(UC...)로 채널을 검증하고 정확한 메타데이터를 가져온다 (필수 검증 도구)

[공식 채널 탐색 절차]
각 기업에 대해 아래 순서대로 진행해줘.

1단계 - 후보 수집
  a) tavily_search로 웹 검색하여 채널 핸들(@handle)이나 채널 URL을 수집
     - 검색어: "{기업명} 공식 유튜브 채널", "{기업명} youtube 채널"
     - 추가 검색어: "{기업 영문명} official youtube channel"
  b) search_youtube_channel로 유튜브에서 직접 "{기업명}"을 검색하여 후보 채널 ID 수집
     - 1a에서 핸들을 충분히 찾았더라도, 추가 공식 채널이 있을 수 있으므로 병행 권장
     - 검색 결과에서 기업명과 관련 있어 보이는 채널의 channel_id를 수집

2단계 - 검증
  - 1단계에서 수집한 모든 핸들과 채널 ID를 verify_youtube_channel로 검증
    - 핸들로 찾았으면: verify_youtube_channel("@핸들")
    - 채널 ID로 찾았으면: verify_youtube_channel("UC...")
  - 채널이 존재하지 않으면 해당 후보는 제외

3단계 - 공식 채널 판별
  검증 결과의 title, description을 보고 아래 기준으로 공식 채널 여부를 판별:
  - title에 기업명 또는 기업 브랜드가 포함되어 있는가
  - description에 "공식", "official", 기업 홈페이지 URL 등이 포함되어 있는가
  - 제외 대상: 팬 채널, 뉴스/리뷰 채널, 개인 채널 등 기업이 직접 운영하지 않는 채널

4단계 - 재시도 (필요 시)
  - 공식 채널을 하나도 찾지 못한 경우: 검색어를 바꿔서 1단계부터 재시도 (최대 3회)
  - 최종적으로 찾지 못한 경우: channels를 빈 리스트로, has_official=False로 저장

[중요 - 출력 데이터 규칙]
- 절대 채널 정보를 직접 추측하거나 만들어내지 말 것
- channels의 모든 필드는 반드시 verify_youtube_channel 검증 결과에서 가져올 것:
    channel_id     ← 검증 결과의 "channel_id" 그대로
    channel_name   ← 검증 결과의 "title" 그대로
    channel_handle ← 검증 결과의 "custom_url" 그대로 (예: @SamsungKorea)
    channel_url    ← "https://www.youtube.com/" + channel_handle
- verify_youtube_channel을 거치지 않은 채널은 절대 출력에 포함하지 말 것

[출력 형식]
1. 공식 채널을 찾은 경우
    - channels 리스트에 검증된 공식 채널들을 모두 추가, has_official=True
    - 각 채널마다 channel_reason에 공식 채널로 판별한 근거를 구체적으로 기록
2. 공식 채널을 찾지 못한 경우
    - channels는 빈 리스트 [], has_official=False
3. reason에는 전체 탐색 과정 요약 또는 찾지 못한 이유를 기록
"""

async def run_agent(company_names: list) -> list:
    # 체크포인트 복원
    all_results      = load_checkpoint()
    processed_names  = {r['company_name'] for r in all_results}

    # 미처리 기업 필터링
    pending = [name for name in company_names if name not in processed_names]

    if processed_names:
        print(f"스킵: {len(processed_names)}개 (이미 처리됨)")
    print(f"처리 대상: {len(pending)}개")
    print(f"처리 방식: 병렬 처리 (최대 {MAX_CONCURRENT}개 동시 실행)")
    print(f"재시도: 최대 {MAX_RETRIES}회 (지수 백오프, 기본 {BASE_DELAY}초)")
    print("=" * 60)

    stats = {'success': len(all_results), 'fail': 0}
    total_tokens = {'input': 0, 'output': 0}

    # Agent 초기화
    # - mcp-server-fetch 제거 → YouTube Data API 도구로 대체
    # - verify_youtube_channel: channels.list API로 채널 존재 여부 및 메타데이터 검증
    agent = Agent(
        model_id,
        output_type=Company,
        system_prompt=it_system_prompt,
        retries=3,
        tools=[
            tavily_search_tool(TAVILY_API_KEY),              # Tavily 웹 검색 도구
            Tool(search_youtube_channel, takes_ctx=False),   # YouTube 채널 검색 도구 (보조)
            Tool(verify_youtube_channel, takes_ctx=False),   # YouTube 채널 검증 도구
        ],
    )

    # output_validator: 논리적 일관성 검증
    @agent.output_validator
    def validate_output(ctx, output: Company) -> Company:
        # has_official=True인데 channels가 비어있으면 재시도
        if output.has_official and len(output.channels) == 0:
            raise ModelRetry(
                f"{output.company_name}의 has_official이 True인데 "
                f"channels가 비어있습니다. 공식 채널을 channels에 추가해주세요."
            )
        # has_official=False인데 channels가 있으면 재시도
        if not output.has_official and len(output.channels) > 0:
            raise ModelRetry(
                f"{output.company_name}의 has_official이 False인데 "
                f"channels에 {len(output.channels)}개 채널이 있습니다. "
                f"has_official을 True로 변경해주세요."
            )
        # 각 채널의 channel_id 형식 검증 (UC로 시작해야 함)
        for ch in output.channels:
            if not ch.channel_id.startswith("UC"):
                raise ModelRetry(
                    f"{output.company_name}의 채널 '{ch.channel_name}'의 "
                    f"channel_id가 '{ch.channel_id}'인데 유효한 형식(UC로 시작)이 아닙니다. "
                    f"verify_youtube_channel 도구로 검증한 정확한 channel_id를 사용해주세요."
                )
        return output

    # 진행률 바 생성
    settings = GoogleModelSettings(temperature=0.3)
    pbar = tqdm(total=len(pending), desc="채널 탐색")

    # 병렬 처리
    tasks = [
        analyze_one(agent, name, settings, stats, all_results, total_tokens, pbar)
        for name in pending
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    # 최종 체크포인트 저장
    save_checkpoint(all_results)

    return all_results, stats, total_tokens
        
# ========================================
# 7. 성능 테스트
# ========================================

async def main():
    # 테스트용 기업 리스트 (Agent 성능 테스트용 샘플 데이터)
    test_companies = [
        "삼성전자",
        "LG전자",
        "네이버",
        "네이버페이",
        "그로비랩스 (그로비 교육)",
        "거민시스템"
    ]
    
# async def main():
#     # 실제 기업 리스트 
#     df = pd.read_csv("./data/IT_company_list.csv", encoding="utf-8-sig")
#     company_names = df['company_name'].tolist()  # 기업명 컬럼명에 맞게 수정

#     print(f"전체 기업 수: {len(company_names)}개")
#     print("=" * 60)

    # Agent 호출
    all_results, stats, total_tokens = await run_agent(test_companies)

    # 처리 결과 요약
    print()
    print("=" * 60)
    print("배치 처리 완료 (병렬 처리 방식)")
    print(f"  성공: {stats['success']}개")
    print(f"  실패: {stats['fail']}개")
    print(f"  성공률: {stats['success'] / max(stats['success'] + stats['fail'], 1) * 100:.1f}%")
    print(f"  체크포인트: {CHECKPOINT_FILE}")
    print("=" * 60)

    # 토큰 사용량 출력
    print(f"\n[토큰 사용량]")
    print(f"  입력 토큰: {total_tokens['input']:,}")
    print(f"  출력 토큰: {total_tokens['output']:,}")
    if stats['success'] > 0:
        print(f"  1건 평균: 입력 {total_tokens['input'] / stats['success']:.0f} / 출력 {total_tokens['output'] / stats['success']:.0f} tokens")

    # 결과 확인
    print(f"\n[결과 요약]")
    for i, company in enumerate(all_results, start=1):
        status = "✅" if company['has_official'] else "❌"
        ch_count = len(company.get('channels', []))
        print(f"{i}. {status} {company['company_name']} (공식 채널 {ch_count}개)")
        for j, ch in enumerate(company.get('channels', []), start=1):
            print(f"   {j}) {ch['channel_name']} | {ch['channel_url']}")
            print(f"      └── {ch['channel_reason']}")
        if ch_count == 0:
            print(f"   └── {company['reason']}")


# ========================================
# 8. main 실행 블록
# ========================================
if __name__ == '__main__':
    # asyncio.run()은 프로그램 전체에서 딱 한 번만 호출하는 것이 원칙
    # 이유: asyncio.run()은 새 이벤트 루프 생성 → 코루틴 실행 → 루프 종료를
    #       한 번에 처리하므로, 여러 번 호출하면 루프 충돌이 발생할 수 있음
    # __name__ == '__main__': 터미널에서 이 파일을 직접 실행할 때만 루프가 돌아가도록 함
    asyncio.run(main())  # 비동기 함수의 진입점 - 최초 1회만 실행