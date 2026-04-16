# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import json
import shutil
import asyncio
import pandas as pd
from pprint import pprint
from typing import Optional

import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.mcp import MCPServerStdio                      # MCP 서버 연결 (mcp-server-fetch)
from pydantic_ai.common_tools.tavily import tavily_search_tool  # Tavily 검색 도구 직접 연결
from pydantic_ai.models.google import GoogleModelSettings       # 모델 설정 (temperature 등)

# ========================================
# 2. 환경변수 로드 및 API 연결 확인
# ========================================

# override=True: 시스템 환경변수에 이미 등록된 값이 있어도 .env 파일 값으로 덮어씀 
#              -> 모델명을 바꿔도 .env 파일 다시 안 만들어도 됨
load_dotenv(dotenv_path=".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", '')

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
    
# uvx 확인 (Python MCP 서버 실행에 필요)
uvx_path = shutil.which('uvx')
print(f"\nuvx 설치 확인: {'✅' if uvx_path else '❌'}")

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
├── channel_id (Optional[str]): 유튜브 채널 ID (UCxxxxxxx 형식)
├── channel_name (Optional[str]): 유튜브 채널명
├── channel_handle (Optional[str]): 유튜브 채널 핸들 (@handle 형식)
├── channel_url (Optional[str]): 유튜브 채널 URL
├── has_official (bool): 공식 채널 소유 여부
└── reason (str): 해당 채널을 공식 채널로 판별한 이유 또는 공식 채널을 찾지 못한 이유 (최소 10자 이상)

참고:
- [] 는 "리스트 안의 각 항목"을 의미하는 표기 관례로, API 문서나 JSON Schema에서 흔히 쓰는 방식
- Optional 필드는 공식 채널을 찾지 못한 경우 None으로 저장
"""
class Company(BaseModel):
    company_name: str = Field(
        description="기업명"
    )
    channel_id: Optional[str] = Field(
        default=None,
        description="유튜브 채널 ID (UCxxxxxxx 형식)"
    )
    channel_name: Optional[str] = Field(
        default=None,
        description="유튜브 채널명"
    )
    channel_handle: Optional[str] = Field(
        default=None,
        description="유튜브 채널 핸들 (@handle 형식)"
    )
    channel_url: Optional[str] = Field(
        default=None,
        description="유튜브 채널 URL"
    )
    has_official: bool = Field(
        description="공식 채널 소유 여부 (True/False)"
    )
    reason: str = Field(
        description="해당 채널을 공식 채널로 판별한 이유 또는 공식 채널을 찾지 못한 이유",
        min_length=10
    )
# ========================================
# 4. 체크포인트 및 지수 백오프 함수 정의
# ========================================

CHECKPOINT_FILE = Path("./outputs/checkpoint_IT(test용_순차처리).json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
CHECKPOINT_EVERY = 3         # N건마다 중간 저장
MAX_RETRIES = 4              # 최대 재시도 횟수
BASE_DELAY = 7               # 기본 대기 시간 (초)
MAX_DELAY = 60               # 최대 대기 시간 (초)

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
    기업 1개를 처리하고, 실패 시 지수 백오프로 재시도
    """
    prompt = f"{company_name}의 공식 유튜브 채널을 찾아줘."
    
    for attempt in range(MAX_RETRIES):
        try:
            result = await agent.run(prompt, model_settings=settings)

            # 토큰 사용량 추출 (비용 계산용)
            usage = result.usage()
            input_tok = usage.input_tokens or 0
            output_tok = usage.output_tokens or 0
            total_tokens['input']  += input_tok
            total_tokens['output'] += output_tok

            # 성공: agent의 응답을 리스트로 저장
            # model_dump(): Pydantic 모델 → dict 변환
            all_results.append(result.output.model_dump())
            stats['success'] += 1

            # 진행률 업데이트 + 실시간 토큰 사용량 확인
            pbar.update(1)  # 진행바를 1칸씩 앞으로 이동
            pbar.set_postfix(
                성공=stats['success'],
                실패=stats['fail'],
                입력토큰=total_tokens['input'],
                출력토큰=total_tokens['output']
            )

            # N건마다 중간 저장
            if stats['success'] % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_results)
                print(f" 💾 체크포인트 저장: {stats['success']}건 완료")

            # Rate Limit 방지 대기 (슬롯 점유 상태)
            await asyncio.sleep(BASE_DELAY)
            return  # for 루프(재시도) 탈출

        except Exception as e:
            error_msg = str(e)
            is_rate_limit = '429' in error_msg or 'rate' in error_msg.lower()

            if attempt < MAX_RETRIES - 1:
                # 지수 백오프: BASE_DELAY * 2^attempt
                # 1회: BASE_DELAY 대기
                # 2회: BASE_DELAY * 2 대기
                # 3회: BASE_DELAY * 2^2 대기
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                if is_rate_limit:
                    delay = min(delay * 2, MAX_DELAY)  # 429 에러는 서버 부하 → 일반 에러보다 2배 더 대기
                print(f"⚠️ [{company_name}] 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기)")
                await asyncio.sleep(delay)
            else:
                # max_retries 초과 → 최종 실패
                print(f" ❌ [최종 실패] {company_name} | {error_msg[:100]}")
                stats['fail'] += 1
                pbar.update(1)
                pbar.set_postfix(성공=stats['success'], 실패=stats['fail'])

# ========================================
# 5. 시스템 프롬프트 및 Agent 초기화
# ========================================

it_system_prompt = """
너는 기업의 공식 유튜브 채널을 찾는 데이터 전문가야.
입력된 기업 리스트를 바탕으로 각 기업의 공식 유튜브 채널을 찾아줘.

[공식 채널 탐색 방법]
각 기업에 대해 반드시 아래 순서로 진행해줘.

1. tavily_search 도구로 "{기업명} 공식 유튜브 채널" 검색 후 유튜브 채널 URL 추출
2. fetch 도구로 찾은 URL에 접속해서 페이지 내용 수집
3. 아래 판별 기준을 바탕으로 공식 채널인지 직접 판별
    - 채널명이 기업명과 유사한지
    - 페이지 내용에 "공식" 또는 "official" 키워드가 있는지
    - 페이지 내용에 기업 홈페이지 링크가 포함되어 있는지
    - 공식 채널이 아닌 경우: 검색어를 바꿔서 1번인 tavily 도구 사용부터 다시 시도
    - 3번 재시도 후에도 공식 채널을 찾지 못한 경우: has_official=False로 저장

[출력 형식]
1. 공식 채널을 찾은 경우
    - 채널 정보를 정해진 스키마에 맞게 저장하고, has_official=True로 저장
2. 공식 채널을 찾지 못한 경우
    - channel 관련 필드는 None으로 저장하고, has_official=False로 저장
3. 기업마다 해당 채널을 공식 채널로 판별한 이유 또는 찾지 못한 이유를 reason에 저장
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
    print(f"처리 방식: 순차 처리")
    print(f"재시도: 최대 {MAX_RETRIES}회 (지수 백오프, 기본 {BASE_DELAY}초)")
    print("=" * 60)

    stats = {'success': len(all_results), 'fail': 0}
    total_tokens = {'input': 0, 'output': 0}

    # mcp-server-fetch 서버 설정
    fetch_server = MCPServerStdio(
        'uvx',                          # Python 패키지 러너 (npx의 Python 버전)
        args=['mcp-server-fetch'],      # 실행할 MCP 서버 패키지명 (웹페이지 내용 수집용)
        timeout=30,                     # 서버 시작 대기 시간(초)
    )

    # MCP 서버 실행 - async with 블록이 끝나면 서버 자동 종료
    async with fetch_server:
        # Agent 초기화
        # - model_id: 사용할 Gemini 모델
        # - output_type: 응답 스키마 지정 → 자동으로 타입 검증 및 파싱
        # - system_prompt: 역할 및 조건 정의
        # - retries: 응답 제약 조건을 만족하지 못할 경우 최대 재시도 횟수
        # - tools: Tavily 검색 도구 등록
        # - toolsets: mcp-server-fetch 도구 등록
        agent = Agent(
            model_id,
            output_type=Company,
            system_prompt=it_system_prompt,
            retries=3,
            tools=[tavily_search_tool(TAVILY_API_KEY)],  # Tavily 검색 도구 등록
            toolsets=[fetch_server]                      # mcp-server-fetch 도구 등록
        )

        # output_validator: has_official=True인데 channel_url이 None이면 재시도
        @agent.output_validator
        def validate_output(ctx, output: Company) -> Company:
            if output.has_official and not output.channel_url:
                raise ModelRetry(
                    f"{output.company_name}의 has_official이 True인데 "
                    f"channel_url이 없습니다. 채널 URL을 찾아주세요."
                )
            return output

        # 진행률 바 생성
        settings = GoogleModelSettings(temperature=0.3)
        pbar = tqdm(total=len(pending), desc="채널 탐색")

        # 순차 처리
        for name in pending:
            await analyze_one(agent, name, settings, stats, all_results, total_tokens, pbar)

        pbar.close()

    # 최종 체크포인트 저장
    save_checkpoint(all_results)

    return all_results, stats, total_tokens
        
# ========================================
# 6. 성능 테스트
# ========================================

# async def main():
#     # 테스트용 기업 리스트 (Agent 성능 테스트용 샘플 데이터)
#     test_companies = [
#         "삼성전자",
#         "LG전자"
#     ]
    
async def main():
    # 실제 기업 리스트 
    # window에서 만든 한글 csv 파일 인코딩: cp949
    df = pd.read_csv("./data/IT_company_list.csv", encoding="cp949")
    company_names = df['name'].tolist()  # 기업명 컬럼명에 맞게 수정

    print(f"전체 기업 수: {len(company_names)}개")
    print("=" * 60)

    # Agent 호출
    # all_results, stats, total_tokens = await run_agent(test_companies) # 테스트용
    all_results, stats, total_tokens = await run_agent(company_names) # 실전

    # 처리 결과 요약
    print()
    print("=" * 60)
    print("배치 처리 완료 (순차 처리 방식)")
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
        print(f"{i}. {status} {company['company_name']} | {company['channel_name']} | {company['channel_url']}")
        print(f"   └── {company['reason']}")


# ========================================
# 7. main 실행 블록
# ========================================
if __name__ == '__main__':
    # asyncio.run()은 프로그램 전체에서 딱 한 번만 호출하는 것이 원칙
    # 이유: asyncio.run()은 새 이벤트 루프 생성 → 코루틴 실행 → 루프 종료를
    #       한 번에 처리하므로, 여러 번 호출하면 루프 충돌이 발생할 수 있음
    # __name__ == '__main__': 터미널에서 이 파일을 직접 실행할 때만 루프가 돌아가도록 함
    asyncio.run(main())  # 비동기 함수의 진입점 - 최초 1회만 실행