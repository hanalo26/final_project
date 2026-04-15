# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import time
import json
import csv
import asyncio
from pprint import pprint
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.tools import Tool
from pydantic_ai.mcp import MCPServerStdio                      # MCP 서버 연결 (mcp-server-fetch)
from pydantic_ai.common_tools.tavily import tavily_search_tool  # Tavily 검색 도구 직접 연결
from pydantic_ai.models.google import GoogleModelSettings       # 모델 설정 (temperature 등)

# ========================================
# 2. 환경변수 로드 및 API 연결 확인
# ========================================

load_dotenv(dotenv_path="../../.env")

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # 기본값 설정

# PydanticAI는 GEMINI_API_KEY 환경변수를 자동으로 인식
# 모델 ID 형식: 'google-gla:{모델명}'
model_id = f"google-gla:{GEMINI_MODEL}"

# API 키 유효성 검사
api_key_valid = GEMINI_API_KEY and "YOUR_API_KEY" not in GEMINI_API_KEY
print(f"Gemini API 키 설정 확인: {'✅' if api_key_valid else '❌'}")
if not api_key_valid:
    print("[주의] .env 파일에서 GEMINI_API_KEY를 실제 API 키로 설정해주세요!")
print(f"사용할 모델: {model_id}")

# YouTube Data API 연결 확인
try:
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={
            "part":      "snippet",
            "forHandle": "@YouTube",
            "key":       YOUTUBE_API_KEY,
        }
    )
    print(f"\nYouTube Data API 연결: {'✅' if response.status_code == 200 else '❌'}")
except Exception as e:
    print(f"\nYouTube Data API 연결: ❌ ({e})")

# Tavily API 연결 확인
try:
    response = requests.post(
        "https://api.tavily.com/search",
        json={"query": "test", "api_key": TAVILY_API_KEY}
    )
    print(f"\nTavily API 연결: {'✅' if response.status_code == 200 else '❌'}")
except Exception as e:
    print(f"\nTavily API 연결: ❌ ({e})")

# API 호출 간격 (초)
API_DELAY = 3

# ========================================
# 3. 응답 스키마 정의
# ========================================

"""
응답 스키마 구조
====================================
최상위 구조: CompanyListResponse
├── companies (list[Company]): 기업 리스트
│   └── Company (BaseModel): 기업 1개
│       ├── company_name (str): 기업명
│       ├── channel_id (Optional[str]): 유튜브 채널 ID (UCxxxxxxx 형식)
│       ├── channel_name (Optional[str]): 유튜브 채널명
│       ├── channel_handle (Optional[str]): 유튜브 채널 핸들 (@handle 형식)
│       ├── channel_url (Optional[str]): 유튜브 채널 URL
│       └── is_official (bool): 공식 채널 여부
└── reason (str): 공식 채널 판별 기준 요약 (최소 10자 이상)

참고:
- [] 는 "리스트 안의 각 항목"을 의미하는 표기 관례로, API 문서나 JSON Schema에서 흔히 쓰는 방식
- Optional 필드는 공식 채널을 찾지 못한 경우 None으로 저장
"""

# 개별 기업의 유튜브 채널 정보에 대한 응답 스키마
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
    is_official: bool = Field(
        description="공식 채널 여부 (True/False)"
    )

# 전체 기업 리스트에 대한 응답 스키마
class CompanyListResponse(BaseModel):
    companies: list[Company] = Field(
        description="기업 리스트"
    )
    reason: str = Field(
        description="공식 채널 판별 기준 요약",
        min_length=10
    )
    
# ========================================
# 4. 도구 함수 정의
# ========================================

def is_official_channel(company_name: str, page_content: str) -> bool:
    """
    3단계: 순수 파이썬 로직으로 채널이 공식 채널인지 최종 판별
    - company_name: 기업명
    - page_content: mcp-server-fetch로 가져온 채널 페이지 내용
    아래 조건 중 2개 이상 충족 시 공식 채널로 판별
    """
    # 영문 대소문자 구분 없이 비교하기 위해 소문자로 변환
    # 한국어는 대소문자 개념이 없으므로 그대로 반환되고, 영문만 소문자로 변환됨
    content_normalized = page_content.lower()
    company_normalized = company_name.lower()

    # 조건 1: 페이지 내용에 "공식" 또는 "official" 키워드 포함 여부
    has_official_keyword = any(
        keyword in content_normalized
        for keyword in ["공식", "official", "official channel", "공식 채널"]
    )

    # 조건 2: 페이지 내용에 기업명 포함 여부
    has_company_name = company_normalized in content_normalized

    # 조건 3: 페이지 내용에 홈페이지 링크 포함 여부
    has_homepage_link = any(
        keyword in content_normalized
        for keyword in [".com", ".co.kr", ".kr", "홈페이지", "website"]
    )

    # 모든 조건을 점수화하여 2개 이상 충족 시 공식 채널로 판별
    # 각 조건을 만족할 때마다 1점씩 부여
    score = sum([
        has_official_keyword,
        has_company_name,
        has_homepage_link
    ])

    return score >= 2  # 3개 조건 중 2개 이상 충족 시 공식 채널로 판별

# ========================================
# 5. 시스템 프롬프트 및 Agent 초기화
# ========================================

it_system_prompt = """
너는 기업의 공식 유튜브 채널을 찾는 데이터 전문가야.
입력된 기업 리스트를 바탕으로 각 기업의 공식 유튜브 채널을 찾아줘.

[공식 채널 탐색 방법]
각 기업에 대해 반드시 아래 순서로 진행해줘.

1. tavily_search 도구로 "{기업명} 공식 유튜브 채널" 검색 후 유튜브 채널 URL 추출
2. fetch 도구로 찾은 URL이 실제 접속 가능한지 확인 및 페이지 내용 수집
3. is_official_channel 도구로 해당 채널이 공식 채널인지 최종 판별
    - 공식 채널이 아닌 경우: 검색어를 바꿔서 1번부터 다시 시도
    - 3번 재시도 후에도 공식 채널을 찾지 못한 경우: is_official=False로 저장

[출력 형식]
1. 공식 채널을 찾은 경우
    - 채널 정보를 정해진 스키마에 맞게 저장하고, is_official=True로 저장
2. 공식 채널을 찾지 못한 경우
    - channel 관련 필드는 None으로 저장하고, is_official=False로 저장
3. 모든 기업에 대한 처리가 끝나면 공식 채널 판별 기준을 reason에 요약해서 출력
"""


async def run_agent(company_names: list) -> CompanyListResponse:
    # mcp-server-fetch 서버 설정 (함수 안에서 정의 → 호출마다 새로운 서버 인스턴스 생성)
    # - uvx로 "mcp-server-fetch" 패키지를 실행하는 MCP 서버 객체
    # - Agent가 유튜브 채널 URL에 접속해서 페이지 내용을 수집할 때 사용
    fetch_server = MCPServerStdio(
        'uvx',                          # Python 패키지 러너 (npx의 Python 버전으로, mcp-server-fetch가 python 패키지이기 때문에 사용함)
        args=['mcp-server-fetch'],      # 실행할 MCP 서버 패키지명 (웹페이지 내용 수집용)
        timeout=30,                     # 서버 시작 대기 시간(초). 첫 실행 시 패키지 다운로드로 오래 걸릴 수 있음
    )

    # MCP 서버 실행 
    # - async with 블록이 끝나면 서버 자동 종료 -> 에러가 발생해도 자동 종료
    async with fetch_server:
        # Agent 초기화
        # - model_id: 사용할 Gemini 모델
        # - output_type: 응답 스키마 지정 → 자동으로 타입 검증 및 파싱
        # - system_prompt: 역할 및 조건 정의
        # - retries: 응답 제약 조건을 만족하지 못할 경우 최대 재시도 횟수
        # - tools: Tavily 검색, 공식 채널 판별 도구 등록
        # - toolsets: mcp-server-fetch 도구 등록
        agent = Agent(
            model_id,
            output_type=CompanyListResponse,
            system_prompt=it_system_prompt,
            retries=3,
            tools=[
                tavily_search_tool(TAVILY_API_KEY),          # Tavily 검색 도구 등록
                Tool(is_official_channel, takes_ctx=False),  # 공식 채널 판별 도구 등록
            ],
            toolsets=[fetch_server],                         # mcp-server-fetch 도구 등록
        )

        # output_validator: 응답값의 논리적 일관성 검증
        # - is_official=True인데 channel_url이 None이면 재시도 요청
        @agent.output_validator
        def validate_output(ctx, output: CompanyListResponse) -> CompanyListResponse:
            for company in output.companies:
                if company.is_official and not company.channel_url:
                    raise ModelRetry(
                        f"{company.company_name}의 is_official이 True인데 "
                        f"channel_url이 없습니다. 채널 URL을 찾아주세요."
                    )
            return output

        # Agent 호출 - MCP 서버의 수명주기가 자동으로 관리 <- async with fetch_server 내부에서 실행되기 때문에 실행/종료가 자동으로 수행됨
        return await agent.run(
            f"다음 기업들의 공식 유튜브 채널을 찾아줘: {company_names}",
            model_settings=GoogleModelSettings(temperature=0.3)
        )
        
# ========================================
# 6. 성능 테스트
# ========================================
async def main():
    # 테스트용 기업 리스트 (Agent 성능 테스트용 샘플 데이터)
    test_companies = [
        "삼성전자",
        "LG전자",
        "카카오",
        "네이버",
        "SK텔레콤",
        "카카오페이",
        "네이버페이"
    ]

    print("Agent 성능 테스트 시작...")
    print("=" * 60)

    # Agent 호출
    result = await run_agent(test_companies)

    # 응답 스키마 구조에 맞게 companies와 reason 분리
    company_list = result.output.companies
    reason       = result.output.reason

    # 수업 코드 스타일 - 구조화된 결과 출력
    print("\n--- [결과] 공식 유튜브 채널 탐색 결과 ---")
    pprint(result.output)

    print("\n--- [결과] 기업별 요약 ---")
    print(f"처리된 기업 수: {len(company_list)}개")
    for i, company in enumerate(company_list, start=1):
        status = "✅" if company.is_official else "❌"
        print(f"{i}. {status} {company.company_name} | {company.channel_name} | {company.channel_url}")

    print(f"\n--- [결과] 판별 기준 ---")
    print(reason)
    print("=" * 60)

# ========================================
# 7. main 실행 블록
# ========================================
if __name__ == '__main__':
    # asyncio.run()은 프로그램 전체에서 딱 한 번만 호출하는 것이 원칙
    # 이유: asyncio.run()은 새 이벤트 루프를 생성 → 코루틴 실행 → 루프 종료를
    #       한 번에 처리하므로, 여러 번 호출하면 루프 충돌이 발생할 수 있음
    asyncio.run(main()) # 비동기 함수의 진입점 - 최초 1회만 실행