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
from pydantic_ai import Agent, ModelRetry

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
parser = argparse.ArgumentParser()

parser.add_argument('csv_path')                   # 필수 위치 인자: video_agent 결과 CSV 경로
parser.add_argument("--concurrent", type=int, default=3)  # 동시 실행 수 (기본값 3)
parser.add_argument("--delay", type=int, default=10)      # 기본 대기 시간 (기본값 10초)
parser.add_argument("--output_dir", type=str,             # 결과 저장 폴더 위치 지정
    default="works/Haeun/video_analysis/reports")

args = parser.parse_args()
csv_stem = Path(args.csv_path).stem

# ========================================
# 4. 응답 스키마 정의
# ========================================
# TODO: 집계 결과 스키마 작성

# ========================================
# 5. 체크포인트 함수 정의
# ========================================
CHECKPOINT_FILE = Path(f"works/Haeun/video_analysis/checkpoints/checkpoint_aggregator_{csv_stem}.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

CHECKPOINT_EVERY = 5
MAX_RETRIES = 4
BASE_DELAY = args.delay
MAX_DELAY = 60
MAX_CONCURRENT = args.concurrent

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
# 6. CSV 로드 및 집계 함수 정의
# ========================================
# TODO: CSV 읽어서 채널 단위 집계
# TODO: 도메인 단위 (IT vs F&B) 비교

def aggregate_results(csv_path: str) -> list | None:
    pass

# ========================================
# 7. 지수 백오프 및 병렬 처리 함수 정의
# ========================================
async def analyze_one(
    agent,
    data,
    settings,
    stats,
    all_results,
    total_tokens,
    pbar,
):
    async with sem:
        # TODO: 집계 결과를 Gemini에 넘기는 프롬프트 작성 예정
        prompt = None

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
                    print(f"⚠️ 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기)")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ [최종 실패] | {error_msg[:100]}")
                    stats['fail'] += 1
                    pbar.update(1)
                    pbar.set_postfix(성공=stats['success'], 실패=stats['fail'])

# ========================================
# 8. 시스템 프롬프트
# ========================================
# TODO: 집계 결과 해석용 시스템 프롬프트 작성
system_prompt = None

# ========================================
# 9. Agent 실행 함수
# ========================================
async def run_agent(aggregated_data: list) -> tuple:
    all_results = load_checkpoint()
    # ⚠️ 주의: aggregated_data의 각 딕셔너리에 'channel_name' 키가 반드시 있어야 함
    # → 섹션 6 aggregate_results() 작성 시 'channel_name' 키 포함 필수
    processed = {r.get('channel_name') for r in all_results}
    pending = [d for d in aggregated_data if d.get('channel_name') not in processed]

    if processed:
        print(f"스킵: {len(processed)}개 (이미 처리됨)")
    print(f"처리 대상: {len(pending)}개")
    print(f"처리 방식: 병렬 처리 (최대 {MAX_CONCURRENT}개 동시 실행)")
    print(f"재시도: 최대 {MAX_RETRIES}회 (지수 백오프, 기본 {BASE_DELAY}초)")
    print("=" * 60)

    stats = {'success': len(all_results), 'fail': 0}
    total_tokens = {'input': 0, 'output': 0}

    # Agent 초기화
    agent = Agent(
        model_id,
        output_type=None,  # TODO: 집계 결과 스키마로 교체
        system_prompt=system_prompt,
        retries=3
    )

    settings = GoogleModelSettings(temperature=0.3)
    pbar = tqdm(total=len(pending), desc="집계 분석")

    tasks = [
        analyze_one(agent, data, settings, stats, all_results, total_tokens, pbar)
        for data in pending
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    save_checkpoint(all_results)
    return all_results, stats, total_tokens

# ========================================
# 10. main 실행 블록
# ========================================
async def main():
    aggregated_data = aggregate_results(args.csv_path)
    if aggregated_data is None:
        print("❌ 집계 실패")
        return

    print(f"결과 집계 완료: {len(aggregated_data)}개 채널")
    print("=" * 60)

    all_results, stats, total_tokens = await run_agent(aggregated_data)

    print()
    print("=" * 60)
    print("집계 완료 (병렬 처리 방식)")
    print(f"  성공: {stats['success']}개")
    print(f"  실패: {stats['fail']}개")
    print(f"  성공률: {stats['success'] / max(stats['success'] + stats['fail'], 1) * 100:.1f}%")
    print(f"  체크포인트: {CHECKPOINT_FILE}")
    print("=" * 60)

    # TODO: 집계 결과 저장 형식 결정 후 추가
    output_path = Path(args.output_dir) / f"aggregated_{csv_stem}.csv"
    pd.DataFrame(all_results).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n집계 결과 저장 완료: {output_path}")
    print(f"  입력 토큰: {total_tokens['input']:,} / 출력 토큰: {total_tokens['output']:,}")

# 해당 파일을 직접 호출해서 실행하면 실행되도록 하는 함수
if __name__ == '__main__':
    asyncio.run(main())