# 목표: 댓글 긍/부정 분석 (LLM)

# ========================================
# 1. 라이브러리 로드
# ========================================
import os
import warnings
import platform
import json
import asyncio
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from pydantic_ai import Agent, ModelRetry

# Vertex AI 연결
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

# 프로젝트 루트를 import 경로에 추가
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

# 운영체제별 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

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
    
# 데이터 파일을 터미널에서 변수처럼 전달해주기 위해 필요한 변수 정의
parser = argparse.ArgumentParser()
parser.add_argument('csv_path')     # 필수인자 : 댓글 데이터가 담긴 csv 파일의 경로

# 옵션 인자 추가
parser.add_argument("--concurrent", type=int, default=3) # 동시 실행 수 (기본값 3, 미입력 시 기본값 사용)
parser.add_argument("--delay", type=int, default=10)     # 기본 대기 시간 (기본값 10초, 미입력 시 기본값 사용)

parser.add_argument('--video-id-col', default='video_id')     # 비디오_id가 담긴 컬럼의 이름
parser.add_argument('--comment-col', default='comment')       # 댓글의 내용이 담긴 컬럼의 이름
parser.add_argument('--comment-id-col', default='comment_id') # 댓글_id가 담긴 컬럼의 이름

args = parser.parse_args()      # 터미널에서 입력한 인자를 파싱하여 args 객체에 저장

# ========================================
# 3. 응답 스키마 정의
# ========================================
"""
응답 스키마 구조
====================================
CommentSentiment (BaseModel)
 comment_id  (str)       : 댓글 고유 ID
 video_id    (str)       : 영상 고유 ID
 sentiment   (Literal)   : 댓글의 감성 레이블 (긍정 / 부정 / 중립)
 reason      (str)       : 위와 같은 감성으로 분류한 근거 (15자 이상)
"""

class CommentSentiment(BaseModel):
    comment_id: str = Field(
        description="댓글의 고유 ID. 입력 데이터에서 그대로 가져온다."
    )
    video_id: str = Field(
        description="댓글이 달린 영상의 고유 ID. 입력 데이터에서 그대로 가져온다."
    )
    sentiment: Literal["긍정", "부정", "중립"] = Field(
        description="댓글의 맥락을 분석하여 긍정적인 댓글인지, 부정적인 댓글인지 판단한다."
    )
    reason: str = Field(
        description="위 감성으로 분류한 근거를 15자 이상의 문장으로 설명한다.",
        min_length=15
    )
    
# ========================================
# 4. 체크포인트 및 지수 백오프 함수 정의
# ========================================
csv_stem = Path(args.csv_path).stem

# 체크포인트 파일명을 CSV 파일명 기반으로 동적 생성
CHECKPOINT_FILE = Path(f"./outputs/checkpoint_{csv_stem}.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

CHECKPOINT_EVERY = 5              # N건마다 중간 저장
MAX_RETRIES = 4                   # 최대 재시도 횟수
BASE_DELAY = args.delay           # 기본 대기 시간 (초)
MAX_DELAY = 60                    # 최대 대기 시간 (초)
MAX_CONCURRENT = args.concurrent  # 동시 API 호출 수  

# Semaphore() 생성
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
    pbar
):
    """
    Semaphore로 동시 실행을 제한하면서 댓글 1개를 처리하고,
    실패 시 지수 백오프로 재시도
    """
    async with sem:
        # LLM의 텍스트 분류 판단의 근거가 되는 입력 데이터 (댓글 데이터 입력)
        prompt = f"""
        댓글 ID  : {row[args.comment_id_col]}
        영상 ID  : {row[args.video_id_col]}
        댓글 내용: {row[args.comment_col]}
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
                    print(f"⚠️ [{row[args.comment_id_col]}] 재시도 {attempt+1}/{MAX_RETRIES} ({delay}초 대기)")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ [최종 실패] {row[args.comment_id_col]} | {error_msg[:100]}")
                    stats['fail'] += 1
                    pbar.update(1)
                    pbar.set_postfix(성공=stats['success'], 실패=stats['fail'])
                    
# ========================================
# 5. 시스템 프롬프트 및 Agent 초기화
# ========================================
system_prompt = """
너는 유튜브 댓글의 감성을 분석하는 전문가야.
입력된 댓글을 읽고 아래 규칙에 따라 감성을 분류해.

[분류 기준]
- 긍정: 칭찬, 감사, 공감, 기대, 만족 등 긍정적인 감정이 담긴 댓글.
        "ㅋㅋㅋㅋ", "ㄹㅇ", "인정" 등 짧은 공감·웃음 표현도 긍정으로 분류해
- 부정: 불만, 비판, 실망, 분노, 악의적 표현이 담긴 댓글.
        제품/서비스 이용 중 불편한 점, 개선 요청, 불만 사항을 언급한 댓글도 부정으로 분류해
- 중립: 단순 질문, 정보 요청, 감정이 드러나지 않는 사실적 댓글.
        출시 일정, 환불, 사용법 등 문의성 댓글은 부정적 감정이 없으면 중립으로 분류해.
        "오", "아", "ㅇㅎ", "그렇군요" 등 단순 인지·수용 반응도 중립으로 분류해

[주의사항]
- 이모지, 줄임말, 신조어도 문맥에 맞게 해석해
- 비꼬는 표현(예: "와 진짜 대단하네요~")은 문맥을 보고 부정으로 분류해
- 이모티콘이나 말투만으로 감성을 판단하지 말고 댓글의 핵심 내용을 기준으로 분류해.
- 긍정 이모티콘(😊❤️ 등)이나 부드러운 경어체로 마무리했더라도 개선 요청, 불편함, 기술 개발 바람, 아쉬움이 담겨있으면 부정으로 분류해.
- 긍정과 부정이 혼재된 댓글은 핵심 내용을 기준으로 분류해.
- 칭찬보다는 불만·개선 요청이 주된 내용이면 부정으로, 불만보다는 만족·공감이 주된 내용이면 긍정으로 분류해.
- "다음엔 더 잘해주세요", "예전이 더 좋았는데", "기대했는데 ㅠㅠ" 등과 같이 아쉬움·실망·간접 비교가 담긴 댓글은 부정으로 분류해
- 개선 요청이나 불편함을 언급한 뒤 응원·기원으로 마무리한 댓글은 부정으로 분류해
- "저만 이런가요?", "이거 맞나요?", "음...", "글쎄요" 등 불신·의심·부정적 뉘앙스가 담긴 표현은 반드시 문맥을 보고 판단해
- "미쳤다", "말도 안 된다", "어이없어", "이거 실화냐", "죽겠다", "미치겠다" 등은 한국어에서 극찬·감탄으로 쓰이는 경우가 많으므로 반드시 문맥을 보고 긍정/부정을 판단해
- "역시", "그럼 그렇지" 등 긍정·부정 모두 해석 가능한 표현은 앞뒤 문맥을 보고 판단해
- 댓글 이벤트의 빈칸 정답, 퀴즈 답변 등 이벤트 참여 목적으로 작성된 댓글은 단어 자체의 의미와 관계없이 중립으로 분류해.
  단, 정답 외에 이벤트와 무관한 독립적인 감정 표현이 있으면 해당 감정을 기준으로 분류해.
  (예: 정답 + 형식적 응원 문구 → 중립 / 정답 + 실제 불만·개선 요청 → 부정 / 정답 + 진심 어린 칭찬 → 긍정)
- 광고성 댓글이나 의미 없는 반복 문자는 중립으로 분류해
- 19금 댓글이 섞여있다면 분석에서 제외해
- comment_id와 video_id는 분류하지 말고 입력값 그대로 출력해
"""

async def run_agent(df: pd.DataFrame) -> tuple:
    all_results  = load_checkpoint()
    processed_ids = {r['comment_id'] for r in all_results}

    pending = [row for _, row in df.iterrows() if row[args.comment_id_col] not in processed_ids]

    if processed_ids:
        print(f"스킵: {len(processed_ids)}개 (이미 처리됨)")
    print(f"처리 대상: {len(pending)}개")
    print(f"처리 방식: 병렬 처리 (최대 {MAX_CONCURRENT}개 동시 실행)")
    print(f"재시도: 최대 {MAX_RETRIES}회 (지수 백오프, 기본 {BASE_DELAY}초)")
    print("=" * 60)

    stats = {'success': len(all_results), 'fail': 0}
    total_tokens = {'input': 0, 'output': 0}

    agent = Agent(
        model_id,
        output_type=CommentSentiment,
        system_prompt=system_prompt,
        retries=3
    )

    # output_validator: 필수값 및 감성 레이블 검증 (논리적 일관성을 검증하는 단계)
    @agent.output_validator
    def validate_output(ctx, output: CommentSentiment) -> CommentSentiment:
        if not output.comment_id:
            raise ModelRetry("comment_id가 비어있습니다. 입력 데이터의 comment_id를 그대로 넣어주세요.")
        if not output.video_id:
            raise ModelRetry("video_id가 비어있습니다. 입력 데이터의 video_id를 그대로 넣어주세요.")
        if output.sentiment not in ("긍정", "부정", "중립"):
            raise ModelRetry("sentiment는 '긍정', '부정', '중립' 중 하나를 선택해야 합니다.")
        return output

    # 진행률 바 생성
    settings = GoogleModelSettings(temperature=0.3)
    pbar = tqdm(total=len(pending), desc="댓글 감성 분석")

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
# 6. Agent 작업 실행
# ========================================

def save_results(all_results: list, csv_stem: str) -> pd.DataFrame:
    """분석 결과를 CSV로 저장하고 DataFrame 반환"""
    df_result = pd.DataFrame(all_results)
    output_path = Path(f"./outputs/sentiment_{csv_stem}.csv")
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장 완료: {output_path}")
    print(f" ㄴ 행: {len(df_result)}개 / 열: {len(df_result.columns)}개")
    return df_result

def visualize(df_result: pd.DataFrame, csv_stem: str) -> None:
    """전체 감성 분포 + 영상별 감성 비율 시각화"""

    SENTIMENT_ORDER  = ["긍정", "중립", "부정"]
    SENTIMENT_COLORS = {"긍정": "#4CAF50", "중립": "#9E9E9E", "부정": "#F44336"}

    # 영상별 감성 비율 계산
    grouped = (
        df_result
        .groupby(["video_id", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    
    # .transform() : 집계 결과를 원래 행 수에 맞춰 늘려서 반환하는 메서드
    total_per_video = grouped.groupby("video_id")["count"].transform("sum")
    grouped["ratio"] = grouped["count"] / total_per_video * 100

    pivot = (
        grouped
        .pivot_table(index="video_id", columns="sentiment", values="ratio", fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)  # 열 순서 고정
        # fill_value=0의 의미 : NaN값이 있다면 0으로 채움 -> 그래프 그릴 때, 혹시 발생할 에러 예방
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(pivot) * 0.6 + 2)))
    fig.suptitle("유튜브 댓글 감성 분석 결과", fontsize=15, fontweight="bold", y=1.01)

    # 차트 1: 전체 감성 분포 (파이차트)
    ax1 = axes[0]
    total = df_result["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total_sum = total.sum()

    wedges, texts = ax1.pie(
        total,
        labels=None,
        colors=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    for wedge, sentiment, val in zip(wedges, SENTIMENT_ORDER, total):
        pct = val / total_sum * 100 if total_sum > 0 else 0
        if pct == 0:
            continue

        # 슬라이스 중심 각도 → x, y 방향 계산
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))

        if pct < 5:  # 작은 슬라이스 → 외부에 선 연결
            ax1.annotate(
                f"{pct:.1f}%",
                xy=(x * 0.85, y * 0.85),       # 선 시작점 (파이 위)
                xytext=(x * 1.25, y * 1.25),   # 텍스트 위치 (외부)
                arrowprops=dict(arrowstyle="-", color="gray", lw=1),
                ha="center", va="center",
                fontsize=8,
            )
        else:         # 큰 슬라이스 → 내부에 표시
            ax1.text(
                x * 0.65, y * 0.65,
                f"{pct:.1f}%",
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
            )

    ax1.set_title("전체 데이터에서의 댓글 감성 분포", fontsize=15, pad=12)
    
    ax1.legend(
        wedges,
        SENTIMENT_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=10,
    )

    # 차트 2: 영상별 감성 비율 (히트맵)
    ax2 = axes[1]

    SENTIMENT_ORDER_DISPLAY = ["긍정", "중립", "부정"]
    x_pos = np.arange(len(SENTIMENT_ORDER_DISPLAY))
    video_ids = pivot.index.tolist()
    y_pos = np.arange(len(video_ids))

    for j, sentiment in enumerate(SENTIMENT_ORDER_DISPLAY):
        for i, video_id in enumerate(video_ids):
            val = pivot.loc[video_id, sentiment]
            alpha = max(0.1, val / 100)
            base_color = SENTIMENT_COLORS[sentiment]

            ax2.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                color=base_color, alpha=alpha
            ))
            if val > 0:
                ax2.text(
                    j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=9,
                    color="white" if val >= 30 else "black",
                    fontweight="bold",
                )

    ax2.set_xlim(-0.5, len(SENTIMENT_ORDER_DISPLAY) - 0.5)
    ax2.set_ylim(-0.5, len(video_ids) - 0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(SENTIMENT_ORDER_DISPLAY, fontsize=11)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(video_ids, fontsize=9)
    ax2.set_title("영상별 댓글의 감성 비율", fontsize=13)
    ax2.invert_yaxis()
    sns.despine(ax=ax2, left=True, bottom=True)
    ax2.tick_params(left=False, bottom=False)

    plt.tight_layout()
    chart_path = Path(f"./outputs/sentiment_{csv_stem}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 차트 저장 완료: {chart_path}")
    
# ========================================
# 7. main 실행 블록
# ========================================

async def main():
    df = pd.read_csv(args.csv_path, encoding="utf-8")

    print(f"전체 댓글 수: {len(df)}개")
    print("=" * 60)

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

    # CSV 저장
    df_result = save_results(all_results, csv_stem)

    # 시각화
    visualize(df_result, csv_stem)

    # 토큰 사용량 출력
    print(f"\n  입력 토큰: {total_tokens['input']:,} / 출력 토큰: {total_tokens['output']:,}")
    if stats['success'] > 0:
        print(f"  1건 평균: 입력 {total_tokens['input'] / stats['success']:.0f} / 출력 {total_tokens['output'] / stats['success']:.0f} tokens")

if __name__ == '__main__':
    asyncio.run(main())