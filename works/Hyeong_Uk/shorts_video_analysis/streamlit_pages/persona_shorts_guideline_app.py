# works/Hyeong_Uk/streamlit_pages/persona_shorts_guideline_app.py

import os
import re
import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from yt_dlp import YoutubeDL

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider


# ============================================================
# 0. 기본 설정
# ============================================================

st.set_page_config(
    page_title="페르소나 기업 맞춤형 숏츠 영상 진단",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
/* 전체 배경 */
.stApp {
    background-color: #0F0F0F;
    color: #FFFFFF;
}

/* 제목 */
h1, h2, h3 {
    color: #FFFFFF;
}

/* 일반 텍스트 */
p, div, span {
    color: #EAEAEA;
}

/* 카드 스타일 */
.yt-card {
    background-color: #212121;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #333333;
    margin-bottom: 16px;
}

/* 빨간 포인트 카드 */
.yt-highlight {
    background: linear-gradient(135deg, #FF0000 0%, #B00000 100%);
    padding: 18px;
    border-radius: 16px;
    color: white;
    font-weight: 600;
    margin-bottom: 16px;
}

/* 작은 설명 */
.yt-muted {
    color: #AAAAAA;
    font-size: 14px;
}

/* 버튼 느낌 */
.stButton > button {
    background-color: #FF0000;
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #CC0000;
    color: white;
}

/* metric 카드 */
[data-testid="stMetric"] {
    background-color: #212121;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #333333;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #181818;
}

/* dataframe */
[data-testid="stDataFrame"] {
    background-color: #212121;
}
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(".")
AGENT_PATH = PROJECT_ROOT / "works/Hyeong_Uk/shorts_video_analysis/Video-Analysis_agent.py"

load_dotenv(dotenv_path=".env", override=True)

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

BASELINE_PATH = PROJECT_ROOT / "works/Hyeong_Uk/shorts_video_analysis/results/result_sample_shorts_all_for_video_agent_fixed.csv"

PERSONA_BASE_DIR = PROJECT_ROOT / "works/Hyeong_Uk/persona_shorts_guideline"
PERSONA_DATA_DIR = PERSONA_BASE_DIR / "data"
PERSONA_RESULT_DIR = PERSONA_BASE_DIR / "results"
PERSONA_VIDEO_DIR = PERSONA_BASE_DIR / "videos"
PERSONA_REPORT_DIR = PERSONA_BASE_DIR / "reports"

for d in [PERSONA_DATA_DIR, PERSONA_RESULT_DIR, PERSONA_VIDEO_DIR, PERSONA_REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 유틸 함수
# ============================================================

def safe_filename(text: str) -> str:
    """파일명으로 쓰기 안전하게 문자열 정리"""
    text = str(text).strip()
    text = re.sub(r"[^\w가-힣\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:60] or "persona_channel"


def extract_video_id(url_or_id: str) -> str:
    """유튜브 URL 또는 id에서 video_id 추출"""
    s = str(url_or_id)

    patterns = [
        r"shorts/([A-Za-z0-9_-]{11})",
        r"watch\?v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]

    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    return ""


def is_youtube_url(text: str) -> bool:
    return "youtube.com" in text or "youtu.be" in text


def to_shorts_tab_url(channel_url: str) -> str:
    """채널 URL을 shorts 탭 URL로 변환"""
    url = channel_url.strip().rstrip("/")

    if "/shorts" in url:
        return url

    if "/videos" in url:
        return url.replace("/videos", "/shorts")

    return url + "/shorts"


@st.cache_data(show_spinner=False)
def load_baseline() -> pd.DataFrame:
    """기존 200개 숏츠 영상 분석 결과 로드"""
    if not BASELINE_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(BASELINE_PATH, encoding="utf-8-sig")

    # 안전 처리
    df["domain"] = df["domain"].astype(str)
    df["success_label"] = df["success_label"].astype(str)

    return df


def get_domain_success_baseline(baseline_df: pd.DataFrame, domain: str) -> dict:
    """
    도메인 성공 숏츠 기준 패턴 생성
    - 수치형 평균
    - 주요 범주형 비율
    """
    if baseline_df.empty:
        return {}

    success_df = baseline_df[
        (baseline_df["domain"] == domain) &
        (baseline_df["success_label"] == "success")
    ].copy()

    if success_df.empty:
        return {}

    def rate(col: str, value: str) -> float:
        if col not in success_df.columns:
            return np.nan
        return round((success_df[col] == value).mean(), 3)

    def mean(col: str) -> float:
        if col not in success_df.columns:
            return np.nan
        return round(success_df[col].mean(), 3)

    baseline = {
        "n_success": len(success_df),

        # numeric
        "person_ratio": mean("person_ratio"),
        "face_ratio": mean("face_ratio"),
        "text_ratio": mean("text_ratio"),
        "avg_brightness": mean("avg_brightness"),

        # category rates
        "first_3sec_인물": rate("first_3sec", "인물"),
        "first_3sec_텍스트": rate("first_3sec", "텍스트"),
        "motion_graphic_보조적": rate("motion_graphic", "보조적"),
        "motion_graphic_핵심요소": rate("motion_graphic", "핵심요소"),
        "video_format_기술설명": rate("video_format", "기술설명"),
        "video_format_제품리뷰": rate("video_format", "제품리뷰"),
        "video_format_웹예능": rate("video_format", "웹예능"),
        "video_format_웹드라마": rate("video_format", "웹드라마"),
    }

    return baseline


def summarize_persona_result(result_df: pd.DataFrame) -> dict:
    """페르소나 기업 최근 숏츠 분석 결과 요약"""
    if result_df.empty:
        return {}

    def rate(col: str, value: str) -> float:
        if col not in result_df.columns:
            return np.nan
        return round((result_df[col] == value).mean(), 3)

    def mode_value(col: str) -> str:
        if col not in result_df.columns or result_df[col].dropna().empty:
            return "-"
        return str(result_df[col].mode().iloc[0])

    def mean(col: str) -> float:
        if col not in result_df.columns:
            return np.nan
        return round(result_df[col].mean(), 3)

    summary = {
        "n": len(result_df),
        "main_first_3sec": mode_value("first_3sec"),
        "main_motion_graphic": mode_value("motion_graphic"),
        "main_video_format": mode_value("video_format"),

        "person_ratio": mean("person_ratio"),
        "face_ratio": mean("face_ratio"),
        "text_ratio": mean("text_ratio"),
        "avg_brightness": mean("avg_brightness"),

        "first_3sec_인물": rate("first_3sec", "인물"),
        "first_3sec_텍스트": rate("first_3sec", "텍스트"),
        "motion_graphic_보조적": rate("motion_graphic", "보조적"),
        "motion_graphic_핵심요소": rate("motion_graphic", "핵심요소"),
        "video_format_기술설명": rate("video_format", "기술설명"),
        "video_format_제품리뷰": rate("video_format", "제품리뷰"),
        "video_format_웹예능": rate("video_format", "웹예능"),
        "video_format_웹드라마": rate("video_format", "웹드라마"),
    }

    return summary


def judge_gap(current: float, baseline: float, higher_is_better: bool = True, threshold: float = 0.10) -> str:
    """
    현재 값과 성공 기준 차이 진단
    ratio 기준 threshold 0.10 = 10%p
    numeric ratio에도 일단 같은 방식 적용
    """
    if pd.isna(current) or pd.isna(baseline):
        return "판단 불가"

    diff = current - baseline

    if higher_is_better:
        if diff < -threshold:
            return "부족"
        elif diff > threshold:
            return "높음"
        else:
            return "유사"
    else:
        if diff > threshold:
            return "과다"
        elif diff < -threshold:
            return "낮음"
        else:
            return "유사"


def build_compare_table(domain: str, persona_summary: dict, baseline_summary: dict) -> pd.DataFrame:
    """도메인별 핵심 비교표 생성"""

    rows = []

    if domain == "FnB":
        compare_specs = [
            ("person_ratio", "인물 등장 비율", True),
            ("face_ratio", "얼굴 등장 비율", True),
            ("first_3sec_인물", "첫 3초 인물 비율", True),
            ("motion_graphic_보조적", "모션그래픽 보조 활용 비율", True),
            ("motion_graphic_핵심요소", "모션그래픽 핵심요소 비율", False),
        ]
    else:
        compare_specs = [
            ("text_ratio", "텍스트/자막 비율", True),
            ("first_3sec_텍스트", "첫 3초 텍스트 비율", True),
            ("motion_graphic_핵심요소", "모션그래픽 핵심요소 비율", True),
            ("video_format_기술설명", "기술설명형 포맷 비율", True),
            ("person_ratio", "인물 등장 비율", False),
        ]

    for key, label, higher_is_better in compare_specs:
        cur = persona_summary.get(key, np.nan)
        base = baseline_summary.get(key, np.nan)

        rows.append({
            "항목": label,
            "현재 채널": cur,
            "도메인 성공 패턴": base,
            "차이": round(cur - base, 3) if not pd.isna(cur) and not pd.isna(base) else np.nan,
            "진단": judge_gap(cur, base, higher_is_better=higher_is_better)
        })

    return pd.DataFrame(rows)


def build_improvement_tasks(domain: str, compare_df: pd.DataFrame) -> list[dict]:
    """비교표 기반 우선 개선 과제 생성"""
    tasks = []

    if domain == "FnB":
        for _, row in compare_df.iterrows():
            item = row["항목"]
            diagnosis = row["진단"]

            if item == "인물 등장 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "인물 경험 장면 강화",
                    "desc": "제품 단독 컷보다 사람이 제품을 먹거나 사용하는 장면을 늘려야 합니다.",
                    "action": "다음 숏츠는 첫 컷 또는 3초 안에 인물의 사용·섭취·반응 장면을 배치하세요."
                })
            elif item == "얼굴 등장 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "얼굴·반응 컷 강화",
                    "desc": "FnB 성공 숏츠는 얼굴과 반응이 더 자주 노출되는 경향이 있습니다.",
                    "action": "맛, 놀람, 만족, 호기심 등 감정이 보이는 얼굴 클로즈업을 포함하세요."
                })
            elif item == "첫 3초 인물 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "첫 3초 인물 후킹 강화",
                    "desc": "FnB 성공 패턴과 비교했을 때 초반 인물 등장 비율이 부족합니다.",
                    "action": "제품 설명 텍스트보다 인물이 제품을 경험하는 장면으로 시작하세요."
                })
            elif item == "모션그래픽 보조 활용 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "모션그래픽은 보조 장치로 활용",
                    "desc": "FnB에서는 그래픽보다 제품 경험 장면 자체가 중심이 되는 구성이 더 적합합니다.",
                    "action": "자막, 제품명, 반응 포인트 강조 정도로 모션그래픽을 제한하세요."
                })
            elif item == "모션그래픽 핵심요소 비율" and diagnosis == "과다":
                tasks.append({
                    "title": "과한 모션그래픽 비중 축소",
                    "desc": "FnB 성공 숏츠에서는 모션그래픽이 핵심요소인 비율이 낮게 나타났습니다.",
                    "action": "그래픽 연출보다 실제 제품 사용 장면과 인물 반응을 중심에 두세요."
                })

        # 기본 과제가 부족하면 보충
        if len(tasks) < 3:
            tasks.append({
                "title": "경험형 포맷 강화",
                "desc": "FnB 숏츠는 제품을 설명하기보다 경험하게 만드는 구성이 중요합니다.",
                "action": "제품리뷰, 웹예능형 반응 콘텐츠, 짧은 상황극 포맷을 우선 고려하세요."
            })

    else:
        for _, row in compare_df.iterrows():
            item = row["항목"]
            diagnosis = row["진단"]

            if item == "첫 3초 텍스트 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "첫 3초 텍스트 후킹 강화",
                    "desc": "IT 성공 숏츠는 초반에 핵심 메시지를 텍스트로 제시하는 경향이 있습니다.",
                    "action": "문제 상황, 기능명, 혜택을 첫 3초 안에 짧은 자막으로 제시하세요."
                })
            elif item == "모션그래픽 핵심요소 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "모션그래픽 기반 정보 시각화 강화",
                    "desc": "IT 성공 숏츠는 모션그래픽을 핵심요소로 활용하는 비율이 높았습니다.",
                    "action": "서비스 구조, 기능 흐름, 기술 개념을 아이콘·화면전환·인포그래픽으로 설명하세요."
                })
            elif item == "기술설명형 포맷 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "기술설명형 포맷 강화",
                    "desc": "IT 성공 숏츠에서는 기술설명형 포맷이 상대적으로 많이 나타났습니다.",
                    "action": "인터뷰형 긴 설명보다 문제-해결-기능 제시 흐름의 짧은 기술설명형 숏츠를 제작하세요."
                })
            elif item == "텍스트/자막 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "핵심 메시지 자막 강화",
                    "desc": "IT 콘텐츠는 짧은 시간 안에 기능과 효용을 이해시키는 것이 중요합니다.",
                    "action": "영상 전반에 핵심 기능, 수치, 혜택을 짧은 자막으로 보조하세요."
                })

        if len(tasks) < 3:
            tasks.append({
                "title": "정보 전달 구조 단순화",
                "desc": "IT 숏츠는 복잡한 내용을 짧은 시간에 이해시키는 구성이 중요합니다.",
                "action": "문제 제기 → 기능 시연 → 결과/효용 제시 흐름으로 구성하세요."
            })

    # 최대 3개만 반환
    return tasks[:3]


def build_rule_based_guideline(domain: str, persona_summary: dict, compare_df: pd.DataFrame, tasks: list[dict]) -> dict:
    """룰 기반 최종 가이드라인 생성"""

    if domain == "FnB":
        guideline = {
            "핵심 전략": "인물·경험 중심 숏츠를 강화하세요.",
            "첫 3초 전략": "제품 설명 텍스트보다 사람이 제품을 먹거나 사용하는 장면으로 시작하세요.",
            "추천 포맷": "제품리뷰, 웹예능형 반응 콘텐츠, 짧은 상황극, 제품 경험형 콘텐츠",
            "모션그래픽 활용": "모션그래픽은 핵심 연출보다 제품명, 반응 포인트, CTA를 강조하는 보조 장치로 활용하세요.",
            "피해야 할 구성": "텍스트 설명만으로 시작하거나, 그래픽이 제품 경험 장면을 가리는 구성은 지양하세요.",
            "체크리스트": [
                "첫 3초 안에 인물 또는 제품 경험 장면이 등장하는가?",
                "제품만 보여주기보다 사람이 먹거나 사용하는 장면이 있는가?",
                "얼굴, 반응, 행동이 명확히 보이는가?",
                "모션그래픽은 보조적으로 쓰이고 있는가?",
                "경험형/오락형 포맷을 활용했는가?",
            ],
            "다음 숏츠 기획안": [
                "인물이 제품을 처음 먹고 반응하는 15초 숏츠",
                "제품 사용 상황을 짧은 상황극으로 보여주는 숏츠",
                "제품의 장점을 하나만 잡아 반응 컷과 함께 보여주는 숏츠",
            ]
        }
    else:
        guideline = {
            "핵심 전략": "정보 시각화형 숏츠를 강화하세요.",
            "첫 3초 전략": "문제 상황, 기능명, 혜택을 짧은 텍스트로 먼저 제시하세요.",
            "추천 포맷": "기술설명형, 기능 시연형, 문제 해결형, 짧은 튜토리얼형 콘텐츠",
            "모션그래픽 활용": "서비스 구조, 작동 방식, 기능 흐름을 아이콘, 자막, 화면 전환으로 시각화하세요.",
            "피해야 할 구성": "핵심 메시지가 늦게 나오거나, 긴 인터뷰형 설명으로 시작하는 구성은 지양하세요.",
            "체크리스트": [
                "첫 3초 안에 핵심 메시지나 문제 상황이 텍스트로 제시되는가?",
                "기능이나 서비스 구조를 모션그래픽으로 시각화했는가?",
                "자막, 아이콘, 화면 전환이 정보 전달을 돕는가?",
                "인터뷰형 장황한 설명보다 짧은 기술설명형 구조인가?",
                "시청자가 3초 안에 영상 주제를 이해할 수 있는가?",
            ],
            "다음 숏츠 기획안": [
                "문제 상황을 텍스트로 던지고 기능으로 해결하는 15초 숏츠",
                "서비스 작동 방식을 모션그래픽으로 설명하는 숏츠",
                "사용 전/후 차이를 짧게 보여주는 기능 시연 숏츠",
            ]
        }

    return guideline


def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, sort="-y"),
            y=alt.Y(y),
            tooltip=list(df.columns)
        )
        .properties(title=title, height=260)
    )
    return chart

def render_video_card(row):
    title = row.get("title", "제목 없음")
    channel = row.get("채널명", "-")
    url = row.get("final_url", "")
    first_3sec = row.get("first_3sec", "-")
    motion = row.get("motion_graphic", "-")
    video_format = row.get("video_format", "-")
    person = row.get("person_ratio", "-")
    face = row.get("face_ratio", "-")
    text = row.get("text_ratio", "-")
    reason = row.get("reason", "-")

    st.markdown(f"""
    <div class="yt-card">
        <h4 style="margin-bottom: 6px;">{title}</h4>
        <p class="yt-muted">채널: {channel}</p>
        <p>
            <b>첫 3초:</b> {first_3sec} &nbsp; | &nbsp;
            <b>모션그래픽:</b> {motion} &nbsp; | &nbsp;
            <b>포맷:</b> {video_format}
        </p>
        <p>
            <b>person:</b> {person} &nbsp;
            <b>face:</b> {face} &nbsp;
            <b>text:</b> {text}
        </p>
        <p class="yt-muted">{reason}</p>
        <a href="{url}" target="_blank" style="color:#FF4B4B;">영상 열기</a>
    </div>
    """, unsafe_allow_html=True)

def generate_ai_guideline(prompt_text: str) -> str:
    """
    Gemini / Vertex AI를 사용해 맞춤형 숏츠 가이드라인 생성
    """
    if not GOOGLE_CLOUD_PROJECT or not GOOGLE_CLOUD_REGION or not GEMINI_MODEL:
        raise ValueError(
            "Google Cloud 환경변수가 설정되지 않았습니다. "
            "GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION, GEMINI_MODEL을 확인하세요."
        )

    provider = GoogleProvider(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_REGION,
    )

    model = GoogleModel(GEMINI_MODEL, provider=provider)

    agent = Agent(
        model,
        system_prompt="""
        너는 기업 유튜브 숏츠 전략 컨설턴트다.
        사용자가 제공한 숏츠 영상 분석 결과와 도메인 성공 패턴을 바탕으로,
        실무자가 바로 실행할 수 있는 구체적인 숏츠 제작 가이드라인을 작성한다.

        답변은 한국어로 작성한다.
        분석 결과에 없는 내용을 과장하지 않는다.
        최근 5개 영상 기반의 빠른 진단이라는 한계를 함께 언급한다.
        단순 도메인 일반론이 아니라, 현재 채널 분석 결과와 성공 패턴의 차이를 중심으로 제안한다.
        """
    )

    settings = GoogleModelSettings(
        temperature=0.4,
        max_output_tokens=1800,
    )

    result = agent.run_sync(
        prompt_text,
        model_settings=settings
    )

    return result.output

# ============================================================
# 2. 채널/숏츠 수집 함수
# ============================================================

def resolve_channel_url(channel_input: str) -> tuple[str, str]:
    """
    채널 URL 또는 검색어를 받아 채널 URL 추정.
    안정성은 채널 URL 입력이 가장 높음.
    """
    raw = channel_input.strip()

    if not raw:
        return "", ""

    if is_youtube_url(raw):
        return raw, raw

    # 이름 입력 시 yt-dlp 검색으로 uploader_url 추정
    # 완벽하지 않으므로 UI에서 안내 필요
    query = f"ytsearch5:{raw} 공식 유튜브"
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "ignoreerrors": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)

        entries = info.get("entries", []) if info else []

        for e in entries:
            uploader_url = e.get("uploader_url") or e.get("channel_url")
            uploader = e.get("uploader") or e.get("channel")

            if uploader_url:
                return uploader_url, uploader or raw

    except Exception:
        pass

    return "", raw


def collect_recent_shorts(channel_input: str, limit: int = 5) -> pd.DataFrame:
    """
    채널 URL/채널명에서 최근 shorts 후보 수집.
    yt-dlp 기반.
    """
    channel_url, channel_name_guess = resolve_channel_url(channel_input)

    if not channel_url:
        raise ValueError("채널 URL을 찾지 못했습니다. 채널 URL을 직접 입력해 주세요.")

    shorts_url = to_shorts_tab_url(channel_url)

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "ignoreerrors": True,
        "playlistend": max(limit * 3, 15),
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(shorts_url, download=False)

    entries = info.get("entries", []) if info else []
    channel_title = info.get("channel") or info.get("uploader") or channel_name_guess or "unknown"

    rows = []
    seen = set()

    for e in entries:
        if not e:
            continue

        vid = e.get("id") or extract_video_id(e.get("url", ""))

        if not vid or vid in seen:
            continue

        # shorts 탭에서 가져오기 때문에 shorts URL로 구성
        final_url = f"https://www.youtube.com/shorts/{vid}"

        rows.append({
            "video_id": vid,
            "final_url": final_url,
            "채널명": channel_title,
            "title": e.get("title", ""),
            "channel_url": channel_url,
            "source": "yt-dlp_shorts_tab",
        })

        seen.add(vid)

        if len(rows) >= limit:
            break

    if not rows:
        raise ValueError("최근 숏츠를 찾지 못했습니다. 채널 URL의 /shorts 탭이 접근 가능한지 확인해 주세요.")

    return pd.DataFrame(rows)


# ============================================================
# 3. Agent 실행 함수
# ============================================================

def run_video_agent(input_csv: Path, concurrent: int = 5, delay: int = 3) -> tuple[bool, str, Path]:
    """
    기존 Video-Analysis_agent.py 실행
    """
    PERSONA_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PERSONA_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    csv_stem = input_csv.stem
    output_path = PERSONA_RESULT_DIR / f"result_{csv_stem}.csv"

    cmd = [
        sys.executable,
        str(AGENT_PATH),
        str(input_csv),
        "--concurrent", str(concurrent),
        "--delay", str(delay),
        "--video_dir", str(PERSONA_VIDEO_DIR),
        "--output_dir", str(PERSONA_RESULT_DIR),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    log = (proc.stdout or "") + "\n" + (proc.stderr or "")

    ok = proc.returncode == 0 and output_path.exists()

    return ok, log, output_path


def merge_result_with_input(result_df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """agent 결과에 title/final_url 등 입력 메타데이터 붙이기"""
    if result_df.empty:
        return result_df

    result_df["video_id"] = result_df["video_id"].astype(str).str.strip()
    input_df["video_id"] = input_df["video_id"].astype(str).str.strip()

    meta_cols = [c for c in input_df.columns if c not in result_df.columns and c != "video_id"]

    merged = result_df.merge(
        input_df[["video_id"] + meta_cols],
        on="video_id",
        how="left"
    )

    return merged


# ============================================================
# 4. UI
# ============================================================

st.markdown("""
<div class="yt-highlight">
    <h1 style="margin-bottom: 4px;">▶ 페르소나 기업 숏츠 진단</h1>
    <p style="margin-bottom: 0;">
        최근 숏츠 5개를 분석해, 성공 숏츠 패턴과 비교하고 맞춤형 제작 가이드라인을 제안합니다.
    </p>
</div>
""", unsafe_allow_html=True)

baseline_df = load_baseline()

if baseline_df.empty:
    st.error(f"기존 200개 분석 기준 파일을 찾지 못했습니다: {BASELINE_PATH}")
    st.stop()

with st.sidebar:
    st.header("분석 설정")

    channel_input = st.text_input(
        "유튜브 채널명 또는 채널 URL",
        placeholder="예: https://www.youtube.com/@channel/shorts"
    )

    domain = st.selectbox("기업 도메인", ["FnB", "IT"])

    limit = st.number_input(
        "분석할 최근 숏츠 수",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="3분 이내 빠른 진단을 위해 기본값은 5개를 추천합니다."
    )

    concurrent = st.slider("동시 실행 수", min_value=1, max_value=10, value=5)
    delay = st.slider("API 대기 시간(초)", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.info("채널명 입력도 가능하지만, 안정적으로는 채널 URL 입력을 권장합니다.")

    run_button = st.button("🚀 최근 숏츠 수집 + 영상 분석 실행", type="primary")


# ============================================================
# 5. 분석 실행
# ============================================================

if run_button:
    if not channel_input.strip():
        st.warning("채널명 또는 채널 URL을 입력해 주세요.")
        st.stop()

    safe_name = safe_filename(channel_input)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_stem = f"persona_{safe_name}_{domain}_shorts_{limit}_{timestamp}"

    input_csv_path = PERSONA_DATA_DIR / f"{csv_stem}.csv"

    try:
        with st.status("최근 숏츠 수집 중...", expanded=True) as status:
            shorts_df = collect_recent_shorts(channel_input, limit=int(limit))
            shorts_df.to_csv(input_csv_path, index=False, encoding="utf-8-sig")

            st.write(f"수집 완료: {len(shorts_df)}개")
            st.dataframe(shorts_df[["video_id", "title", "final_url", "채널명"]], use_container_width=True)

            status.update(label="영상 분석 agent 실행 중...", state="running")

            start = time.time()
            ok, log, result_path = run_video_agent(
                input_csv=input_csv_path,
                concurrent=int(concurrent),
                delay=int(delay),
            )
            elapsed = time.time() - start

            if not ok:
                status.update(label="영상 분석 실패", state="error")
                st.error("영상 분석 agent 실행에 실패했습니다.")
                st.code(log[-4000:])
                st.stop()

            status.update(label=f"영상 분석 완료 | 소요 시간: {elapsed/60:.1f}분", state="complete")

        # 결과 로드
        result_df = pd.read_csv(result_path, encoding="utf-8-sig")
        result_df = merge_result_with_input(result_df, shorts_df)

        merged_result_path = PERSONA_RESULT_DIR / f"merged_{result_path.name}"
        result_df.to_csv(merged_result_path, index=False, encoding="utf-8-sig")

        st.session_state["persona_result_df"] = result_df
        st.session_state["persona_input_df"] = shorts_df
        st.session_state["persona_result_path"] = str(merged_result_path)
        st.session_state["persona_domain"] = domain
        st.session_state["channel_input"] = channel_input

    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()


# ============================================================
# 6. 기존 결과 파일 불러오기 옵션
# ============================================================

st.markdown("## 0. 기존 분석 결과 불러오기")

existing_files = sorted(PERSONA_RESULT_DIR.glob("merged_result_persona_*.csv"), reverse=True)

if existing_files:
    selected_existing = st.selectbox(
        "이미 생성된 페르소나 분석 결과 파일",
        ["선택 안 함"] + [str(p) for p in existing_files]
    )

    if selected_existing != "선택 안 함":
        loaded_df = pd.read_csv(selected_existing, encoding="utf-8-sig")
        st.session_state["persona_result_df"] = loaded_df
        st.session_state["persona_result_path"] = selected_existing
        st.session_state["persona_domain"] = domain
        st.session_state["channel_input"] = "기존 결과 불러오기"


# ============================================================
# 7. 결과 표시
# ============================================================

if "persona_result_df" not in st.session_state:
    st.markdown("## 사용 방법")
    st.write(
        """
        1. 왼쪽 사이드바에 유튜브 채널 URL 또는 채널명을 입력합니다.  
        2. 기업 도메인을 선택합니다.  
        3. 최근 숏츠 5개를 수집하고 영상 분석 agent를 실행합니다.  
        4. 분석 결과를 기존 200개 성공 숏츠 패턴과 비교해 맞춤형 가이드라인을 확인합니다.
        """
    )
    st.stop()


result_df = st.session_state["persona_result_df"].copy()
domain = st.session_state.get("persona_domain", domain)
channel_input = st.session_state.get("channel_input", "")

persona_summary = summarize_persona_result(result_df)
baseline_summary = get_domain_success_baseline(baseline_df, domain)
compare_df = build_compare_table(domain, persona_summary, baseline_summary)
tasks = build_improvement_tasks(domain, compare_df)
guideline = build_rule_based_guideline(domain, persona_summary, compare_df, tasks)


# ============================================================
# Section 1. 분석 개요
# ============================================================

st.markdown("## 1. 분석 개요")

col1, col2, col3, col4 = st.columns(4)

col1.metric("분석 숏츠 수", f"{persona_summary.get('n', 0)}개")
col2.metric("주요 첫 3초", persona_summary.get("main_first_3sec", "-"))
col3.metric("주요 모션그래픽", persona_summary.get("main_motion_graphic", "-"))
col4.metric("주요 포맷", persona_summary.get("main_video_format", "-"))

col5, col6, col7 = st.columns(3)
col5.metric("평균 person_ratio", persona_summary.get("person_ratio", np.nan))
col6.metric("평균 face_ratio", persona_summary.get("face_ratio", np.nan))
col7.metric("평균 text_ratio", persona_summary.get("text_ratio", np.nan))

with st.expander("분석된 영상 목록 보기", expanded=False):
    display_cols = [
        "video_id", "title", "final_url", "채널명",
        "first_3sec", "motion_graphic", "video_format",
        "person_ratio", "face_ratio", "text_ratio", "reason"
    ]
    display_cols = [c for c in display_cols if c in result_df.columns]
    st.dataframe(result_df[display_cols], width="stretch")

st.markdown("### 분석된 숏츠 카드")
for _, row in result_df.head(5).iterrows():
    render_video_card(row)


# ============================================================
# Section 2. 현재 채널 숏츠 특징
# ============================================================

st.markdown("## 2. 현재 채널 숏츠 특징 요약")
st.caption("최근 숏츠 5개 기준 빠른 진단 결과입니다.")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if "first_3sec" in result_df.columns:
        first_df = result_df["first_3sec"].value_counts().reset_index()
        first_df.columns = ["first_3sec", "count"]
        st.altair_chart(
            make_bar_chart(first_df, "first_3sec:N", "count:Q", "first_3sec 분포"),
            use_container_width=True
        )

with chart_col2:
    if "motion_graphic" in result_df.columns:
        motion_df = result_df["motion_graphic"].value_counts().reset_index()
        motion_df.columns = ["motion_graphic", "count"]
        st.altair_chart(
            make_bar_chart(motion_df, "motion_graphic:N", "count:Q", "motion_graphic 분포"),
            use_container_width=True
        )

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    if "video_format" in result_df.columns:
        fmt_df = result_df["video_format"].value_counts().reset_index()
        fmt_df.columns = ["video_format", "count"]
        st.altair_chart(
            make_bar_chart(fmt_df, "video_format:N", "count:Q", "video_format 분포"),
            use_container_width=True
        )

with chart_col4:
    ratio_cols = [c for c in ["person_ratio", "face_ratio", "text_ratio"] if c in result_df.columns]
    if ratio_cols:
        ratio_df = pd.DataFrame({
            "feature": ratio_cols,
            "mean": [result_df[c].mean() for c in ratio_cols]
        })
        st.altair_chart(
            make_bar_chart(ratio_df, "feature:N", "mean:Q", "시각 비율 평균"),
            use_container_width=True
        )


# ============================================================
# Section 3. 성공 패턴과 비교
# ============================================================

st.markdown("## 3. 도메인 성공 숏츠 패턴과 비교")

if not baseline_summary:
    st.warning("도메인 성공 패턴 기준값을 만들 수 없습니다.")
else:
    st.write(
        f"""
        선택한 도메인 **{domain}**의 기존 성공 숏츠 패턴과 현재 채널의 최근 숏츠 분석 결과를 비교합니다.
        """
    )

    st.dataframe(compare_df, use_container_width=True)

    st.markdown("### 비교 해석")

    부족 = compare_df[compare_df["진단"].isin(["부족", "과다"])]
    if 부족.empty:
        st.success("현재 채널은 핵심 항목에서 도메인 성공 패턴과 크게 벗어나지 않습니다.")
    else:
        for _, row in 부족.iterrows():
            if row["진단"] == "부족":
                st.warning(
                    f"**{row['항목']}**이 도메인 성공 패턴보다 낮습니다. "
                    f"현재 {row['현재 채널']} / 성공 패턴 {row['도메인 성공 패턴']}"
                )
            elif row["진단"] == "과다":
                st.warning(
                    f"**{row['항목']}**이 도메인 성공 패턴보다 높습니다. "
                    f"현재 {row['현재 채널']} / 성공 패턴 {row['도메인 성공 패턴']}"
                )


# ============================================================
# Section 4. 우선 개선 과제
# ============================================================

st.markdown("## 4. 우선 개선 과제 Top 3")

for i, task in enumerate(tasks, start=1):
    with st.container(border=True):
        st.markdown(f"### {i}. {task['title']}")
        st.write(task["desc"])
        st.info(task["action"])


# ============================================================
# Section 5. 맞춤형 숏츠 영상 제작 가이드라인
# ============================================================

st.markdown("## 5. 맞춤형 숏츠 영상 제작 가이드라인")

with st.container(border=True):
    st.markdown(f"### 핵심 전략")
    st.write(guideline["핵심 전략"])

    st.markdown("### 첫 3초 전략")
    st.write(guideline["첫 3초 전략"])

    st.markdown("### 추천 영상 포맷")
    st.write(guideline["추천 포맷"])

    st.markdown("### 모션그래픽 활용")
    st.write(guideline["모션그래픽 활용"])

    st.markdown("### 피해야 할 구성")
    st.write(guideline["피해야 할 구성"])

st.markdown("### 제작 체크리스트")

for item in guideline["체크리스트"]:
    st.checkbox(item, value=False)

st.markdown("### 다음 숏츠 기획안 예시")

for idea in guideline["다음 숏츠 기획안"]:
    st.write(f"- {idea}")


# ============================================================
# Section 6. AI 가이드라인 생성용 프롬프트 출력
# ============================================================

prompt_text = f"""
너는 기업 유튜브 숏츠 전략 컨설턴트다.

아래 페르소나 기업의 최근 숏츠 5개 빠른 진단 결과와 도메인 성공 패턴 비교 결과를 바탕으로,
실행 가능한 숏츠 영상 제작 가이드라인을 작성하라.

[기업 정보]
- 입력 채널: {channel_input}
- 도메인: {domain}
- 분석 영상 수: {persona_summary.get("n", 0)}개

[현재 채널 요약]
- 주요 first_3sec: {persona_summary.get("main_first_3sec", "-")}
- 주요 motion_graphic: {persona_summary.get("main_motion_graphic", "-")}
- 주요 video_format: {persona_summary.get("main_video_format", "-")}
- 평균 person_ratio: {persona_summary.get("person_ratio", "-")}
- 평균 face_ratio: {persona_summary.get("face_ratio", "-")}
- 평균 text_ratio: {persona_summary.get("text_ratio", "-")}

[도메인 성공 패턴과 비교]
{compare_df.to_markdown(index=False)}

[우선 개선 과제]
{chr(10).join([f"{i+1}. {t['title']} - {t['action']}" for i, t in enumerate(tasks)])}

출력 형식:
1. 핵심 진단
2. 현재 채널의 강점
3. 보완할 점
4. 첫 3초 전략
5. 추천 영상 포맷
6. 모션그래픽/자막 활용 방식
7. 다음 숏츠 기획안 3개
8. 제작 체크리스트

주의:
- 최근 5개 분석 기반의 빠른 진단이라는 점을 명시할 것
- 분석 결과에 없는 내용을 과장하지 말 것
- 도메인 성공 패턴과 현재 채널의 차이를 중심으로 제안할 것
- 실무자가 바로 사용할 수 있게 구체적으로 작성할 것
"""

st.markdown("## 6. AI 맞춤형 가이드라인 생성")
st.caption(
    "최근 숏츠 5개 분석 결과와 도메인 성공 패턴 비교 결과를 바탕으로 "
    "AI가 맞춤형 숏츠 제작 가이드라인을 생성합니다."
)

with st.expander("LLM Agent에 넘길 프롬프트 보기", expanded=False):
    st.text_area("프롬프트", prompt_text, height=500)

if "ai_guideline_result" not in st.session_state:
    st.session_state["ai_guideline_result"] = ""

if st.button("🤖 AI 맞춤형 가이드라인 생성", type="primary"):
    with st.spinner("AI가 맞춤형 숏츠 가이드라인을 생성하는 중입니다..."):
        try:
            ai_result = generate_ai_guideline(prompt_text)
            st.session_state["ai_guideline_result"] = ai_result
        except Exception as e:
            st.error(f"AI 가이드라인 생성 중 오류가 발생했습니다: {e}")

if st.session_state["ai_guideline_result"]:
    st.markdown("### AI 생성 맞춤형 가이드라인")
    st.markdown(st.session_state["ai_guideline_result"])

# ============================================================
# Section 7. 저장 경로
# ============================================================

st.markdown("## 7. 저장 결과")

st.write("분석 결과 파일:")
st.code(st.session_state.get("persona_result_path", "-"))

report_path = PERSONA_REPORT_DIR / f"guideline_{safe_filename(channel_input)}_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

ai_guideline_text = st.session_state.get("ai_guideline_result", "")

if st.button("📄 현재 가이드라인 리포트 저장"):
    report_md = f"""
# 페르소나 기업 맞춤형 숏츠 영상 진단 리포트

## 1. 분석 개요
- 입력 채널: {channel_input}
- 도메인: {domain}
- 분석 영상 수: {persona_summary.get("n", 0)}개

## 2. 현재 채널 요약
- 주요 first_3sec: {persona_summary.get("main_first_3sec", "-")}
- 주요 motion_graphic: {persona_summary.get("main_motion_graphic", "-")}
- 주요 video_format: {persona_summary.get("main_video_format", "-")}
- 평균 person_ratio: {persona_summary.get("person_ratio", "-")}
- 평균 face_ratio: {persona_summary.get("face_ratio", "-")}
- 평균 text_ratio: {persona_summary.get("text_ratio", "-")}

## 3. 도메인 성공 패턴과 비교

{compare_df.to_markdown(index=False)}

## 4. 우선 개선 과제

{chr(10).join([f"### {i+1}. {t['title']}\\n- {t['desc']}\\n- 실행안: {t['action']}\\n" for i, t in enumerate(tasks)])}

## 5. 맞춤형 가이드라인

### 핵심 전략
{guideline["핵심 전략"]}

### 첫 3초 전략
{guideline["첫 3초 전략"]}

### 추천 영상 포맷
{guideline["추천 포맷"]}

### 모션그래픽 활용
{guideline["모션그래픽 활용"]}

### 피해야 할 구성
{guideline["피해야 할 구성"]}

### 제작 체크리스트
{chr(10).join([f"- {x}" for x in guideline["체크리스트"]])}

### 다음 숏츠 기획안
{chr(10).join([f"- {x}" for x in guideline["다음 숏츠 기획안"]])}

## 6. AI 생성 맞춤형 가이드라인
{ai_guideline_text if ai_guideline_text else "AI 가이드라인은 아직 생성되지 않았습니다."}

"""

    report_path.write_text(report_md, encoding="utf-8")
    st.success(f"리포트 저장 완료: {report_path}")