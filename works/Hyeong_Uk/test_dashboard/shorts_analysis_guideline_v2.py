# works/Hyeong_Uk/streamlit_pages/persona_shorts_guideline_app.py

import os
import re
import time
import subprocess
import sys
import requests
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
from yt_dlp import YoutubeDL

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider


# ============================================================
# 0. 기본 설정
# ============================================================

st.set_page_config(
    page_title="ShortStrategy",
    page_icon="▶",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

/* ── 기본 ── */
html,body,[class*="css"]{font-family:'Roboto','Noto Sans KR',sans-serif;background:#f8f9fa!important;color:#111827!important}
.stApp{background:#f8f9fa}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:.5rem 2rem 2rem 2rem!important;max-width:100%}

/* ── 입력 ── */
.stTextInput>div>div>input,.stTextArea textarea,.stSelectbox>div>div{background:#ffffff!important;border:1px solid #d1d5db!important;color:#111827!important;border-radius:6px!important}
.stTextInput>div>div>input:focus,.stTextArea textarea:focus{border-color:#2563eb!important;box-shadow:none!important}
.stTextInput>div>div>input::placeholder,.stTextArea textarea::placeholder{color:#9ca3af!important}
label{color:#374151!important;font-size:12px!important}
.stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p{color:#111827!important}

/* ── 버튼 ── */
.stButton>button{background:#ffffff!important;color:#111827!important;border:1px solid #d1d5db!important;border-radius:50px!important;font-size:13px!important;font-weight:500!important;padding:6px 16px!important;transition:all .15s!important}
.stButton>button:hover{background:#f3f4f6!important;border-color:#9ca3af!important}
.stDownloadButton>button{background:#cc0000!important;color:#fff!important;border:none!important;border-radius:50px!important;font-size:12px!important;font-weight:600!important}

/* ── Expander ── */
.stExpander{background:#ffffff!important;border:1px solid #e5e7eb!important;border-radius:10px!important;margin-bottom:10px!important}
.stExpander summary{color:#111827!important;font-weight:600!important;font-size:13px!important;background:#ffffff!important}
.stExpander summary:hover{color:#000!important;background:#f9fafb!important}
.stExpander details{background:#ffffff!important}
.stExpander [data-testid="stExpanderDetails"]{border-top:1px solid #e5e7eb!important;padding-top:12px!important;background:#ffffff!important}
details[open]>summary,[data-testid="stExpander"] details[open] summary{background:#ffffff!important;color:#111827!important}

/* ── 스크롤바 ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f3f4f6}
::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:3px}

/* ── 카드 ── */
.yt-card{background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin-bottom:12px}

/* ── 하이라이트 헤더 (기존 yt-highlight 대체) ── */
.yt-highlight{background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;margin-bottom:16px}
.yt-highlight h1{color:#111827!important;font-size:18px!important;margin-bottom:4px!important}
.yt-highlight p{color:#6b7280!important;font-size:13px!important;margin-bottom:0!important}

/* ── muted 텍스트 ── */
.yt-muted{color:#6b7280;font-size:12px}

/* ── 섹션 타이틀 ── */
.sec-title{font-size:14px;font-weight:600;color:#111827;display:flex;align-items:center;gap:8px;margin-bottom:10px}
.tbar{width:3px;height:16px;border-radius:2px;display:inline-block;flex-shrink:0}

/* ── 배지 ── */
.badge{display:inline-block;padding:2px 9px;border-radius:50px;font-size:11px;font-weight:600;margin:2px}
.badge-red{background:rgba(220,38,38,.08);color:#dc2626;border:1px solid rgba(220,38,38,.25)}
.badge-blue{background:rgba(37,99,235,.08);color:#2563eb;border:1px solid rgba(37,99,235,.25)}
.badge-gray{background:rgba(107,114,128,.08);color:#6b7280;border:1px solid rgba(107,114,128,.2)}

/* ── 안내 박스 ── */
.api-notice{background:rgba(161,98,7,.06);border:1px solid rgba(161,98,7,.25);border-radius:8px;padding:9px 13px;font-size:11px;color:#92400e;line-height:1.7;margin-bottom:10px}
.guide-hint{background:rgba(37,99,235,.06);border:1px solid rgba(37,99,235,.25);border-radius:8px;padding:9px 13px;font-size:11px;color:#1e40af;line-height:1.7;margin-bottom:10px}

/* ── Metric ── */
[data-testid="stMetric"]{background:#ffffff!important;padding:12px 14px!important;border-radius:10px!important;border:1px solid #e5e7eb!important}
[data-testid="stMetricLabel"]{color:#6b7280!important;font-size:11px!important}
[data-testid="stMetricValue"]{color:#111827!important;font-size:18px!important;font-weight:700!important}

/* ── 구분선 ── */
hr{border-color:#e5e7eb!important}
</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent

AGENT_PATH = BASE_DIR / "Video-Analysis_agent.py"
BASELINE_PATH = BASE_DIR / "result_sample_shorts_all_for_video_agent_fixed.csv"

GOOGLE_CLOUD_PROJECT = st.secrets.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_REGION = st.secrets.get("GOOGLE_CLOUD_REGION", "")
GEMINI_MODEL = st.secrets.get(
    "GEMINI_PROMPT_MODEL",
    st.secrets.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
)
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")

PERSONA_BASE_DIR = BASE_DIR / "persona_shorts_guideline"
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
            ("first_3sec_인물", "영상 오프닝(3초 이내) 인물 비율", True),
            ("motion_graphic_보조적", "모션그래픽 보조 활용 비율", True),
            ("motion_graphic_핵심요소", "모션그래픽 핵심요소 비율", False),
        ]
    else:
        compare_specs = [
            ("text_ratio", "텍스트/자막 비율", True),
            ("first_3sec_텍스트", "영상 오프닝(3초 이내) 텍스트 비율", True),
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
            elif item == "영상 오프닝(3초 이내) 인물 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "영상 오프닝(3초 이내) 인물 후킹 강화",
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

            if item == "영상 오프닝(3초 이내) 텍스트 비율" and diagnosis == "부족":
                tasks.append({
                    "title": "영상 오프닝(3초 이내) 텍스트 후킹 강화",
                    "desc": "IT 성공 숏츠는 초반에 핵심 메시지를 텍스트로 제시하는 경향이 있습니다.",
                    "action": "문제 상황, 기능명, 혜택을 영상 오프닝(3초 이내)에 짧은 자막으로 제시하세요."
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
            "영상 오프닝(3초 이내) 전략": "제품 설명 텍스트보다 사람이 제품을 먹거나 사용하는 장면으로 시작하세요.",
            "추천 포맷": "제품리뷰, 웹예능형 반응 콘텐츠, 짧은 상황극, 제품 경험형 콘텐츠",
            "모션그래픽 활용": "모션그래픽은 핵심 연출보다 제품명, 반응 포인트, CTA를 강조하는 보조 장치로 활용하세요.",
            "피해야 할 구성": "텍스트 설명만으로 시작하거나, 그래픽이 제품 경험 장면을 가리는 구성은 지양하세요.",
            "체크리스트": [
                "영상 오프닝(3초 이내)에 인물 또는 제품 경험 장면이 등장하는가?",
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
            "영상 오프닝(3초 이내) 전략": "문제 상황, 기능명, 혜택을 짧은 텍스트로 먼저 제시하세요.",
            "추천 포맷": "기술설명형, 기능 시연형, 문제 해결형, 짧은 튜토리얼형 콘텐츠",
            "모션그래픽 활용": "서비스 구조, 작동 방식, 기능 흐름을 아이콘, 자막, 화면 전환으로 시각화하세요.",
            "피해야 할 구성": "핵심 메시지가 늦게 나오거나, 긴 인터뷰형 설명으로 시작하는 구성은 지양하세요.",
            "체크리스트": [
                "영상 오프닝(3초 이내)에 핵심 메시지나 문제 상황이 텍스트로 제시되는가?",
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


def generate_ai_guideline(prompt_text: str) -> str:
    """
    Gemini / Vertex AI를 사용해 맞춤형 숏츠 가이드라인 생성
    """
    if not GOOGLE_CLOUD_PROJECT or not GOOGLE_CLOUD_REGION or not GEMINI_MODEL:
        raise ValueError(
            "Google Cloud 환경변수가 설정되지 않았습니다. "
            "GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION, GEMINI_PROMPT_MODEL을 확인하세요."
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


def collect_recent_shorts(shorts_url: str, channel_name: str, limit: int = 5) -> pd.DataFrame:
    """
    채널 숏츠 탭 URL에서 최근 숏츠 수집.
    yt_channel_info 에서 반환된 shorts_url을 바로 사용 — resolve 단계 없음.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "ignoreerrors": True,
        "playlistend": max(limit * 3, 15),
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(shorts_url, download=False)

    entries       = info.get("entries", []) if info else []
    channel_title = info.get("channel") or info.get("uploader") or channel_name or "unknown"

    rows = []
    seen = set()

    for e in entries:
        if not e:
            continue
        vid = e.get("id") or extract_video_id(e.get("url", ""))
        if not vid or vid in seen:
            continue

        rows.append({
            "video_id":  vid,
            "final_url": f"https://www.youtube.com/shorts/{vid}",
            "채널명":    channel_title,
            "title":     e.get("title", ""),
            "source":    "yt-dlp_shorts_tab",
        })
        seen.add(vid)

        if len(rows) >= limit:
            break

    if not rows:
        raise ValueError("최근 숏츠를 찾지 못했습니다. 채널의 /shorts 탭이 공개되어 있는지 확인해 주세요.")

    return pd.DataFrame(rows)


# ============================================================
# 3. Agent 실행 함수
# ============================================================

def run_video_agent(
    input_csv: Path,
    concurrent: int = 10,
    delay: int = 1,
    log_placeholder=None,
    elapsed_placeholder=None,
) -> tuple[bool, str, Path]:
    """
    Video-Analysis_agent.py를 Popen으로 실행해 로그를 실시간 스트리밍.
    log_placeholder  : st.empty() — 로그 출력용
    elapsed_placeholder : st.empty() — 경과 시간 표시용
    """
    PERSONA_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PERSONA_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    csv_stem    = input_csv.stem
    output_path = PERSONA_RESULT_DIR / f"result_{csv_stem}.csv"

    cmd = [
        sys.executable,
        str(AGENT_PATH),
        str(input_csv),
        "--concurrent", str(concurrent),
        "--delay",      str(delay),
        "--video_dir",  str(PERSONA_VIDEO_DIR),
        "--output_dir", str(PERSONA_RESULT_DIR),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]        = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # stderr → stdout 합치기
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,                  # 줄 단위 버퍼
    )

    log_lines: list[str] = []
    start = time.time()

    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        log_lines.append(line)

        # 최근 12줄만 표시
        visible = log_lines[-12:]
        if log_placeholder is not None:
            log_placeholder.code("\n".join(visible), language=None)

        elapsed = time.time() - start
        if elapsed_placeholder is not None:
            elapsed_placeholder.markdown(
                f'<div style="font-size:11px;color:#6b7280">⏱ 경과 {elapsed:.0f}초</div>',
                unsafe_allow_html=True,
            )

    proc.wait()
    log = "\n".join(log_lines)
    ok  = proc.returncode == 0 and output_path.exists()
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
# 3-b. YouTube Data API 채널 후보 검색 헬퍼
# ============================================================

def fmt_num(n: int) -> str:
    n = int(n)
    if n >= 100_000_000: return f"{n/100_000_000:.1f}억"
    if n >= 10_000_000:  return f"{n/10_000_000:.1f}천만"
    if n >= 10_000:      return f"{n/10_000:.1f}만"
    if n >= 1_000:       return f"{n/1_000:.1f}천"
    return str(n)


def yt_channel_info(cid: str, key: str) -> dict:
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "snippet,statistics", "id": cid, "key": key},
        timeout=10,
    )
    d = r.json()
    if "error" in d:
        raise Exception(d["error"]["message"])
    if not d.get("items"):
        raise Exception("채널 정보를 찾을 수 없습니다")
    ch = d["items"][0]
    handle = ch["snippet"].get("customUrl", "")  # @handle 형태
    shorts_url = (
        f"https://www.youtube.com/{handle}/shorts"
        if handle
        else f"https://www.youtube.com/channel/{cid}/shorts"
    )
    return {
        "id":          cid,
        "name":        ch["snippet"]["title"],
        "description": ch["snippet"].get("description", "")[:150],
        "subscribers": int(ch["statistics"].get("subscriberCount", 0)),
        "video_count": int(ch["statistics"].get("videoCount", 0)),
        "handle":      handle,
        "shorts_url":  shorts_url,
    }


def yt_search_channel_candidates(q: str, key: str, limit: int = 8) -> list:
    query = q.strip()
    if query.startswith("UC"):
        return [yt_channel_info(query, key)]
    if query.startswith("@"):
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "snippet,statistics", "forHandle": query, "key": key},
            timeout=10,
        )
        d = r.json()
        if "error" in d: raise Exception(d["error"]["message"])
        if d.get("items"): return [yt_channel_info(d["items"][0]["id"], key)]
    seen, channel_ids = set(), []
    for sq in [query, f"{query} 공식", f"{query} official", f"{query} 유튜브"]:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "type": "channel", "q": sq,
                    "maxResults": 5, "key": key},
            timeout=10,
        )
        d = r.json()
        if "error" in d: raise Exception(d["error"]["message"])
        for item in d.get("items", []):
            cid = item["snippet"]["channelId"]
            if cid not in seen:
                seen.add(cid)
                channel_ids.append(cid)
            if len(channel_ids) >= limit: break
        if len(channel_ids) >= limit: break
    if not channel_ids:
        raise Exception("채널을 찾을 수 없습니다")
    return [yt_channel_info(cid, key) for cid in channel_ids]


# ============================================================
# 4. UI
# ============================================================



baseline_df = load_baseline()
if baseline_df.empty:
    st.error(f"기존 200개 분석 기준 파일을 찾지 못했습니다: {BASELINE_PATH}")
    st.stop()

# ── 도메인 + 채널 검색 카드 ──────────────────────────────────

guide_domain = st.radio(
    "업종 선택",
    ["🍔 FnB", "💻 IT"],
    horizontal=True,
    key="guide_domain_sel",
    label_visibility="collapsed",
)
domain_now = "FnB" if guide_domain.startswith("🍔") else "IT"
accent_color = "#ff0000" if domain_now == "FnB" else "#2563eb"

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown(
    f'<div class="sec-title">' +
    f'<span class="tbar" style="background:{accent_color}"></span>' +
    f'{domain_now} 공식 채널 검색</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns([4, 1])
with c1:
    search_q = st.text_input(
        "채널명",
        placeholder="예: CU씨유튜브  /  @CUtube  /  UCxxxxxxxx",
        key="search_q",
        label_visibility="collapsed",
    )
with c2:
    search_btn = st.button("🔍 채널 검색", key="search_btn", use_container_width=True)

if search_btn:
    if not search_q.strip():
        st.warning("채널명을 입력해주세요.")
    elif not YOUTUBE_API_KEY:
        st.error("YOUTUBE_API_KEY 가 설정되지 않았습니다.")
    else:
        with st.spinner("공식 채널 후보를 찾는 중..."):
            try:
                st.session_state["channel_candidates"] = yt_search_channel_candidates(
                    search_q, YOUTUBE_API_KEY
                )
                st.session_state["selected_channel_info"] = None
            except Exception as e:
                st.error(f"채널 검색 오류: {e}")

# 후보 선택
candidates = st.session_state.get("channel_candidates", [])
selected_channel = None
if candidates:
    def _label(i):
        ch = candidates[i]
        return f"{ch['name']} · 구독자 {fmt_num(ch['subscribers'])} · 영상 {fmt_num(ch['video_count'])}개"

    idx = st.selectbox(
        "공식 채널을 선택하세요",
        range(len(candidates)),
        format_func=_label,
        key="candidate_idx",
    )
    selected_channel = candidates[idx]
    st.markdown(
        f'<div class="guide-hint" style="margin-top:6px">' +
        f'선택됨: <b>{selected_channel["name"]}</b><br>' +
        f'<span style="color:#6b7280">{selected_channel["description"][:120]}</span>' +
        f'</div>',
        unsafe_allow_html=True,
    )


# ── 분석 설정 + 실행 ──────────────────────────────────────────
if selected_channel:
    with st.expander("⚙️ 분석 설정", expanded=True):
        limit = st.number_input(
            "분석할 숏츠 개수",
            min_value=1, max_value=10, value=5, step=1,
            help="빠른 진단을 위해 기본값 5개를 추천합니다. (최대 10개 분석 가능)",
        )

    st.markdown(
        f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:10px 14px;font-size:12px;color:#6b7280;line-height:1.8;margin-bottom:10px">'
        f'숏츠 <b style="color:#111827">{int(limit)}개</b>를 수집한 뒤, 각 영상을 프레임 단위로 분석합니다.<br>'
        f'영상 1개당 약 <b style="color:#111827">1~2분</b>이 소요되며, '
        f'{int(limit)}개 기준 최대 <b style="color:#111827">{int(limit)*2}분</b> 정도 걸릴 수 있습니다.<br>'
        f'분석이 진행되는 동안 이 페이지를 닫지 마세요.'
        f'</div>',
        unsafe_allow_html=True,
    )

    run_button = st.button(
        f"'{selected_channel['name']}' 숏츠 분석 실행",
        key="run_btn",
        use_container_width=True,
    )

    if run_button:
        safe_name   = safe_filename(selected_channel["name"])
        domain_run  = domain_now
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_stem    = f"persona_{safe_name}_{domain_run}_shorts_{int(limit)}_{timestamp}"
        input_csv_path = PERSONA_DATA_DIR / f"{csv_stem}.csv"

        try:
            with st.status("숏츠 수집 및 영상 분석 중...", expanded=True) as status:

                # 1단계: 숏츠 URL 수집
                status.update(label="1/2 최근 숏츠 수집 중...")
                shorts_df = collect_recent_shorts(selected_channel["shorts_url"], selected_channel["name"], limit=int(limit))
                shorts_df.to_csv(input_csv_path, index=False, encoding="utf-8-sig")
                st.write(f"✅ 수집 완료: {len(shorts_df)}개 숏츠")
                st.dataframe(
                    shorts_df[["video_id", "title", "final_url", "채널명"]],
                    use_container_width=True,
                )

                # 2단계: agent 실시간 실행
                status.update(label="2/2  영상 분석 agent 실행 중...")
                log_box     = st.empty()
                elapsed_box = st.empty()

                start = time.time()
                ok, log, result_path = run_video_agent(
                    input_csv=input_csv_path,
                    concurrent=8,
                    delay=3,
                    log_placeholder=log_box,
                    elapsed_placeholder=elapsed_box,
                )
                elapsed = time.time() - start
                elapsed_box.empty()

                if not ok:
                    status.update(label="❌ 영상 분석 실패", state="error")
                    st.error("영상 분석 agent 실행에 실패했습니다.")
                    with st.expander("에러 로그 보기"):
                        st.code(log[-3000:], language=None)
                    st.stop()

                status.update(
                    label=f"✅ 분석 완료 — 소요 시간 {elapsed/60:.1f}분",
                    state="complete",
                )

            result_df_run = pd.read_csv(result_path, encoding="utf-8-sig")
            result_df_run = merge_result_with_input(result_df_run, shorts_df)
            merged_result_path = PERSONA_RESULT_DIR / f"merged_{result_path.name}"
            result_df_run.to_csv(merged_result_path, index=False, encoding="utf-8-sig")

            st.session_state["persona_result_df"]   = result_df_run
            st.session_state["persona_input_df"]    = shorts_df
            st.session_state["persona_result_path"] = str(merged_result_path)
            st.session_state["persona_domain"]      = domain_run
            st.session_state["channel_input"]       = selected_channel["name"]
            st.rerun()

        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.stop()



st.markdown("<hr style='border-color:#e5e7eb;margin:20px 0'>", unsafe_allow_html=True)

# ── 결과 없으면 안내 후 종료 ──────────────────────────────────
if "persona_result_df" not in st.session_state:
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#6b7280">' +
        '<div style="font-size:40px;margin-bottom:10px">📊</div>' +
        '<div style="font-size:14px;font-weight:600;color:#111827;margin-bottom:8px">' +
        '채널을 검색하고 분석을 실행하세요</div>' +
        '<div style="font-size:12px;color:#9ca3af;line-height:2">' +
        '1. 업종(FnB / IT)을 선택합니다<br>' +
        '2. 채널명을 검색해 공식 채널을 선택합니다<br>' +
        '3. 분석을 실행하면 최근 숏츠를 수집하고 성공 패턴과 비교합니다' +
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()



result_df     = st.session_state["persona_result_df"].copy()
domain        = st.session_state.get("persona_domain", domain_now)
channel_input = st.session_state.get("channel_input", "")

persona_summary  = summarize_persona_result(result_df)
baseline_summary = get_domain_success_baseline(baseline_df, domain)
compare_df       = build_compare_table(domain, persona_summary, baseline_summary)
tasks            = build_improvement_tasks(domain, compare_df)
guideline        = build_rule_based_guideline(domain, persona_summary, compare_df, tasks)
accent_color     = "#ff0000" if domain == "FnB" else "#2563eb"

# 표시용 컬럼명·값 변환 (내부 용어 → 직관적 한국어)
DIAG_LABEL = {"부족": "개선 필요", "과다": "과다", "유사": "적정", "높음": "우수", "낮음": "양호", "판단 불가": "데이터 부족"}
display_df = compare_df.rename(columns={
    "항목": "비교 항목", "현재 채널": "현재", "도메인 성공 패턴": "성공 기준", "진단": "평가"
}).copy()
display_df["평가"] = display_df["평가"].map(lambda v: DIAG_LABEL.get(v, v))

def _fmt(v):
    """비율값(0~1)을 백분율 문자열로 변환 (예: 0.78 → 78%)"""
    return f"{v*100:.0f}%" if v is not None and str(v) not in ("nan", "None") else "-"

# ──────────────────────────────────────────────────────────────
# 채널 숏츠 분석 현황
# ──────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-title"><span class="tbar" style="background:{accent_color}"></span>'
    f'{channel_input} 숏츠 분석 현황 ({domain})</div>',
    unsafe_allow_html=True,
)
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("분석한 숏츠",   f"{persona_summary.get('n', 0)}개")
m2.metric("영상 오프닝(3초 이내) 전략",   persona_summary.get("main_first_3sec",    "-"))
m3.metric("모션그래픽",    persona_summary.get("main_motion_graphic", "-"))
m4.metric("주요 포맷",     persona_summary.get("main_video_format",   "-"))
m5.metric("인물 비중",     _fmt(persona_summary.get("person_ratio")))
m6.metric("얼굴 비중",     _fmt(persona_summary.get("face_ratio")))
m7.metric("자막 비중",     _fmt(persona_summary.get("text_ratio")))

st.markdown("<hr style='border-color:#e5e7eb;margin:24px 0 16px'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 분석 요약
# ──────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-title"><span class="tbar" style="background:#16a34a"></span>'
    f'잘하고 있는 점</div>',
    unsafe_allow_html=True,
)
good_rows = compare_df[compare_df["진단"].isin(["유사", "높음", "낮음"])]
if good_rows.empty:
    st.info("성공 기준에 부합하는 항목이 없습니다.")
else:
    good_cols = st.columns(min(len(good_rows), 3))
    for col, (_, row) in zip(good_cols * 5, good_rows.iterrows()):
        with col:
            diag_ko = DIAG_LABEL.get(row["진단"], row["진단"])
            color   = "#16a34a" if row["진단"] == "유사" else "#2563eb"
            st.markdown(
                f'<div class="yt-card" style="border-left:3px solid {color};height:100%">'
                f'<div style="font-size:10px;font-weight:700;color:{color};margin-bottom:4px">✅ {diag_ko}</div>'
                f'<div style="font-size:13px;font-weight:700;color:#111827;margin-bottom:6px">{row["항목"]}</div>'
                f'<div style="font-size:12px;color:#6b7280">'
                f'현재 <b style="color:#111827">{row["현재 채널"]}</b> · 성공 기준 {row["도메인 성공 패턴"]}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="sec-title"><span class="tbar" style="background:#ea580c"></span>'
    '개선이 필요한 점</div>',
    unsafe_allow_html=True,
)
bad_rows = compare_df[compare_df["진단"].isin(["부족", "과다"])]
if not tasks:
    st.success("우선 개선이 필요한 항목이 없습니다.")
else:
    imp_cols = st.columns(len(tasks))
    for col, task in zip(imp_cols, tasks):
        with col:
            st.markdown(
                f'<div class="yt-card" style="border-left:3px solid #ea580c;height:100%">'
                f'<div style="font-size:10px;font-weight:700;color:#ea580c;margin-bottom:4px">주의</div>'
                f'<div style="font-size:13px;font-weight:700;color:#111827;margin-bottom:6px">{task["title"]}</div>'
                f'<div style="font-size:12px;color:#374151;line-height:1.6;margin-bottom:8px">{task["desc"]}</div>'
                f'<div style="font-size:11px;background:#fff7ed;border-radius:6px;padding:8px 10px;'
                f'color:#c2410c;line-height:1.6">개선 방향: {task["action"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="sec-title"><span class="tbar" style="background:#6b7280"></span>종합 진단</div>',
    unsafe_allow_html=True,
)
n_good  = len(good_rows)
n_bad   = len(bad_rows)
n_total = len(compare_df)
overall_ok    = n_good >= n_total / 2
overall_label = "양호" if overall_ok else "개선 필요"
overall_color = "#16a34a" if overall_ok else accent_color
overall_bg    = "rgba(22,163,74,.08)"  if overall_ok else "rgba(220,38,38,.08)"
overall_bd    = "rgba(22,163,74,.25)"  if overall_ok else "rgba(220,38,38,.25)"

st.markdown(
    f'<div class="yt-card">'
    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
    f'<div style="font-size:12px;font-weight:700;color:{overall_color};'
    f'background:{overall_bg};border:1px solid {overall_bd};'
    f'border-radius:50px;padding:2px 12px">종합: {overall_label}</div>'
    f'<div style="font-size:11px;color:#6b7280">비교 {n_total}개 항목 중 적정 {n_good}개 · 개선 필요 {n_bad}개</div>'
    f'</div>'
    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">'
    f'<div style="background:#f9fafb;border-radius:8px;padding:12px">'
    f'<div style="font-size:11px;font-weight:600;color:#6b7280;margin-bottom:6px">현재 채널 특징</div>'
    f'<div style="font-size:12px;color:#374151;line-height:1.9">'
    f'영상 오프닝(3초 이내) 전략: <b>{persona_summary.get("main_first_3sec", "-")}</b><br>'
    f'모션그래픽: <b>{persona_summary.get("main_motion_graphic", "-")}</b><br>'
    f'주요 포맷: <b>{persona_summary.get("main_video_format", "-")}</b><br>'
    f'인물 비중: <b>{_fmt(persona_summary.get("person_ratio"))}</b>'
    f'</div></div>'
    f'<div style="background:#f9fafb;border-radius:8px;padding:12px">'
    f'<div style="font-size:11px;font-weight:600;color:#6b7280;margin-bottom:6px">핵심 개선 방향</div>'
    f'<div style="font-size:12px;color:#374151;line-height:1.7">'
    f'<b>전략:</b> {guideline.get("핵심 전략", "-")}<br><br>'
    f'<b>영상 오프닝(3초 이내):</b> {guideline.get("영상 오프닝(3초 이내) 전략", "-")}'
    f'</div></div>'
    f'</div>'
    f'<div style="font-size:11px;color:#9ca3af;border-top:1px solid #f3f4f6;padding-top:8px">'
    f'※ 최근 숏츠 {persona_summary.get("n", 0)}개 기반 빠른 진단입니다. 영상 수가 적을수록 대표성이 제한될 수 있습니다.'
    f'</div></div>',
    unsafe_allow_html=True,
)

st.markdown("<hr style='border-color:#e5e7eb;margin:24px 0 16px'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 숏츠 제작 가이드라인
# ──────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-title"><span class="tbar" style="background:{accent_color}"></span>'
    f'숏츠 제작 가이드라인 — {domain}</div>',
    unsafe_allow_html=True,
)
gl_left, gl_right = st.columns([3, 2])

with gl_left:
    for label, key in [
        ("핵심 전략",                   "핵심 전략"),
        ("영상 오프닝(3초 이내) 전략",  "영상 오프닝(3초 이내) 전략"),
        ("추천 영상 포맷",              "추천 포맷"),
        ("모션그래픽·자막 활용",        "모션그래픽 활용"),
        ("피해야 할 구성",              "피해야 할 구성"),
    ]:
        st.markdown(
            f'<div class="yt-card" style="margin-bottom:8px">'
            f'<div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:6px">{label}</div>'
            f'<div style="font-size:13px;color:#374151;line-height:1.7">{guideline[key]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with gl_right:
    st.markdown(
        '<div class="yt-card">'
        '<div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:10px">제작 전 체크리스트</div>',
        unsafe_allow_html=True,
    )
    for item in guideline["체크리스트"]:
        st.checkbox(item, value=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ideas_html = "".join(
        f'<div style="padding:6px 0;border-bottom:1px solid #f3f4f6;font-size:12px;color:#374151;line-height:1.6">{idea}</div>'
        for idea in guideline["다음 숏츠 기획안"]
    )
    st.markdown(
        f'<div class="yt-card">'
        f'<div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:10px">다음 숏츠 기획 아이디어</div>'
        f'{ideas_html}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border-color:#e5e7eb;margin:24px 0 16px'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# AI 종합 평가
# ──────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-title"><span class="tbar" style="background:{accent_color}"></span>'
    f'AI 종합 평가</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"분석 결과·요약·가이드라인을 종합해 AI가 {channel_input} 채널의 숏츠 전략을 평가합니다."
)

good_items_text = "\n".join(
    f"- {r['항목']}: 현재 {r['현재 채널']} / 성공 기준 {r['도메인 성공 패턴']} ({DIAG_LABEL.get(r['진단'], r['진단'])})"
    for _, r in good_rows.iterrows()
) or "해당 없음"

improve_items_text = "\n".join(
    f"- {r['항목']}: 현재 {r['현재 채널']} / 성공 기준 {r['도메인 성공 패턴']} ({DIAG_LABEL.get(r['진단'], r['진단'])})"
    for _, r in bad_rows.iterrows()
) or "해당 없음"

prompt_text = f"""너는 기업 유튜브 숏츠 전략 컨설턴트다.
아래 분석 데이터를 바탕으로 {channel_input} 채널의 숏츠 전략을 종합 평가하고, 실무자가 바로 실행할 수 있는 제언을 작성하라.

[채널 정보]
- 채널: {channel_input} / 도메인: {domain}
- 분석 숏츠 수: {persona_summary.get("n", 0)}개
- 영상 오프닝(3초 이내) 전략: {persona_summary.get("main_first_3sec", "-")}
- 모션그래픽: {persona_summary.get("main_motion_graphic", "-")}
- 주요 포맷: {persona_summary.get("main_video_format", "-")}
- 인물 비중: {_fmt(persona_summary.get("person_ratio"))} / 얼굴 비중: {_fmt(persona_summary.get("face_ratio"))} / 자막 비중: {_fmt(persona_summary.get("text_ratio"))}

[강점(Strengths)]
{good_items_text}

[취약점(Weaknesses)]
{improve_items_text}

[취약점 개선 방안]
{chr(10).join([f"{i+1}. {t['title']}: {t['action']}" for i, t in enumerate(tasks)])}

[가이드라인 요약]
- 핵심 전략: {guideline["핵심 전략"]}
- 영상 오프닝(3초 이내): {guideline["영상 오프닝(3초 이내) 전략"]}
- 추천 포맷: {guideline["추천 포맷"]}
- 피해야 할 구성: {guideline["피해야 할 구성"]}

[성공 기준 전체 비교]
{compare_df.to_markdown(index=False)}

아래 형식으로 작성하라:

## 1. 종합 진단
(현재 채널의 숏츠 전략 수준을 2~3문장으로 요약)

## 2. 강점(Strengths)
(성공 기준에 부합하는 강점을 구체적으로 서술)

## 3. 취약점(Weaknesses) 및 개선 방안
(가장 시급한 항목과 이유, 실행 방법 포함)

## 4. 숏츠 제작 전략 추천
(성공 기준을 반영한 구체적인 제작 방향)

## 5. 향후 숏츠 기획안 제안
(바로 실행 가능한 구체적인 기획안)

주의:
- 최근 {persona_summary.get("n", 0)}개 숏츠 기반 빠른 진단임을 명시할 것
- 데이터에 없는 내용을 과장하지 말 것
- 실무자가 바로 실행할 수 있도록 구체적으로 작성할 것"""

if "ai_eval_result" not in st.session_state:
    st.session_state["ai_eval_result"] = ""

btn_col, _ = st.columns([1, 3])
with btn_col:
    gen_btn = st.button("종합 평가 보고서 생성", key="gen_ai_btn", use_container_width=True)

if gen_btn:
    with st.spinner("AI가 종합 평가를 작성하는 중입니다..."):
        try:
            st.session_state["ai_eval_result"] = generate_ai_guideline(prompt_text)
        except Exception as e:
            st.error(f"AI 평가 생성 중 오류가 발생했습니다: {e}")

if st.session_state["ai_eval_result"]:
    with st.container(border=True):
        st.markdown(st.session_state["ai_eval_result"])

# ──────────────────────────────────────────────────────────────
# 보고서 저장
# ──────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#e5e7eb;margin:24px 0 16px'>", unsafe_allow_html=True)
st.markdown(
    '<div class="sec-title"><span class="tbar" style="background:#6b7280"></span>보고서 저장</div>',
    unsafe_allow_html=True,
)

ai_eval_text = st.session_state.get("ai_eval_result", "")
if not ai_eval_text:
    st.info("'AI 종합 평가 생성' 버튼을 먼저 눌러주세요. 평가 내용이 보고서에 함께 포함됩니다.")

if st.button("📄 보고서 저장 (.md)", key="save_report_btn"):
    report_md = f"""# 숏츠 영상 전략 진단 보고서

> **채널** {channel_input} | **도메인** {domain} | **분석 영상** {persona_summary.get("n", 0)}개 | **작성일** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. 채널 숏츠 분석 현황

| 항목 | 현재 값 |
|---|---|
| 영상 오프닝(3초 이내) 전략 | {persona_summary.get("main_first_3sec", "-")} |
| 모션그래픽 유형 | {persona_summary.get("main_motion_graphic", "-")} |
| 주요 영상 포맷 | {persona_summary.get("main_video_format", "-")} |
| 인물 비중 | {_fmt(persona_summary.get("person_ratio"))} |
| 얼굴 비중 | {_fmt(persona_summary.get("face_ratio"))} |
| 자막 비중 | {_fmt(persona_summary.get("text_ratio"))} |

---

## 2. 분석 요약

### 강점(Strengths)
{chr(10).join([f"- **{r['항목']}**: 현재 {r['현재 채널']} / 성공 기준 {r['도메인 성공 패턴']} → {DIAG_LABEL.get(r['진단'], r['진단'])}" for _, r in good_rows.iterrows()]) or "- 해당 없음"}

### 취약점(Weaknesses)
{chr(10).join([f"- **{r['항목']}**: 현재 {r['현재 채널']} / 성공 기준 {r['도메인 성공 패턴']} → {DIAG_LABEL.get(r['진단'], r['진단'])}" for _, r in bad_rows.iterrows()]) or "- 해당 없음"}

### 종합 진단
- 비교 항목 {n_total}개 중 적정 {n_good}개 · 개선 필요 {n_bad}개 → **{overall_label}**
- 핵심 방향: {guideline.get("핵심 전략", "-")}

---

## 3. 숏츠 제작 가이드라인

### 핵심 전략
{guideline["핵심 전략"]}

### 영상 오프닝(3초 이내) 전략
{guideline["영상 오프닝(3초 이내) 전략"]}

### 추천 영상 포맷
{guideline["추천 포맷"]}

### 모션그래픽·자막 활용
{guideline["모션그래픽 활용"]}

### 피해야 할 영상 구성
{guideline["피해야 할 구성"]}

### 우선 개선 과제
{chr(10).join([f"#### {i+1}. {t['title']}{chr(10)}{t['desc']}{chr(10)}> 실행 방법: {t['action']}" for i, t in enumerate(tasks)])}

### 숏츠 제작 전 체크리스트
{chr(10).join([f"- [ ] {x}" for x in guideline["체크리스트"]])}

### 향후 숏츠 기획 아이디어 제안
{chr(10).join([f"- {x}" for x in guideline["다음 숏츠 기획안"]])}

---

## 4. AI 종합 평가

{ai_eval_text if ai_eval_text else "_아직 AI 종합 평가가 생성되지 않았습니다._"}
"""
    report_path = (
        PERSONA_REPORT_DIR
        / f"report_{safe_filename(channel_input)}_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path.write_text(report_md, encoding="utf-8")
    st.success(f"보고서 저장 완료 — {report_path.name}")