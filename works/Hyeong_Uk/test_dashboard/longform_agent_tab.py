# ═══════════════════════════════════════════════════════════════════════════
# longform_predict_v2.py
# ─ YouTube 롱폼 영상 성과 예측 + 댓글 긍정 반응 예측 대시보드
#   (Gemini Embedding + XGBoost/Pipeline 기반 모델)
# ═══════════════════════════════════════════════════════════════════════════
#
# ── 연동 방식 ─────────────────────────────────────────────────────────────
#   이 탭은 개별 롱폼 영상 조건을 입력해 성과 예측과 AI 코멘트를 확인하는 독립 agent입니다.
#   결과는 가이드라인 페이지에 자동 반영하지 않고, 사용자가 Markdown 보고서로 직접 다운로드합니다.
#
# ── secrets.toml 작성 형식 (.streamlit/secrets.toml) ─────────────────────
#
#   GOOGLE_CLOUD_PROJECT    = "your-project-id"
#   GOOGLE_CLOUD_REGION     = "us-central1" 
#   GEMINI_EMBEDDING_MODEL  = "gemini-embedding-2"
#   GEMINI_PROMPT_MODEL     = "gemini-2.5-flash-lite"
#
# ── 추가 설치 패키지 ──────────────────────────────────────────────────────
#
#   uv add streamlit pydantic-ai[vertexai] google-genai joblib plotly shap
#
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
import bisect
import json
from pathlib import Path
from datetime import datetime
import re

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap as shap_lib
import streamlit as st
from google import genai
from google.genai import types
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# ── 페이지 설정은 app_test_dashboard.py에서 처리합니다. 탭 파일에서는 호출하지 않습니다.

# ── 경로 설정 ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
LONGFORM_MODELS = BASE_DIR / "longform" / "models"
COMMENT_MODELS  = BASE_DIR / "longform_comment" / "models"

# ── Secrets ───────────────────────────────────────────────────────────────
# 탭 import 시점에 secrets/model을 바로 읽으면 전체 앱이 깨질 수 있어,
# render_longform_agent_tab() 실행 시점에 지연 로드합니다.
GCP_PROJECT = None
GCP_REGION = "us-central1"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_PROMPT_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_DIM = 768

perf_model = None
perf_meta = None
perf_best_name = None
comment_model = None
comment_meta = None
comment_best_name = None


def _load_settings_from_secrets():
    """Streamlit secrets를 안전하게 읽습니다."""
    global GCP_PROJECT, GCP_REGION, GEMINI_EMBEDDING_MODEL, GEMINI_PROMPT_MODEL

    if "GOOGLE_CLOUD_PROJECT" not in st.secrets:
        raise KeyError(
            "GOOGLE_CLOUD_PROJECT가 secrets.toml에 없습니다. "
            "works/Hyeong_Uk/test_dashboard/.streamlit/secrets.toml 파일을 확인해 주세요."
        )

    GCP_PROJECT = st.secrets["GOOGLE_CLOUD_PROJECT"]
    GCP_REGION = st.secrets.get("GOOGLE_CLOUD_REGION", "us-central1")
    GEMINI_EMBEDDING_MODEL = st.secrets.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    GEMINI_PROMPT_MODEL = st.secrets.get("GEMINI_PROMPT_MODEL", "gemini-2.5-flash-lite")


def _ensure_agent_ready():
    """secrets와 모델을 렌더링 시점에 로드합니다."""
    global perf_model, perf_meta, perf_best_name, comment_model, comment_meta, comment_best_name

    _load_settings_from_secrets()

    if perf_model is None or comment_model is None:
        (
            perf_model,
            perf_meta,
            perf_best_name,
            comment_model,
            comment_meta,
            comment_best_name,
        ) = load_models()

# ── Gemini 임베딩 클라이언트 ──────────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    return genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location="global", # 예측을 위해 사용했던 임베딩 모델이 'global'에서만 지원되기 때문에 하드코딩으로 변경 
    )

# ── PydanticAI 에이전트 (AI 종합 요약) ───────────────────────────────────
@st.cache_resource
def get_comment_agent():
    provider = GoogleProvider(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_REGION,
    )
    return Agent(
        GoogleModel(GEMINI_PROMPT_MODEL, provider=provider),
        system_prompt=(
            "당신은 YouTube 마케팅 전문가입니다. "
            "두 AI 모델의 예측 결과를 바탕으로 실무자가 바로 활용할 수 있는 "
            "구체적인 전략을 한국어로 작성합니다. "
            "반드시 강점 / 약점 / 개선 방안 / 종합 요약 순서로 각 항목을 구분해 작성하세요."
        ),
    )

def get_ai_comment(prompt: str) -> str:
    """PydanticAI 의 run_sync() 로 실행.
    asyncio 이벤트 루프를 직접 건드리지 않아 Streamlit 재실행 시에도 안전.
    """
    agent = get_comment_agent()
    result = agent.run_sync(prompt)
    return result.output

# ── 모델 로드 ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # 성과 예측 모델
    perf_meta = json.loads((LONGFORM_MODELS / "meta.json").read_text(encoding="utf-8"))
    best_map = json.loads((LONGFORM_MODELS / "best_models.json").read_text(encoding="utf-8"))
    perf_best_name = best_map["ALL"]
    perf_model = joblib.load(LONGFORM_MODELS / f"ALL__{perf_best_name}.joblib")

    # 댓글 긍정 비율 모델
    comment_meta = json.loads((COMMENT_MODELS / "meta.json").read_text(encoding="utf-8"))
    # 댓글 긍정 반응 모델은 학습/운영 기준상 MLP 모델을 사용합니다.
    # 기존 best_model.json에 다른 이름이 남아 있어도 화면과 저장 보고서에는 MLP로 표기합니다.
    comment_best_name = "MLP"
    comment_model = joblib.load(COMMENT_MODELS / "longform_comment_model.joblib")

    return perf_model, perf_meta, perf_best_name, comment_model, comment_meta, comment_best_name

# ── 커스텀 CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .prob-high { color: #22c55e; }
    .prob-mid  { color: #ee890c; }
    .prob-low  { color: #ef4444; }
    .insight-box {
        border-radius: -5px; padding: 10px 14px;
        font-size: 0.88rem; margin-top: 8px;
    }
    .insight-green  { background: #f0fdf4; border-left: 4px solid #22c55e; color: #15803d; }
    .insight-orange { background: #fffbeb; border-left: 4px solid #ee890c; color: #92400e; }
    .insight-red    { background: #fef2f2; border-left: 4px solid #ef4444; color: #991b1b; }
    .sec-label {
        font-size: 1rem; font-weight: 700; color: #111827;
        letter-spacing: 0; margin: 1.2rem 0 0.2rem;
    }
    .sec-caption {
        font-size: 0.78rem; color: #6b7280; margin-bottom: 0.6rem;
    }
    .ai-section-title {
        font-size: 0.85rem; font-weight: 700; color: #e0e0e0;
        margin: 12px 0 4px;
    }
    .ai-section-body {
        font-size: 0.9rem; line-height: 1.8; opacity: 0.92;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper 함수들 ─────────────────────────────────────────────────────────
def make_text(title: str, desc: str) -> str:
    title = (title or "").strip()
    desc  = (desc or "").strip() or "(설명 없음)"
    return f"task: classification | query: TITLE: {title}\nDESCRIPTION: {desc}"

_LEN_THRESHOLDS = [30, 180, 480, 900]
_LEN_LABELS     = ["ultra_short", "short", "standard", "mid_ads", "long_form"]
_LEN_KR         = {
    "ultra_short": "30초 미만",
    "short": "30초 ~ 3분",
    "standard": "3 ~ 8분",
    "mid_ads": "8 ~ 15분",
    "long_form": "15분 이상",
}

def length_bucket(sec: float) -> str:
    return _LEN_LABELS[bisect.bisect_right(_LEN_THRESHOLDS, sec)]

def build_structured_row(d: dict) -> dict:
    desc = d.get("description") or ""
    tags_count = int(d.get("tags_count") or 0)
    sec = float(d.get("video_length_sec") or 0)
    return {
        "domain": str(d.get("domain", "FnB")),
        "description_missing_flag": int(not desc.strip()),
        "tags_missing_flag": int(tags_count == 0),
        "tags_count": tags_count,
        "영상길이(초)": sec,
        "caption": str(bool(d.get("caption", False))),
        "category_name": str(d.get("category_name", "unknown")),
        "length_bucket": length_bucket(sec),
        "cls_content_type": str(d.get("cls_content_type", "unknown")),
        "cls_marketing_purpose": str(d.get("cls_marketing_purpose", "unknown")),
        "cls_cta_type": str(d.get("cls_cta_type", "unknown")),
        "cls_is_series": str(bool(d.get("cls_is_series", False))),
        "cls_is_collaboration": str(bool(d.get("cls_is_collaboration", False))),
        "definition": str(d.get("definition", "hd")),
        "embeddable": str(bool(d.get("embeddable", True))),
        "has_paid_product_placement": str(bool(d.get("has_paid_product_placement", False))),
    }

LABEL_MAP = {
    "domain": "도메인",
    "description_missing_flag": "영상 설명 누락",
    "tags_missing_flag": "태그 누락",
    "tags_count": "태그 수",
    "영상길이(초)": "영상 길이(초)",
    "caption": "자막",
    "category_name": "카테고리",
    "length_bucket": "영상 길이 구간",
    "cls_content_type": "콘텐츠 유형",
    "cls_marketing_purpose": "마케팅 목적",
    "cls_cta_type": "CTA 유형",
    "cls_is_series": "시리즈 여부",
    "cls_is_collaboration": "콜라보 여부",
    "definition": "화질",
    "embeddable": "외부 임베드 허용",
    "has_paid_product_placement": "PPL·협찬 포함",
}

# True/False 변환 대상 피처 분류
_YESNO_KEYS = {"cls_is_series", "cls_is_collaboration"}         # 예/아니오
_USAGE_KEYS = {"caption", "embeddable", "has_paid_product_placement"}  # 사용/사용하지 않음

_BOOL_KR = {
    "hd": "HD(720p 이상)", "sd": "SD(720p 미만)",
}
_YESNO_KR = {
    "True": "예", "False": "아니오",
}
_USAGE_KR = {
    "True": "사용", "False": "사용하지 않음",
}

def feat_label(col: str, struct_row: dict) -> str:
    """피처명 + 입력값을 사용자가 읽기 쉬운 '라벨: 값' 형태로 반환."""
    base = col.split("__", 1)[-1] if "__" in col else col
    for key, label in LABEL_MAP.items():
        if base == key or base.startswith(key + "_"):
            val    = struct_row.get(key, "")
            suffix = base[len(key):].lstrip("_")
            raw    = suffix if suffix else str(val)

            if key in _YESNO_KEYS:
                display = _YESNO_KR.get(raw, raw)
            elif key in _USAGE_KEYS:
                display = _USAGE_KR.get(raw, raw)
            else:
                display = _BOOL_KR.get(raw, raw)

            return f"{label}: {display}"
    return col

def gauge_chart(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "steps": [],
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig

def parse_ai_sections(text: str) -> dict:
    """AI 응답 텍스트를 강점/약점/개선 방안/종합 요약 4개 섹션으로 파싱.
    ##, **, - 등 마크다운 기호가 붙어도 파싱되도록 처리.
    """
    keys = ["강점", "약점", "개선 방안", "종합 요약"]
    result  = {k: "" for k in keys}
    current = None
    for line in text.splitlines():
        # 마크다운 기호(#, *, -) 제거 후 키워드 매칭
        stripped = line.strip().lstrip("#").lstrip("*").lstrip("-").strip()
        matched  = next((k for k in keys if stripped.startswith(k)), None)
        if matched:
            current = matched
            rest = stripped[len(matched):].lstrip(": ").strip()
            if rest:
                result[current] += rest + "\n"
        elif current:
            result[current] += line + "\n"
    return {k: v.strip() for k, v in result.items()}

def render_markdown_in_html(text: str) -> str:
    """**텍스트** → <strong>텍스트</strong> 변환 (HTML div 안에서 굵은 글씨 처리)."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)  # 굵은 글씨
    text = re.sub(r'^\* ', '• ', text, flags=re.MULTILINE)          # 불릿 기호
    return text

# ═══════════════════════════════════════════════════════════════════════════
# 선택지 정의
# ═══════════════════════════════════════════════════════════════════════════
CONTENT_TYPE_OPTIONS = [
    "브이로그", "인터뷰", "제품리뷰", "웹예능", "기술설명", "튜토리얼",
    "에피소드소개", "이벤트/행사", "시설소개", "요리/레시피", "웹드라마",
    "광고/CF", "다큐멘터리", "애니메이션", "영양정보", "고객후기", "기타",
]
MARKETING_PURPOSE_OPTIONS = [
    "브랜드캠페인", "제품홍보", "고객유입", "고객유지", "기업이미지",
    "채용", "사회공헌/환경", "서비스활용", "정보제공", "기타",
]
CTA_TYPE_OPTIONS = [
    "구매유도", "구독유도", "이벤트참여", "정보탐색", "앱다운로드", "방문유도", "기타",
]
CATEGORY_NAME_OPTIONS = [
    "영화/애니메이션", "자동차", "음악", "반려동물", "스포츠", "여행/이벤트",
    "게임", "브이로그/라이프스타일", "코미디", "예능/오락 콘텐츠", "뉴스/정치", 
    "정보형 콘텐츠(튜토리얼)", "교육", "과학/기술", "비영리/사회운동", "기타",
]


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 2~5 렌더링 함수 (session_state 에서 호출)
# ═══════════════════════════════════════════════════════════════════════════
def _render_results(r: dict):
    """예측 결과 섹션 2~5를 렌더링. 새로고침 없이 재예측 시에도 결과 유지."""
    prob_val   = r["prob_val"]
    sent_val   = r["sent_val"]
    prob_pct   = r["prob_pct"]
    sent_pct   = r["sent_pct"]
    shap_top5  = r["shap_top5"]
    sections   = r["sections"]
    struct_row = r["struct_row"]
    dom_key          = r["dom_key"]
    title_input      = r["title_input"]
    content_type     = r["content_type"]
    marketing_purpose = r["marketing_purpose"]
    cta_type         = r["cta_type"]
    category_name    = r["category_name"]
    video_min        = r["video_min"]
    video_sec_extra  = r["video_sec_extra"]
    lb               = r["lb"]
    tags_count_val   = r["tags_count_val"]
    desc_input       = r["desc_input"]
    caption_use      = r["caption_use"]
    definition       = r["definition"]
    cls_is_series    = r["cls_is_series"]
    cls_is_collaboration = r["cls_is_collaboration"]
    has_paid         = r["has_paid"]

    # ═══════════════════════════════════════════════════════
    # 섹션 2 ─ 예측 결과
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-label">예측 결과</div>', unsafe_allow_html=True)

    res_c1, res_c2 = st.columns(2)

    with res_c1:
        perf_color = "#22c55e" if prob_val >= 0.7 else ("#ee890c" if prob_val >= 0.4 else "#ef4444")
        perf_label = "상위 18% 수준" if prob_val >= 0.7 else ("상위 40% 수준" if prob_val >= 0.4 else "하위 40% 수준")
        st.plotly_chart(gauge_chart(prob_val, "영상 성과 예측 (성공 확률)", perf_color), use_container_width=True)
        st.caption(f"등급: **{perf_label}**  |  모델: {perf_best_name}")

    with res_c2:
        sent_color = "#22c55e" if sent_val >= 0.7 else ("#ee890c" if sent_val >= 0.5 else "#ef4444")
        sent_label = "매우 긍정적" if sent_val >= 0.7 else ("보통" if sent_val >= 0.5 else "주의 필요")
        st.plotly_chart(gauge_chart(sent_val, "댓글 긍정 반응 예측", sent_color), use_container_width=True)
        st.caption(f"상태: **{sent_label}**  |  모델: {comment_best_name}")

    # 종합 판정 배너
    both_good = prob_val >= 0.5 and sent_val >= 0.55
    both_bad  = prob_val <  0.4 and sent_val <  0.5
    if both_good:
        banner_cls, banner_icon = "insight-green",  "✅"
        banner_text = "성과·댓글 반응 모두 긍정적으로 예측됩니다. 업로드를 적극 권장합니다."
    elif both_bad:
        banner_cls, banner_icon = "insight-red",    "⚠️"
        banner_text = "성과·댓글 반응 모두 낮게 예측됩니다. 콘텐츠 전략 재검토를 권장합니다."
    else:
        banner_cls, banner_icon = "insight-orange", "🔶"
        banner_text = "성과와 댓글 반응 중 하나가 개선이 필요합니다. 아래 분석을 참고하세요."

    st.markdown(f"""
    <div class="insight-box {banner_cls}">
        {banner_icon} {banner_text}
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # 섹션 3 ─ 예측에 영향을 준 요인 TOP 5 (SHAP)
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-label">예측에 영향을 준 요인 TOP 5</div>', unsafe_allow_html=True)

    if shap_top5:
        labels     = [label for label, _, _ in shap_top5]
        abs_vals   = [abs_v for _, abs_v, _ in shap_top5]
        sign_vals  = [signed for _, _, signed in shap_top5]
        bar_colors = ["#22c55e" if v > 0 else "#ef4444" for v in sign_vals]
        directions = ["▲ 유리" if v > 0 else "▼ 불리" for v in sign_vals]

        fig = go.Figure(go.Bar(
            x=abs_vals,
            y=labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{d}  ({v:.3f})" for d, v in zip(directions, abs_vals)],
            textposition="outside",
            hovertemplate="%{y}<br>SHAP 영향도: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=160),
            xaxis_title="SHAP 영향도 (클수록 예측에 더 큰 영향)",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            xaxis=dict(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=2),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("▲ 성공 확률을 높이는 요인   ▼ 성공 확률을 낮추는 요인   |   막대 길이 = 예측에 미친 영향의 크기")
    else:
        st.info("SHAP 분석을 수행할 수 없습니다.")

    # ═══════════════════════════════════════════════════════
    # 섹션 4 ─ AI 종합 요약
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-label">AI 종합 요약</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:12px;
         padding:20px;color:white;margin-top:4px;">
        <div class="ai-section-title">강점</div>
        <div class="ai-section-body">{render_markdown_in_html(sections['강점'] or '-')}</div>
        <div class="ai-section-title" style="margin-top:14px;">약점</div>
        <div class="ai-section-body">{render_markdown_in_html(sections['약점'] or '-')}</div>
        <div class="ai-section-title" style="margin-top:14px;">개선 방안</div>
        <div class="ai-section-body">{render_markdown_in_html(sections['개선 방안'] or '-')}</div>
        <div class="ai-section-title" style="margin-top:14px;">종합 요약</div>
        <div class="ai-section-body">{render_markdown_in_html(sections['종합 요약'] or '-')}</div>
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # 섹션 5 ─ 예측 결과 다운로드
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-label">예측 결과 보고서 다운로드</div>', unsafe_allow_html=True)
    st.caption("이 결과는 가이드라인 페이지에 자동 반영하지 않습니다. 필요한 경우 Markdown 파일로 직접 다운로드해 활용하세요.")

    top5_text = "\n".join(
        [f"  - {label}: {'유리' if signed > 0 else '불리'} (영향도 {abs_v:.3f})"
         for label, abs_v, signed in shap_top5]
    ) if shap_top5 else "  - SHAP 분석 불가"

    md_lines = [
        "# YouTube 롱폼 영상 성과 예측 결과",
        "",
        "## 입력 조건",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 도메인 | {dom_key} |",
        f"| 영상 제목 | {title_input} |",
        f"| 콘텐츠 유형 | {content_type} |",
        f"| 마케팅 목적 | {marketing_purpose} |",
        f"| CTA 유형 | {cta_type} |",
        f"| 카테고리 | {category_name} |",
        f"| 영상 길이 | {video_min}분 {video_sec_extra}초 ({_LEN_KR.get(lb, lb)}) |",
        f"| 태그 수 | {tags_count_val}개 |",
        f"| 영상 설명 | {'있음' if desc_input.strip() else '없음'} |",
        f"| 자막 | {'있음' if caption_use else '없음'} |",
        f"| 화질 | {definition.upper()} |",
        f"| 시리즈 여부 | {'예' if cls_is_series else '아니오'} |",
        f"| 콜라보 여부 | {'예' if cls_is_collaboration else '아니오'} |",
        f"| PPL·협찬 | {'예' if has_paid else '아니오'} |",
        "",
        "## 예측 결과",
        "| 모델 | 예측값 |",
        "|---|---|",
        f"| 영상 성과 (성공 확률) | **{prob_pct}%** |",
        f"| 댓글 긍정 반응 비율 | **{sent_pct}%** |",
        "",
        "## 예측에 영향을 준 주요 요인 TOP 5",
    ]

    if shap_top5:
        md_lines += ["| 요인 | 방향 | 영향도 |", "|---|---|---|"]
        for label, abs_v, signed in shap_top5:
            direction = "▲ 유리" if signed > 0 else "▼ 불리"
            md_lines.append(f"| {label} | {direction} | {abs_v:.3f} |")
    else:
        md_lines.append("SHAP 분석 불가")

    md_lines += [
        "",
        "## AI 종합 요약",
        "",
        "### 강점",
        sections.get("강점", "-"),
        "",
        "### 약점",
        sections.get("약점", "-"),
        "",
        "### 개선 방안",
        sections.get("개선 방안", "-"),
        "",
        "### 종합 요약",
        sections.get("종합 요약", "-"),
    ]

    md_content = "\n".join(md_lines)

    safe_title = re.sub(r"[^\w가-힣-]+", "_", (title_input or "롱폼_영상")).strip("_")[:40] or "롱폼_영상"
    file_name = f"롱폼_성과_예측_보고서_{dom_key}_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    st.download_button(
        "Markdown 보고서 다운로드",
        data=md_content,
        file_name=file_name,
        mime="text/markdown",
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# 메인 페이지
# ═══════════════════════════════════════════════════════════════════════════
def page_simulator():
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:20px;
                    padding:20px 22px;box-shadow:0 5px 18px rgba(15,23,42,.045);margin-bottom:16px;">
            <div style="font-size:23px;font-weight:800;color:#111827;margin-bottom:6px;">
                🤖 롱폼 영상 성과 예측 agent
            </div>
            <div style="font-size:13.5px;color:#6b7280;line-height:1.7;word-break:keep-all;">
                영상 정보를 입력하면 <b>성과 성공 확률</b>과 <b>댓글 긍정 반응 비율</b>을 함께 예측합니다.
                예측에 영향을 준 주요 요인과 AI 전략 코멘트를 확인하고, 결과는 Markdown 보고서로 다운로드할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        _ensure_agent_ready()
    except Exception as e:
        st.error("롱폼 agent 실행에 필요한 설정 또는 모델 파일을 불러오지 못했습니다.")
        st.caption("secrets.toml, longform/models, longform_comment/models 경로를 확인해 주세요.")
        st.exception(e)
        return

    st.divider()

    # ═══════════════════════════════════════════════════════
    # 섹션 1 ─ 영상 데이터 입력
    # ═══════════════════════════════════════════════════════

    # ── 직접 입력 ─────────────────────────────────────────
    st.markdown('<div class="sec-label">영상 기본 정보</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-caption">제목, 설명, 태그 수, 영상 길이를 입력하세요.</div>', unsafe_allow_html=True)

    title_input = st.text_input(
        "영상 제목*",
        placeholder="예: [화제의 대상 ep.10] 대체당부터 K-분식까지! 신규 브랜드 첫만남 댓글이벤트",
        help="영상의 제목과 설명은 Gemini 임베딩 모델을 통해 예측에 반영됩니다.",
    )
    desc_input = st.text_area(
        "영상 설명",
        placeholder="유튜브 영상 설명란에 작성할 내용을 입력하세요.\n비워두면 '(설명 없음)'으로 처리됩니다.",
        height=100,
    )

    ti_c1, ti_c2 = st.columns(2)
    with ti_c1:
        tags_input = st.text_input(
            "영상 설명 내 태그",
            placeholder="예: #브이로그, #맛집, #서울, #신상, #리뷰",
            help="해시태그(#)로 구분해 입력하면 태그 수가 자동으로 계산됩니다. 15~30개를 권장합니다.",
        )
        if tags_input.strip():
            # 쉼표가 있으면 쉼표로 구분, 없으면 # 기준으로 구분
            if "," in tags_input:
                tags_count_val = len([t for t in tags_input.split(",") if t.strip()])
            else:
                tags_count_val = len([t for t in re.split(r'#', tags_input) if t.strip()])
        else:
            tags_count_val = 0
            
        st.caption(f"태그 {tags_count_val}개 입력됨")
        
    with ti_c2:
        lb_placeholder = st.empty()
        len_c1, len_c2, len_c3, len_c4 = st.columns([3, 0.5, 3, 0.5])
        with len_c1:
            video_min = st.number_input("", min_value=0, max_value=300, value=5, step=1, label_visibility="collapsed", key="longform_agent_video_min")
        with len_c2:
            st.markdown("<div style='padding-top:-5px;font-size:0.9rem;'>분</div>", unsafe_allow_html=True)
        with len_c3:
            video_sec_extra = st.number_input("", min_value=0, max_value=59, value=0, step=1, label_visibility="collapsed", key="longform_agent_video_sec")
        with len_c4:
            st.markdown("<div style='padding-top:-5px;font-size:0.9rem;'>초</div>", unsafe_allow_html=True)
        
        video_length_sec = video_min * 60 + video_sec_extra
        lb = length_bucket(video_length_sec)
        lb_placeholder.markdown(
            f'<label style="font-size:0.875rem;font-weight:400;color:inherit;'
            f'display:block;margin-bottom:0;line-height:1;">'
            f'영상 길이 &nbsp;→&nbsp;{_LEN_KR.get(lb, lb)}</label>',
            unsafe_allow_html=True,
        )

    # ── 선택 입력 ─────────────────────────────────────────
    st.markdown('<div class="sec-label">영상 분류 정보</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-caption">도메인, 카테고리, 콘텐츠 유형 등 영상의 성격을 선택하세요.</div>', unsafe_allow_html=True)  

    sel_c1, sel_c2 = st.columns(2)
    with sel_c1:
        domain_sim = st.selectbox("도메인", ["FnB (식음료)", "IT"], key="longform_agent_domain")
    with sel_c2:
        category_name = st.selectbox("유튜브 카테고리", CATEGORY_NAME_OPTIONS, key="longform_agent_category")

    sel_c3, sel_c4, sel_c5 = st.columns(3)
    with sel_c3:
        content_type = st.selectbox(
            "콘텐츠 유형", CONTENT_TYPE_OPTIONS,
            help="영상이 어떤 형식으로 만들어졌는지 선택하세요.",
            key="longform_agent_content_type",
        )
    with sel_c4:
        marketing_purpose = st.selectbox(
            "마케팅 목적", MARKETING_PURPOSE_OPTIONS,
            help="이 영상을 올리는 목적을 선택하세요.",
            key="longform_agent_marketing_purpose",
        )
    with sel_c5:
        cta_type = st.selectbox(
            "CTA 유형", CTA_TYPE_OPTIONS,
            help="영상에서 시청자에게 유도하는 다음 행동을 선택하세요.",
            key="longform_agent_cta_type",
        )

    sel_c6, _ = st.columns([1, 3])
    with sel_c6:
        definition = st.selectbox(
            "화질",
            ["hd", "sd"],
            format_func=lambda x: "HD (720p 이상)" if x == "hd" else "SD (720p 미만)",
            key="longform_agent_definition",
        )

    # ── 토글 입력 ─────────────────────────────────────────
    st.markdown('<div class="sec-label">영상 세부 설정</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-caption">자막, 시리즈 여부 등 영상의 설정값을 선택하세요.</div>', unsafe_allow_html=True)

    tog_c1, tog_c2, tog_c3, tog_c4, tog_c5 = st.columns(5)
    with tog_c1:
        caption_use = st.toggle("자막 포함 여부", value=True, key="longform_agent_caption")
    with tog_c2:
        embeddable = st.toggle("외부 임베드 허용 여부", value=True, key="longform_agent_embed")
    with tog_c3:
        cls_is_series = st.toggle("시리즈물 여부", value=False)
    with tog_c4:
        cls_is_collaboration = st.toggle("콜라보 여부", value=False)
    with tog_c5:
        has_paid = st.toggle("PPL·협찬 포함 여부", value=False)

    # ═══════════════════════════════════════════════════════
    # 예측 버튼
    # ═══════════════════════════════════════════════════════
    st.info("제목과 설명을 분석하는 과정이 포함되어 있어 예측 완료까지 약 30초가량 소요될 수 있습니다.")
    st.divider()
    predict_clicked = st.button(
        "🔮 입력된 영상에 대한 예측 시작",
        use_container_width=True,
        type="primary",
    )

    if not predict_clicked:
        if "last_result" in st.session_state:
            _render_results(st.session_state["last_result"])
        return

    if not title_input.strip():
        st.warning("영상 제목을 입력해주세요. 제목은 임베딩 모델의 핵심 입력입니다.")
        return

    dom_key = "FnB" if "FnB" in domain_sim else "IT"

    # ═══════════════════════════════════════════════════════
    # 임베딩 생성
    # ═══════════════════════════════════════════════════════
    with st.spinner("입력된 정보를 바탕으로 분석 중입니다. 잠시만 기다려주세요..."):
        try:
            client = get_gemini_client()
            resp   = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=make_text(title_input, desc_input),
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            )
            emb = np.array(resp.embeddings[0].values, dtype=np.float32)
        except Exception as e:
            st.error(f"임베딩 생성 오류: {e}")
            return

    # ═══════════════════════════════════════════════════════
    # 구조적 피처 행 구성
    # ═══════════════════════════════════════════════════════
    input_dict = {
        "domain": dom_key,
        "description": desc_input,
        "tags_count": tags_count_val,
        "video_length_sec": video_length_sec,
        "caption": caption_use,
        "category_name": category_name,
        "definition": definition,
        "embeddable": embeddable,
        "has_paid_product_placement": has_paid,
        "cls_content_type": content_type,
        "cls_marketing_purpose": marketing_purpose,
        "cls_cta_type": cta_type,
        "cls_is_series": cls_is_series,
        "cls_is_collaboration": cls_is_collaboration,
    }
    struct_row = build_structured_row(input_dict)

    # ═══════════════════════════════════════════════════════
    # 성과 예측
    # ═══════════════════════════════════════════════════════
    emb_row_perf = {col: float(v) for col, v in zip(perf_meta["embedding_feature_cols"], emb)}
    full_perf = {**struct_row, **emb_row_perf}
    X_perf = pd.DataFrame([{c: full_perf[c] for c in perf_meta["all_feature_cols"]}])

    try:
        prob_val = float(perf_model.predict_proba(X_perf)[0, 1])
    except Exception as e:
        st.warning(f"성과 모델 예측 오류: {e}")
        prob_val = 0.5

    # ═══════════════════════════════════════════════════════
    # 댓글 긍정 비율 예측
    # ═══════════════════════════════════════════════════════
    emb_row_comment = {col: float(v) for col, v in zip(comment_meta["embedding_feature_cols"], emb)}
    full_comment = {**struct_row, **emb_row_comment}
    X_comment = pd.DataFrame([{c: full_comment[c] for c in comment_meta["all_feature_cols"]}])

    try:
        sent_val = float(np.clip(comment_model.predict(X_comment)[0], 0, 1))
    except Exception as e:
        st.warning(f"댓글 모델 예측 오류: {e}")
        sent_val = 0.5

    prob_pct = round(prob_val * 100, 1)
    sent_pct = round(sent_val * 100, 1)

    # 예측 결과를 session_state 에 저장 (새로고침 없이 재예측 시에도 결과 유지)
    st.session_state["last_result"] = {
        "prob_val": prob_val, "sent_val": sent_val,
        "prob_pct": prob_pct, "sent_pct": sent_pct,
        "shap_top5": [],   # SHAP 계산 후 업데이트
        "sections": {},    # AI 요약 생성 후 업데이트
        "struct_row": struct_row,
        "dom_key": dom_key, "title_input": title_input,
        "content_type": content_type, "marketing_purpose": marketing_purpose,
        "cta_type": cta_type, "category_name": category_name,
        "video_min": video_min, "video_sec_extra": video_sec_extra,
        "lb": lb, "tags_count_val": tags_count_val,
        "desc_input": desc_input, "caption_use": caption_use,
        "definition": definition, "cls_is_series": cls_is_series,
        "cls_is_collaboration": cls_is_collaboration, "has_paid": has_paid,
    }

    # ═══════════════════════════════════════════════════════
    # SHAP 계산 (session_state 업데이트)
    # ═══════════════════════════════════════════════════════
    try:
        preprocessor  = perf_model[:-1]
        final_est     = perf_model[-1]
        X_transformed = preprocessor.transform(X_perf)

        try:
            feat_names = list(preprocessor.get_feature_names_out())
        except AttributeError:
            feat_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        explainer = shap_lib.TreeExplainer(final_est)
        sv        = explainer(X_transformed).values[0]

        # 임베딩 피처 인덱스 / 구조적 피처 인덱스 분리
        emb_col_set = set(perf_meta.get("embedding_feature_cols", []))
        emb_idx, struct_idx = [], []
        for i, fn in enumerate(feat_names):
            base = fn.split("__", 1)[-1] if "__" in fn else fn
            (emb_idx if base in emb_col_set else struct_idx).append(i)

        # ── 학습 코드의 quantify_shap_grouped 와 동일한 로직 ──────────────
        # 임베딩 768개 → "제목·설명" 한 줄로 합산
        #   mean_abs_shap = sum(|sv[i]|) for all emb_idx   (학습 코드와 동일)
        #   방향(부호)     = sign(sum(sv[i]))               (mean_shap 부호 기준)
        emb_abs_sum  = float(np.sum(np.abs(sv[emb_idx])))
        emb_sign_sum = float(np.sum(sv[emb_idx]))
        emb_signed   = emb_abs_sum * np.sign(emb_sign_sum)

        grouped = [("제목·설명", emb_abs_sum, emb_signed)]
        for i in struct_idx:
            fn  = feat_names[i]
            val = float(sv[i])
            grouped.append((feat_label(fn, struct_row), abs(val), val))

        shap_top5 = sorted(grouped, key=lambda x: x[1], reverse=True)[:5]
        st.session_state["last_result"]["shap_top5"] = shap_top5  # session_state 업데이트

    except Exception:
        shap_top5 = []

    # ═══════════════════════════════════════════════════════
    # AI 종합 요약 생성 (session_state 업데이트)
    # ═══════════════════════════════════════════════════════
    top5_text = "\n".join(
        [f"  - {label}: {'유리' if signed > 0 else '불리'} (영향도 {abs_v:.3f})"
         for label, abs_v, signed in shap_top5]
    ) if shap_top5 else "  - SHAP 분석 불가"

    ai_prompt = f"""
아래는 YouTube 영상 업로드 조건과 두 AI 모델의 예측 결과입니다.

[입력 조건]
- 도메인: {dom_key}
- 영상 제목: {title_input}
- 콘텐츠 유형: {content_type} / 마케팅 목적: {marketing_purpose} / CTA: {cta_type}
- 카테고리: {category_name}
- 영상 길이: {video_min}분 {video_sec_extra}초 ({_LEN_KR.get(lb, lb)} 구간)
- 태그 수: {tags_count_val}개 / 영상 설명: {'있음' if desc_input.strip() else '없음'}
- 자막: {'있음' if caption_use else '없음'} / 화질: {definition.upper()}
- 시리즈: {'예' if cls_is_series else '아니오'} / 콜라보: {'예' if cls_is_collaboration else '아니오'} / PPL: {'예' if has_paid else '아니오'}

[예측 결과]
- 영상 성과 예측 (성공 확률): {prob_pct}%
- 댓글 긍정 반응 비율 예측: {sent_pct}%

[예측에 영향을 준 주요 요인 TOP 5]
{top5_text}

위 내용을 바탕으로 반드시 아래 4개 항목을 순서대로, 각 항목 이름을 첫 줄에 써서 작성해주세요.

강점: (이 영상의 강점)
약점: (이 영상의 약점)
개선 방안: (구체적인 개선 방안)
종합 요약: (전체를 아우르는 한 단락 요약)
"""

    with st.spinner("AI 종합 요약 생성 중..."):
        try:
            ai_raw = get_ai_comment(ai_prompt)
        except Exception as e:
            ai_raw = f"강점: 분석 오류\n약점: {e}\n개선 방안: -\n종합 요약: -"

    sections = parse_ai_sections(ai_raw)
    st.session_state["last_result"]["sections"] = sections  # session_state 업데이트

    # 섹션 2~5 렌더링
    _render_results(st.session_state["last_result"])


# ── 진입점 ──
if __name__ == "__main__":
    page_simulator()


def render_longform_agent_tab():
    """app_test_dashboard.py의 '롱폼 분석 agent' 탭에서 호출하는 함수."""
    page_simulator()



def render_longform_agent():
    """기존 app_test_dashboard.py import 이름과 호환하기 위한 alias 함수."""
    render_longform_agent_tab()
