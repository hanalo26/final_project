"""
TubeStrategy — 유튜브 기업 마케팅 채널 썸네일 전략 대시보드
- 롱폼/숏폼 자동 구분 (60초 기준)
- 썸네일 클릭 → Gemini Vision 분석 팝업
- 이미지 생성: imagen-4.0-generate-001 (스타일 참고 재생성)
- 프롬프트 생성: gemini-2.5-flash-lite
- 썸네일 비전 분석: gemini-2.5-flash
- 개선 탭: YouTube URL 입력 → 썸네일 자동 분석 + 개선 생성
- 보고서/이미지생성 설정: REPORT_CONFIG / IMAGE_GEN_CONFIG (코드 상단에서 수정)
- 업로드 요일 전략 제외
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
import base64
import re
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
# Vertex AI
import vertexai
from vertexai.generative_models import (
    GenerativeModel, Part, GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold
)
try:
    from vertexai.vision_models import ImageGenerationModel
except ImportError:
    from vertexai.preview.vision_models import ImageGenerationModel

# Pydantic 응답 스키마
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

# ══════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="TubeStrategy",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

/* ── 기본 ── */
html,body,[class*="css"]{font-family:'Roboto','Noto Sans KR',sans-serif;background:#f8f9fa!important;color:#111827!important}
.stApp{background:#f8f9fa}
#MainMenu,footer{visibility:hidden}
header{visibility:hidden}
.block-container{padding:.5rem 1.5rem 2rem 1.5rem!important;max-width:100%}

/* ── 사이드바 ── */
[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid #e5e7eb!important}
[data-testid="stSidebar"] *{color:#111827!important}

/* ── 사이드바 토글 버튼 ── */
[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stExpandSidebarButton"] {
    background:#ff0000!important;
    border-radius:6px!important;
    width:32px!important;
    height:32px!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    opacity:1!important;
    visibility:visible!important;
    border:none!important;
}
[data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="stExpandSidebarButton"] span,
[data-testid="stBaseButton-headerNoPadding"] span span,
[data-testid="stExpandSidebarButton"] span span {
    color:#fff!important;
    font-size:18px!important;
}

/* ── 탭 ── */
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid #e5e7eb!important;gap:0;padding:0}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#6b7280!important;font-size:13px;font-weight:500;padding:10px 18px!important;border-radius:0!important;border-bottom:2px solid transparent!important;margin-bottom:-1px}
.stTabs [aria-selected="true"]{color:#111827!important;border-bottom:2px solid #111827!important;background:transparent!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}

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
.stExpander summary:hover{color:#000000!important;background:#f9fafb!important}
.stExpander details{background:#ffffff!important}
.stExpander details[open] summary{background:#ffffff!important;color:#111827!important}
.stExpander details summary::-webkit-details-marker{color:#111827!important}
.stExpander [data-testid="stExpanderDetails"]{border-top:1px solid #e5e7eb!important;padding-top:12px!important;background:#ffffff!important}
details[open]>summary{background:#ffffff!important;color:#111827!important}
[data-testid="stExpander"] details[open]{background:#ffffff!important}
[data-testid="stExpander"] details[open] summary{background:#ffffff!important;color:#111827!important}

/* ── 스크롤바 ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f3f4f6}
::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:3px}

/* ── 카드 ── */
.yt-card{background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin-bottom:12px}
.stat-box{background:#ffffff;border-radius:8px;padding:12px 8px;text-align:center;border:1px solid #e5e7eb}
.stat-val{font-size:20px;font-weight:700;margin-bottom:3px}
.stat-lbl{font-size:11px;color:#6b7280}

/* ── 섹션 타이틀 ── */
.sec-title{font-size:14px;font-weight:600;color:#111827;display:flex;align-items:center;gap:8px;margin-bottom:10px}
.tbar{width:3px;height:16px;border-radius:2px;display:inline-block;flex-shrink:0}

/* ── 배지 ── */
.badge{display:inline-block;padding:2px 9px;border-radius:50px;font-size:11px;font-weight:600;margin:2px}
.badge-red{background:rgba(220,38,38,.08);color:#dc2626;border:1px solid rgba(220,38,38,.25)}
.badge-blue{background:rgba(37,99,235,.08);color:#2563eb;border:1px solid rgba(37,99,235,.25)}
.badge-green{background:rgba(22,163,74,.08);color:#16a34a;border:1px solid rgba(22,163,74,.25)}
.badge-yellow{background:rgba(161,98,7,.08);color:#a16207;border:1px solid rgba(161,98,7,.25)}
.badge-gray{background:rgba(107,114,128,.08);color:#6b7280;border:1px solid rgba(107,114,128,.2)}
.badge-orange{background:rgba(234,88,12,.08);color:#ea580c;border:1px solid rgba(234,88,12,.25)}

/* ── 알림 박스 ── */
.api-notice{background:rgba(161,98,7,.06);border:1px solid rgba(161,98,7,.25);border-radius:8px;padding:9px 13px;font-size:11px;color:#92400e;line-height:1.7;margin-bottom:10px}
.guide-hint{background:rgba(37,99,235,.06);border:1px solid rgba(37,99,235,.25);border-radius:8px;padding:9px 13px;font-size:11px;color:#1e40af;line-height:1.7;margin-bottom:10px}
.good-point{background:rgba(22,163,74,.06);border-left:3px solid #16a34a;border-radius:0 8px 8px 0;padding:9px 12px;font-size:12px;line-height:1.6;margin-bottom:7px;color:#14532d}
.bad-point{background:rgba(234,88,12,.06);border-left:3px solid #ea580c;border-radius:0 8px 8px 0;padding:9px 12px;font-size:12px;line-height:1.6;margin-bottom:7px;color:#7c2d12}
.action-point{background:rgba(37,99,235,.06);border-left:3px solid #2563eb;border-radius:0 8px 8px 0;padding:9px 12px;font-size:12px;line-height:1.6;margin-bottom:7px;color:#1e3a5f}

/* ── 프로그레스 ── */
.prog-row{margin-bottom:9px}
.prog-label-row{display:flex;justify-content:space-between;font-size:11px;color:#6b7280;margin-bottom:4px}
.prog-bar{height:5px;background:#e5e7eb;border-radius:3px;overflow:hidden}
.prog-fill{height:100%;border-radius:3px}

/* ── 전략 리스트 ── */
.strategy-item{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid #f0f0f0}
.strategy-item div{color:#374151}
.strategy-num{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#fff;flex-shrink:0;margin-top:1px}

/* ── 기타 ── */
.coming-soon{text-align:center;padding:80px 20px;color:#9ca3af}
.video-thumb-card{background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;cursor:pointer;transition:border-color .15s}
.video-thumb-card:hover{border-color:#2563eb}
.longform-badge{background:rgba(37,99,235,.1);color:#2563eb;border:1px solid rgba(37,99,235,.3);padding:1px 7px;border-radius:4px;font-size:10px;font-weight:600}
.shortform-badge{background:rgba(220,38,38,.1);color:#dc2626;border:1px solid rgba(220,38,38,.3);padding:1px 7px;border-radius:4px;font-size:10px;font-weight:600}
.analysis-modal{background:#f8f9fa;border:1px solid #2563eb;border-radius:12px;padding:16px;margin-top:10px}

/* ── 보고서 스타일 ── */
.report-container{background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:20px 24px;margin-top:12px}
.report-h1{font-size:18px;font-weight:700;color:#111827;border-bottom:2px solid #e5e7eb;padding-bottom:10px;margin-bottom:16px}
.report-h2{font-size:14px;font-weight:700;color:#1f2937;margin:16px 0 8px 0;display:flex;align-items:center;gap:6px}
.report-h2::before{content:"";display:inline-block;width:3px;height:14px;border-radius:2px;background:currentColor;flex-shrink:0}
.report-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #f3f4f6;font-size:12px}
.report-row:last-child{border-bottom:none}
.report-key{color:#6b7280}
.report-val{color:#111827;font-weight:600}
.report-tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin:2px}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.kpi-box{background:#f8f9fa;border:1px solid #e5e7eb;border-radius:8px;padding:10px;text-align:center}
.kpi-val{font-size:18px;font-weight:700;margin-bottom:3px}
.kpi-lbl{font-size:10px;color:#6b7280}

/* ── 저장함 ── */
.save-guideline-card{background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:0;margin-bottom:10px;overflow:hidden}
.save-guideline-header{padding:12px 14px;display:flex;align-items:center;gap:10px;cursor:pointer}
.save-thumb-strip{display:flex;gap:4px;padding:0 14px 12px 14px;flex-wrap:wrap}
.save-thumb-img{width:80px;height:45px;object-fit:cover;border-radius:4px;border:1px solid #e5e7eb}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 세션 상태
# ══════════════════════════════════════════════
_DEFAULTS = {
    "saved_items": [],
    "fnb_channel": None, "fnb_videos": None, "fnb_analysis": None, "fnb_guideline": None,
    "it_channel":  None, "it_videos":  None, "it_analysis":  None, "it_guideline":  None,
    "generated_thumb": None,
    "generated_prompt": "",    # 자동생성 영어 프롬프트 캐시
    "final_prompt_ko": "",     # 새 썸네일 한국어 설명
    "fnb_selected_video": None,
    "it_selected_video":  None,
    "fnb_thumb_analysis": None,
    "it_thumb_analysis":  None,
    "thumb_analysis_queue": None,
    "imp_auto_analysis": None,
    "fnb_report_requested": False,
    "it_report_requested":  False,
    "jump_to_improve": False,
    # YouTube URL 모드 캐시
    "_imp_last_vid_id":    "",
    "_imp_last_thumb_url": "",
    "_imp_last_title":     "",
    # 새 썸네일 프롬프트 출처 ("auto" = 자동생성 결과 보호)
    "_prompt_source":      "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════
# 분석 기준 데이터 (1,093개 롱폼 영상 기반)
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# CSV 데이터 로더 — 업로드 시 자동 집계
# ══════════════════════════════════════════════
def compute_bench(df: pd.DataFrame, domain: str) -> dict:
    """
    CSV DataFrame에서 도메인별 성공 영상 기준 벤치마크 집계.
    하드코딩 FNB/IT dict와 동일한 구조로 반환.
    """
    sub = df[(df["domain"] == domain) & (df["grade"] == "성공")].copy()
    fail = df[(df["domain"] == domain) & (df["grade"] == "실패")].copy()
    if len(sub) == 0:
        return {}

    # bool → float
    for c in ["has_person","has_text","brand_name_visible"]:
        sub[c]  = sub[c].astype(float)
        fail[c] = fail[c].astype(float)

    # 범주형 분포 (value_counts normalize)
    def dist(col):
        return (sub[col].value_counts(normalize=True)*100).round(1).to_dict()

    return {
        "has_person":  round(sub["has_person"].mean()*100, 1),
        "has_text":    round(sub["has_text"].mean()*100, 1),
        "brand":       round(sub["brand_name_visible"].mean()*100, 1),
        "brightness":  round(sub["brightness_mean"].mean(), 1),
        "saturation":  round(sub["saturation_mean"].mean(), 1),
        "contrast":    round(sub["contrast_std"].mean(), 1),
        "visual_hook": round(sub["visual_hook_level"].mean(), 2),
        "design_quality": round(sub["design_quality_level"].mean(), 2),
        "text_len":    round(sub["text_len"].mean(), 1),
        "category":    dist("thumbnail_category"),
        "color_tone":  dist("color_tone"),
        "text_size":   dist("text_size_level"),
        "person_cat":  dist("person_cat"),
        # 성공/실패 비교
        "person_success_vs_fail": {
            "성공": round(sub["has_person"].mean()*100, 1),
            "실패": round(fail["has_person"].mean()*100, 1),
        },
        # 메타
        "_n_success": len(sub),
        "_n_fail":    len(fail),
        "_n_total":   len(df[df["domain"]==domain]),
        "_from_csv":  True,
    }

FNB = {
    "has_person": 91.8, "has_text": 97.9, "brand": 84.9,
    "brightness": 146.8, "saturation": 89.7, "contrast": 70.0,
    "visual_hook": 2.61, "design_quality": 4.14, "text_len": 46.4,
    "category": {"예능/콘텐츠형":40.4,"정보 전달형":31.5,"인터뷰/인물형":10.3,
                 "브랜드 이미지형":8.2,"제품 홍보형":5.5,"리뷰/비교형":2.7},
    "color_tone": {"neutral":39.0,"warm":31.5,"cool":29.5},
    "text_size": {"large":73.3,"medium":24.0},
    "person_cat": {"2명+":60.3,"1명":31.5,"0명":8.2},
}
IT = {
    "has_person": 80.7, "has_text": 98.1, "brand": 79.1,
    "brightness": 141.5, "saturation": 83.6, "contrast": 68.6,
    "visual_hook": 2.30, "design_quality": 3.95, "text_len": 42.6,
    "category": {"정보 전달형":43.7,"예능/콘텐츠형":29.0,"인터뷰/인물형":13.1,
                 "브랜드 이미지형":9.7,"리뷰/비교형":1.9,"제품 홍보형":1.3},
    "color_tone": {"neutral":45.0,"cool":42.4,"warm":12.6},
    "text_size": {"large":69.4,"medium":26.8},
    "person_cat": {"2명+":50.7,"1명":30.0,"0명":19.3},
}

# ══════════════════════════════════════════════
# CSV 자동 로드 — 앱 시작 시 파일 경로에서 읽어 FNB/IT 덮어쓰기
# ══════════════════════════════════════════════
import os as _os

CSV_DATA_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "all_thumbnail.csv")

@st.cache_data(show_spinner=False)
def _load_bench_cache(path: str) -> tuple[dict, dict]:
    """CSV를 읽어 FnB/IT 벤치마크 집계. st.cache_data로 캐싱."""
    if not _os.path.exists(path):
        return {}, {}
    try:
        df = pd.read_csv(path)
        required = ["domain","grade","has_person","has_text","brand_name_visible",
                    "brightness_mean","saturation_mean","contrast_std",
                    "visual_hook_level","design_quality_level","text_len",
                    "thumbnail_category","color_tone","text_size_level","person_cat"]
        if any(c not in df.columns for c in required):
            return {}, {}
        return compute_bench(df, "FnB"), compute_bench(df, "IT")
    except Exception:
        return {}, {}

_csv_fnb, _csv_it = _load_bench_cache(CSV_DATA_PATH)
if _csv_fnb:
    FNB = {**FNB, **_csv_fnb}
if _csv_it:
    IT  = {**IT,  **_csv_it}

# ══════════════════════════════════════════════
# SHAP 기반 변수 영향력 데이터 (ML 분석 결과)
# ══════════════════════════════════════════════
SHAP_IT = [
    {"var": "text_size_level",             "label": "텍스트 크기",      "shap": 0.0739, "direction": "up",   "desc": "텍스트가 클수록 성공률 ↑ — 핵심 키워드를 크게"},
    {"var": "brightness_mean",             "label": "썸네일 밝기",      "shap": 0.0527, "direction": "up",   "desc": "밝은 썸네일이 어두운 것보다 성과 높음"},
    {"var": "avg_blue",                    "label": "파란색 강도",       "shap": 0.0522, "direction": "up",   "desc": "파란 계열 색상이 IT 신뢰감·전문성 전달"},
    {"var": "text_len",                    "label": "텍스트 길이",       "shap": 0.0329, "direction": "down", "desc": "텍스트가 너무 길면 오히려 성과 하락 — 간결하게"},
    {"var": "person_count",                "label": "등장 인물 수",      "shap": 0.0259, "direction": "up",   "desc": "인물 등장이 클릭율 향상에 기여"},
    {"var": "avg_green",                   "label": "초록색 강도",       "shap": 0.0191, "direction": "up",   "desc": "그린 계열 포인트 색상 효과적"},
    {"var": "visual_hook_level",           "label": "시각적 후킹",       "shap": 0.0157, "direction": "up",   "desc": "강한 시각적 후킹 요소가 클릭 유도"},
    {"var": "text_language_영어",           "label": "영어 텍스트 사용",  "shap": 0.0129, "direction": "up",   "desc": "영어 혼용이 IT 도메인에서 전문성 인식 제고"},
    {"var": "saturation_mean",             "label": "색 채도",           "shap": 0.0126, "direction": "down", "desc": "과도한 채도는 오히려 역효과 — 절제된 색감 권장"},
    {"var": "background_complexity_level", "label": "배경 복잡도",       "shap": 0.0125, "direction": "down", "desc": "배경이 복잡할수록 성과 하락 — 단순한 배경 권장"},
]

SHAP_FNB = [
    {"var": "avg_red",           "label": "붉은색 강도",   "shap": 1.4154, "direction": "up",   "desc": "붉은 색감이 식욕·감성 자극 — FnB 핵심 색상"},
    {"var": "avg_green",         "label": "초록색 강도",   "shap": 1.3168, "direction": "up",   "desc": "신선함·자연스러움 전달 — 식품 신뢰감 향상"},
    {"var": "avg_blue",          "label": "파란색 강도",   "shap": 1.1142, "direction": "up",   "desc": "색상 대비 강화 — 전체적 색감 풍부함에 기여"},
    {"var": "brightness_mean",   "label": "썸네일 밝기",   "shap": 0.9633, "direction": "up",   "desc": "밝고 선명한 음식 사진이 클릭율 결정적 요인"},
    {"var": "text_len",          "label": "텍스트 길이",   "shap": 0.6908, "direction": "down", "desc": "텍스트가 길면 음식 이미지 가림 — 간결하게"},
    {"var": "saturation_mean",   "label": "색 채도",       "shap": 0.6069, "direction": "up",   "desc": "채도 높을수록 음식이 맛있어 보임 — 선명한 색감"},
    {"var": "person_count",      "label": "등장 인물 수",  "shap": 0.5125, "direction": "up",   "desc": "인물 등장이 FnB 친근감·신뢰감 강화"},
    {"var": "contrast_std",      "label": "명암 대비",     "shap": 0.5049, "direction": "up",   "desc": "강한 명암 대비가 음식 질감 강조에 효과적"},
    {"var": "design_quality_level","label": "디자인 품질", "shap": 0.2871, "direction": "up",   "desc": "전반적 디자인 완성도가 브랜드 신뢰도에 영향"},
    {"var": "composition_style_인물 중심","label": "구도: 인물 중심","shap": 0.2511, "direction": "up", "desc": "인물 중심 구도가 FnB 스토리텔링에 효과적"},
]

# ══════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════
def fmt_num(n):
    n = int(n)
    if n >= 100_000_000: return f"{n/100_000_000:.1f}억"
    if n >= 10_000_000:  return f"{n/10_000_000:.1f}천만"
    if n >= 10_000:      return f"{n/10_000:.1f}만"
    if n >= 1_000:       return f"{n/1_000:.1f}천"
    return str(n)

def fmt_dur(sec):
    if sec == 0:   return "알 수 없음"
    if sec < 60:   return f"{sec}초"
    if sec < 3600: return f"{sec//60}분 {sec%60}초"
    return f"{sec//3600}시간 {(sec%3600)//60}분"

def parse_duration(iso):
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso or "")
    if not m: return 0
    return int(m.group(1) or 0)*3600 + int(m.group(2) or 0)*60 + int(m.group(3) or 0)


# ──────────────────────────────────────────────
# Redirect 기반 롱폼/숏폼 분류
# youtube.com/shorts/{vid} → GET → 최종 URL 확인
#   /watch 포함  → 롱폼 (Shorts로 redirect 안 됨)
#   /shorts 포함 → 숏폼
# ──────────────────────────────────────────────
_REDIRECT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def _classify_one(vid: str) -> dict:
    url = f"https://www.youtube.com/shorts/{vid}"
    try:
        resp  = requests.get(url, timeout=8, headers=_REDIRECT_HEADERS,
                             allow_redirects=True)
        final = resp.url

        # 실제 에러 상태값만 정확히 체크 (HTML 내 일반 텍스트 ERROR 오탐 방지)
        status_match = re.search(
            r'"status"\s*:\s*"(ERROR|UNPLAYABLE|LOGIN_REQUIRED|CONTENT_CHECK_REQUIRED)"',
            resp.text
        )
        has_error = bool(status_match)

        # redirect URL 우선 판정 → 그 다음 error 체크
        if "/watch" in final and not has_error:
            verdict = "longform"
        elif "/shorts" in final:
            verdict = "shorts"
        elif has_error:
            verdict = "error"
        else:
            verdict = "unknown"
    except Exception as e:
        verdict, final = "error", f"Error: {e}"
    return {"video_id": vid, "verdict": verdict, "final_url": final}

def classify_videos_redirect(video_ids: list, workers: int = 20) -> dict:
    """병렬 redirect 분류. 반환: {video_id: verdict}"""
    result = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_classify_one, vid): vid for vid in video_ids}
        for fut in as_completed(futures):
            r = fut.result()
            result[r["video_id"]] = r["verdict"]
    return result


def get_tier(s):
    if s>=1_000_000: return "Mega","#ff0000"
    if s>=100_000:   return "Macro","#ff8c42"
    if s>=10_000:    return "Mid","#3ea6ff"
    if s>=1_000:     return "Micro","#2ba640"
    return "Nano","#aaa"

def prog_html(label, val, color, mv=255):
    p = min(val/mv*100, 100)
    return (f'<div class="prog-row"><div class="prog-label-row">'
            f'<span>{label}</span><span style="color:#fff;font-weight:600">{val:.1f}</span></div>'
            f'<div class="prog-bar"><div class="prog-fill" style="width:{p:.1f}%;background:{color}"></div></div></div>')

def ring_html(val, color, label, sub):
    return (f'<div style="background:#f0f2f5;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="width:42px;height:42px;border-radius:50%;border:2.5px solid {color};'
            f'display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;'
            f'color:{color};margin:0 auto 6px">{val}</div>'
            f'<div style="font-size:11px;font-weight:600">{label}</div>'
            f'<div style="font-size:10px;color:#6b7280">{sub}</div></div>')

# ──────────────────────────────────────────────────────────────
# 가이드라인 마크다운 보고서 생성 헬퍼
# ──────────────────────────────────────────────────────────────
def build_guideline_report(ch, ana, bench, domain_label, accent, top5_data=None):
    """채널 분석 결과를 4단계 구조 보고서 HTML로 변환. top5_data: analyze_top5_thumbnails 결과"""
    import html as _html

    tier, tc = get_tier(ch["subscribers"])
    er    = ana.get("er_rate", 0)
    lf    = ana.get("longform_count", 0)
    sf    = ana.get("shortform_count", 0)
    avg_v = ana.get("avg_views", 0)
    avg_t = ana.get("avg_title", 0)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    top5_data = top5_data or {}
    stats     = top5_data.get("stats", {})
    t5results = top5_data.get("results", [])
    t5videos  = top5_data.get("videos", [])

    # ── 섹션 헤더 헬퍼 ───────────────────────────
    def sec(num, title, color):
        return (
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'margin:20px 0 10px 0;padding-bottom:8px;border-bottom:1px solid #e5e7eb">'
            f'<div style="width:22px;height:22px;border-radius:50%;background:{color};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;font-weight:700;color:#fff;flex-shrink:0">{num}</div>'
            f'<div style="font-size:13px;font-weight:700;color:#111827">{title}</div>'
            f'</div>'
        )

    def row(k, v):
        return (f'<div class="report-row">'
                f'<span class="report-key">{k}</span>'
                f'<span class="report-val">{v}</span></div>')

    # ════════════════════════════════════════════
    # 1. 채널 기본 정보
    # ════════════════════════════════════════════
    sec1 = sec("1", "채널 기본 정보", "#5ab8ff")
    kpi  = (
        f'<div class="kpi-grid" style="grid-template-columns:repeat(4,1fr)">'
        f'<div class="kpi-box"><div class="kpi-val" style="color:{tc}">{fmt_num(ch["subscribers"])}</div><div class="kpi-lbl">구독자</div></div>'
        f'<div class="kpi-box"><div class="kpi-val" style="color:#5ab8ff">{fmt_num(avg_v)}</div><div class="kpi-lbl">롱폼 평균 조회수</div></div>'
        f'<div class="kpi-box"><div class="kpi-val" style="color:#ffe033">{er:.2f}%</div><div class="kpi-lbl">참여율(ER)</div></div>'
        f'<div class="kpi-box"><div class="kpi-val" style="color:#4dd068">{lf}개</div><div class="kpi-lbl">롱폼 영상 수</div></div>'
        f'</div>'
    )
    info = (
        f'<div style="background:#f0f2f5;border-radius:8px;padding:12px;margin-bottom:4px">'
        + row("채널명", ch["name"])
        + row("채널 티어", f'<span style="color:{tc}">{tier}</span>')
        + row("롱폼 / 숏폼", f"{lf}개 / {sf}개")
        + row("평균 제목 길이", f"{avg_t:.0f}자 &nbsp;<span style='color:#666;font-size:10px'>(업종 기준 {bench['text_len']:.0f}자)</span>")
        + f'</div>'
    )

    # ════════════════════════════════════════════
    # 2. 상위 5개 썸네일 AI 분석 결과
    # ════════════════════════════════════════════
    sec2 = sec("2", f"상위 {'%d'%len(t5videos)}개 롱폼 썸네일 AI 분석 결과", "#ffe033")

    # 썸네일 스트립
    thumb_strip = ""
    for v in t5videos:
        url = v.get("thumbnail_hq") or v.get("thumbnail") or ""
        title_short = _html.escape((v.get("title") or "")[:18])
        views_str   = fmt_num(v.get("views", 0))
        if url:
            thumb_strip += (
                f'<div style="flex:1;min-width:0">'
                f'<img src="{url}" style="width:100%;aspect-ratio:16/9;object-fit:cover;'
                f'border-radius:4px;border:1px solid #e5e7eb">'
                f'<div style="font-size:9px;color:#9ca3af;margin-top:3px;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{title_short}</div>'
                f'<div style="font-size:9px;color:#9ca3af">&#128065; {views_str}</div>'
                f'</div>'
            )
    thumb_html = (
        f'<div style="display:flex;gap:6px;margin-bottom:14px">{thumb_strip}</div>'
        if thumb_strip else
        f'<div style="font-size:11px;color:#9ca3af;margin-bottom:14px">썸네일 데이터 없음</div>'
    )

    # AI 분석 통계 그리드
    def stat_pill(label, val, color="#ffe033"):
        return (
            f'<div style="background:#f3f4f6;border-radius:6px;padding:8px 12px;text-align:center">'
            f'<div style="font-size:16px;font-weight:700;color:{color}">{val}</div>'
            f'<div style="font-size:9px;color:#6b7280;margin-top:2px">{label}</div></div>'
        )

    def score_bar(label, dist, bar_color):
        high = dist.get("높음", 0)
        mid  = dist.get("보통", 0)
        low  = dist.get("낮음", 0)
        return (
            f'<div style="margin-bottom:8px">'
            f'<div style="font-size:10px;color:#6b7280;margin-bottom:3px">{label}</div>'
            f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;gap:1px">'
            f'<div style="width:{high}%;background:{bar_color};opacity:.9"></div>'
            f'<div style="width:{mid}%;background:#888;opacity:.6"></div>'
            f'<div style="width:{low}%;background:#e2e8f0"></div>'
            f'</div>'
            f'<div style="display:flex;gap:8px;font-size:9px;color:#6b7280;margin-top:2px">'
            f'<span style="color:{bar_color}">높음 {high}%</span>'
            f'<span>보통 {mid}%</span>'
            f'<span>낮음 {low}%</span>'
            f'</div></div>'
        )

    if stats:
        stat_grid = (
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:12px">'
            + stat_pill("인물 등장", f'{stats.get("has_person_pct",0)}%', "#4dd068")
            + stat_pill("2인 이상", f'{stats.get("multi_person_pct",0)}%', "#5ab8ff")
            + stat_pill("텍스트 삽입", f'{stats.get("has_text_pct",0)}%', "#ffe033")
            + stat_pill("브랜드 노출", f'{stats.get("has_brand_pct",0)}%', "#ff9944")
            + f'</div>'
        )
        score_bars = (
            f'<div style="background:#f0f2f5;border-radius:8px;padding:12px;margin-bottom:10px">'
            f'<div style="font-size:10px;color:#6b7280;margin-bottom:8px">AI 평가 점수 분포 (5개 기준)</div>'
            + score_bar("인물 구성",  stats.get("person_score",{}), "#4dd068")
            + score_bar("텍스트 가독성", stats.get("text_score",{}),   "#5ab8ff")
            + score_bar("색상/밝기",   stats.get("color_score",{}),  "#ffe033")
            + score_bar("디자인 품질", stats.get("design_score",{}), "#ff9944")
            + f'</div>'
        )
        ctr_dist = stats.get("ctr_dist", {})
        ctr_bar  = score_bar("예상 클릭율", ctr_dist, "#ff5555")

        top_str  = stats.get("top_strengths", [])
        top_iss  = stats.get("top_issues", [])
        str_html = "".join([f'<div class="good-point" style="font-size:11px"> {_html.escape(s)}</div>' for s in top_str]) or '<div class="good-point">분석 결과 없음</div>'
        iss_html = "".join([f'<div class="bad-point"  style="font-size:11px"> {_html.escape(s)}</div>' for s in top_iss])  or '<div class="bad-point">분석 결과 없음</div>'

        analysis_block = (
            f'{thumb_html}'
            f'{stat_grid}'
            f'{score_bars}'
            f'<div style="background:#f0f2f5;border-radius:8px;padding:8px 12px;margin-bottom:8px">{ctr_bar}</div>'
            f'<div style="font-size:11px;font-weight:700;color:#16a34a;margin:10px 0 6px"> 반복 강점 (공통 패턴)</div>'
            f'{str_html}'
            f'<div style="font-size:11px;font-weight:700;color:#ea580c;margin:10px 0 6px"> 반복 개선점 (공통 문제)</div>'
            f'{iss_html}'
        )
    else:
        # GCP 미설정 — 기존 텍스트 분석 fallback
        good_pts = ana.get("good") or []
        bad_pts  = ana.get("bad")  or []
        act_pts  = ana.get("act")  or []
        analysis_block = (
            f'{thumb_html}'
            f'<div style="font-size:10px;color:#6b7280;margin-bottom:10px">'
            f'AI 기반 이미지 분석 통계가 아직 생성되지 않았습니다.</div>'
            + "".join([f'<div class="good-point">; {_html.escape(p)}</div>' for p in good_pts])
            + "".join([f'<div class="bad-point">; {_html.escape(p)}</div>' for p in bad_pts])
        )

    # ════════════════════════════════════════════
    # 3. 벤치마크 기준 비교 (AI 실측값 vs 업종 기준)
    # ════════════════════════════════════════════
    sec3 = sec("3", f"{domain_label} 업종 기준 벤치마크 비교 (AI 실측)", accent)

    def cmp_bar(label, ch_val, bk_val, color="#5ab8ff", unit="%"):
        """채널 실측값 vs 벤치마크 비교 바"""
        diff     = ch_val - bk_val
        diff_str = (f'<span style="color:#4dd068">+{diff:.0f}{unit}</span>'
                    if diff >= 0 else f'<span style="color:#ff7070">{diff:.0f}{unit}</span>')
        ch_w  = min(ch_val, 100)
        bk_w  = min(bk_val, 100)
        return (
            f'<div style="margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#6b7280;margin-bottom:3px">'
            f'<span>{label}</span>'
            f'<span>채널 <b style="color:#1f2937">{ch_val:.0f}{unit}</b> &nbsp;{diff_str} vs 기준 {bk_val}{unit}</span></div>'
            f'<div style="position:relative;height:6px;background:#f3f4f6;border-radius:3px">'
            f'<div style="position:absolute;height:100%;width:{ch_w:.1f}%;background:{color};border-radius:3px;opacity:.75"></div>'
            f'<div style="position:absolute;height:140%;top:-20%;left:{bk_w:.1f}%;width:2px;background:#ffe033;border-radius:1px"></div>'
            f'</div></div>'
        )

    # AI 실측값 (stats에서 추출)
    ai_person = float(stats.get("has_person_pct", 0)) if stats else 0
    ai_multi  = float(stats.get("multi_person_pct", 0)) if stats else 0
    ai_text   = float(stats.get("has_text_pct",   0)) if stats else 0
    ai_brand  = float(stats.get("has_brand_pct",  0)) if stats else 0

    bench_block = (
        f'<div style="background:#f0f2f5;border-radius:8px;padding:14px;margin-bottom:4px">'
        f'<div style="font-size:10px;color:#6b7280;margin-bottom:10px">'
        f'<span style="color:#ffe033">━</span> 노란 선 = 업종 기준값 &nbsp;|&nbsp; 바 = AI 이미지 분석 실측값</div>'
        + (cmp_bar("인물 등장률",    ai_person, bench["has_person"], "#4dd068")
           if stats else row("인물 등장 기준", f'{bench["has_person"]}%'))
        + (cmp_bar("2인 이상 비율", ai_multi, bench["person_cat"].get("2명+",0), "#5ab8ff")
           if stats else row("인물 구성 (2명+)", f'{bench["person_cat"]["2명+"]}%'))
        + (cmp_bar("텍스트 삽입률",  ai_text,   bench["has_text"],   "#ffe033")
           if stats else row("텍스트 삽입 기준", f'{bench["has_text"]}%'))
        + (cmp_bar("브랜드 노출률",  ai_brand,  bench["brand"],      "#ff9944")
           if stats else row("브랜드 노출 기준", f'{bench["brand"]}%'))
        + f'<div style="background:#f3f4f6;border-radius:6px;padding:8px;margin-top:8px">'
        + row("🔸 권장 밝기",      f'{bench["brightness"]}')
        + row("🔸 권장 채도",      f'{bench["saturation"]}')
        + row("🔸 권장 대비",      f'{bench["contrast"]}')
        + row("🔸 디자인 품질 기준", f'{bench["design_quality"]}/5')
        + f'</div></div>'
    )

    # ════════════════════════════════════════════
    # 4. 권장 썸네일 전략
    # ════════════════════════════════════════════
    sec4 = sec("4", "권장 썸네일 전략", "#ffaa33")

    top_cats   = list(bench["category"].items())[:3]
    cat_tags   = "".join([
        f'<span class="report-tag" style="background:rgba(37,99,235,.07);'
        f'color:#1d4ed8;border:1px solid rgba(37,99,235,.3)">{k} {v}%</span>'
        for k, v in top_cats
    ])
    tone_items = list(bench["color_tone"].items())
    tone_top   = tone_items[0] if tone_items else ("—", 0)
    tone_tags  = "".join([
        f'<span class="report-tag" style="background:rgba(161,98,7,.07);'
        f'color:#92400e;border:1px solid rgba(161,98,7,.3)">{k} {v}%</span>'
        for k, v in tone_items
    ])

    # ★ REPORT_CONFIG에서 전략 항목 가져오기 (코드 상단에서 수정 가능)
    n = REPORT_CONFIG["strategy_items_count"]
    if domain_label == "FnB":
        raw_items = REPORT_CONFIG["fnb_strategy"]
        # bench 값 동적 반영 (튜플 내 포맷 문자열 처리)
        strategy_items = [
            (t, d.replace("인물 등장률·2인 이상 구성으로 시선 집중",
                          f"인물 등장 {bench['has_person']}%, 2인 이상 {bench['person_cat']['2명+']}% 목표")
              .replace("브랜드 로고 노출 비율 유지",
                       f"브랜드 로고 {bench['brand']}% 노출 유지"))
            for t, d in raw_items[:n]
        ]
    else:
        raw_items = REPORT_CONFIG["it_strategy"]
        strategy_items = [
            (t, d.replace("밝기 기준 이상, Cool/Neutral 계열",
                          f"밝기 {bench['brightness']} 이상, Cool/Neutral 계열")
              .replace("신뢰감 있는 인물 구도",
                       f"인물 등장 {bench['has_person']}% — 신뢰감 있는 구도"))
            for t, d in raw_items[:n]
        ]

    strategy_html = "".join([
        f'<div class="strategy-item">'
        f'<div class="strategy-num" style="background:{accent}">{i+1}</div>'
        f'<div><div style="font-size:12px;font-weight:600;color:#111827;margin-bottom:2px">{t}</div>'
        f'<div style="font-size:11px;color:#6b7280;line-height:1.5">{d}</div></div></div>'
        for i, (t, d) in enumerate(strategy_items)
    ])

    strategy_block = (
        f'<div style="background:#f0f2f5;border-radius:8px;padding:14px;margin-bottom:4px">'
        f'<div style="font-size:11px;color:#9ca3af;margin-bottom:6px">상위 카테고리</div>'
        f'<div style="margin-bottom:10px">{cat_tags}</div>'
        f'<div style="font-size:11px;color:#9ca3af;margin-bottom:6px">권장 색상 톤</div>'
        f'<div style="margin-bottom:12px">{tone_tags}</div>'
        f'{strategy_html}'
        f'</div>'
    )

    # ── 최종 조합 (REPORT_CONFIG 섹션 온/오프 반영) ────────────────────────────────
    _rc = REPORT_CONFIG
    _title = _rc["report_title_format"].format(channel=ch["name"], domain=domain_label)

    _sec1_html = (f"{sec1}{kpi if _rc['show_kpi_grid'] else ''}{info if _rc['show_tier'] else ''}"
                  if _rc["show_channel_info"] else "")
    _sec2_html = (f"{sec2}{analysis_block}" if _rc["show_thumbnail_analysis"] else "")
    _sec3_html = (f"{sec3}{bench_block}"    if _rc["show_benchmark"] else "")
    _sec4_html = (f"{sec4}{strategy_block}" if _rc["show_strategy"] else "")

    html = f"""
<div class="report-container">
  <div class="report-h1">{_title}</div>
  <div style="font-size:11px;color:#9ca3af;margin-bottom:16px">생성: {now} &nbsp;|&nbsp; 최근 50개 영상 분석 &nbsp;|&nbsp; 롱폼 {lf}개 · 숏폼 {sf}개</div>

  {_sec1_html}
  {_sec2_html}
  {_sec3_html}
  {_sec4_html}
</div>"""
    return html

def build_guideline_markdown(ch, ana, bench, domain_label, top5_data=None):
    """저장/다운로드용 순수 마크다운 텍스트 (4단계 구조)"""
    top5_data = top5_data or {}
    stats     = top5_data.get("stats", {})
    t5videos  = top5_data.get("videos", [])
    tier, _ = get_tier(ch["subscribers"])
    er    = ana.get("er_rate", 0)
    lf    = ana.get("longform_count", 0)
    sf    = ana.get("shortform_count", 0)
    avg_v = ana.get("avg_views", 0)
    avg_t = ana.get("avg_title", 0)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 상위 5개 롱폼 영상 목록
    lf_vids = ana.get("longform_videos") or ana.get("top3") or []
    top5    = sorted(lf_vids, key=lambda x: x.get("views", 0), reverse=True)[:5]
    thumb_lines = []
    for i, v in enumerate(top5):
        thumb_lines.append(f"{i+1}. [{v.get('title','')[:40]}] 조회수: {fmt_num(v.get('views',0))}")

    # 도메인별 전략
    if domain_label == "FnB":
        strategy_lines = [
            "1. 색감 우선 — 붉고 선명한 색감 유지 (avg_red/green 높게)",
            "2. 인물 중심 구도 — 2인 이상 구성으로 시선 집중",
            "3. 간결한 텍스트 — 음식 이미지가 주인공",
            f"4. 브랜드 노출 — {bench['brand']}% 노출 목표",
        ]
    else:
        strategy_lines = [
            "1. 텍스트 크기 최우선 — 핵심 키워드 크고 명확하게 (large)",
            f"2. 밝고 선명하게 — 밝기 {bench['brightness']} 이상, Cool/Neutral 계열",
            f"3. 전문가 인물 활용 — 인물 등장 {bench['has_person']}% 목표",
            "4. 배경 단순화 — 핵심 메시지 집중",
        ]

    lines = [
        f"# {ch['name']} — {domain_label} 썸네일 전략 보고서",
        f"> 생성: {now} | 최근 20개 영상 분석 | 롱폼 {lf}개 · 숏폼 {sf}개",
        "",
        "---",
        "## 1. 채널 기본 정보",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| 채널명 | {ch['name']} |",
        f"| 채널 티어 | {tier} |",
        f"| 구독자 | {fmt_num(ch['subscribers'])} |",
        f"| 롱폼 평균 조회수 | {fmt_num(avg_v)} |",
        f"| 참여율(ER) | {er:.2f}% |",
        f"| 롱폼 / 숏폼 | {lf}개 / {sf}개 |",
        f"| 평균 제목 길이 | {avg_t:.0f}자 (기준 {bench['text_len']:.0f}자) |",
        "",
        "---",
        "## 2. 최근 롱폼 썸네일 분석 (상위 최대 5개)",
    ] + (thumb_lines or ["- 데이터 없음"]) + [
        "",
        "### ✅ 잘하고 있는 점",
    ] + [f"- {p}" for p in (ana.get("good") or ["데이터 없음"])] + [
        "",
        "### ⚠️ 개선이 필요한 점",
    ] + [f"- {p}" for p in (ana.get("bad") or ["데이터 없음"])] + [
        "",
        "### 🚀 실행 권장 사항",
    ] + [f"{i+1}. {a}" for i, a in enumerate(ana.get("act") or [])] + [
        "",
        "---",
        f"## 3. {domain_label} 업종 성공 기준 벤치마크",
        f"| 지표 | 기준값 |",
        f"|------|--------|",
        f"| 인물 등장률 | {bench['has_person']}% |",
        f"| 2인 이상 구성 | {bench['person_cat']['2명+']}% |",
        f"| 텍스트 삽입률 | {bench['has_text']}% |",
        f"| 브랜드 노출률 | {bench['brand']}% |",
        f"| 권장 밝기 | {bench['brightness']} |",
        f"| 권장 채도 | {bench['saturation']} |",
        f"| 디자인 품질 | {bench['design_quality']}/5 |",
        "",
        "---",
        "## 4. 권장 썸네일 전략",
    ] + strategy_lines

    return "\n".join(lines)

# ══════════════════════════════════════════════
# YouTube API  ★ Quota 최적화 버전 ★
# ──────────────────────────────────────────────
# API별 quota 비용 (단위: units)
#   search                    : 100 units  ← 가장 비쌈
#   channels / videos /
#   playlistItems             :   1 unit
#
# 최적화 요점
#   1. 채널 검색: @handle·UC ID면 search 완전 생략 (channels API = 1 unit)
#      일반 키워드도 search 1회로 제한 (기존 최대 4회 → 1회)
#   2. 영상 목록: search(100) → playlistItems(1)로 교체
#      channels API의 contentDetails.relatedPlaylists.uploads 활용
#   3. channels API 호출 시 contentDetails도 같이 받아
#      uploads playlist ID를 별도 호출 없이 확보
#   4. 썸네일·제목 등 snippet 정보는 playlistItems에서 바로 추출
#      → videos API는 statistics + contentDetails(duration)만 조회
# ══════════════════════════════════════════════

def _yt_get(url, params, timeout=10):
    """YouTube API GET + 에러 처리 공통 헬퍼"""
    r = requests.get(url, params=params, timeout=timeout)
    d = r.json()
    if "error" in d:
        raise Exception(d["error"]["message"])
    return d

def _norm(s):
    return re.sub(r"\s+", "", str(s).lower())

def _score_channel(ch, q_norm, candidate_rank):
    """채널 공식성 점수 계산"""
    sn      = ch.get("snippet", {})
    title   = sn.get("title", "")
    desc    = sn.get("description", "")
    handle  = sn.get("customUrl", "")
    cid     = ch.get("id", "")
    subs    = int(ch.get("statistics", {}).get("subscriberCount", 0) or 0)

    title_n  = _norm(title)
    handle_n = _norm(handle)
    all_n    = _norm(f"{title} {handle} {desc}")

    stock_words    = ["주식","투자","증권","부자","백만장자","급등","차트","재테크","stock","invest"]
    official_words = ["공식","official","오피셜","korea","한국","뉴스룸","newsroom","brand"]

    alias_map = {
        "삼성전자": ["samsung","samsungelectronics","삼성전자","삼성"],
        "삼성":    ["samsung","samsungelectronics","삼성"],
        "엘지":    ["lg","lge","lg전자","엘지"],
        "lg전자":  ["lg","lge","lg전자","엘지"],
    }
    aliases = alias_map.get(q_norm, [q_norm])

    rank  = candidate_rank.get(cid, 99)
    score = max(0, 120 - rank * 8)

    if any(a and a == title_n  for a in aliases): score += 700
    if any(a and a in title_n  for a in aliases): score += 420
    if any(a and a in handle_n for a in aliases): score += 360
    if any(a and a in all_n    for a in aliases): score += 180

    if any(w in title_n or w in handle_n for w in official_words): score += 260
    elif any(w in all_n for w in official_words):                   score += 120

    if any(w in title_n for w in stock_words):  score -= 1200
    elif any(w in all_n for w in stock_words):  score -= 700

    score += min(subs / 100_000, 40)
    return score

def _parse_channel_item(item) -> dict:
    """
    channels API 응답 item → 앱에서 사용하는 채널 dict.
    contentDetails가 있으면 uploads playlist ID도 포함.
    quota: 0 (호출 없음, 파싱만)
    """
    sn    = item.get("snippet", {})
    stats = item.get("statistics", {})
    cd    = item.get("contentDetails", {})
    uploads_pid = cd.get("relatedPlaylists", {}).get("uploads", "")
    return {
        "id":          item["id"],
        "name":        sn.get("title", ""),
        "description": sn.get("description", "")[:150],
        "subscribers": int(stats.get("subscriberCount", 0) or 0),
        "views":       int(stats.get("viewCount",       0) or 0),
        "video_count": int(stats.get("videoCount",      0) or 0),
        "uploads_playlist": uploads_pid,   # ★ 영상 목록용 playlist ID
        "avatar": (
            sn.get("thumbnails", {})
              .get("high", sn.get("thumbnails", {}).get("default", {}))
              .get("url", "")
        ),
    }

# ──────────────────────────────────────────────
# 채널 정보 조회  (quota: 1 unit)
# ──────────────────────────────────────────────
def yt_channel_info(cid, key) -> dict:
    """
    채널 ID → 채널 정보 dict.
    contentDetails를 같이 받아 uploads playlist ID를 한 번에 확보.
    quota: channels×1 = 1 unit
    """
    d = _yt_get(
        "https://www.googleapis.com/youtube/v3/channels",
        {"part": "snippet,statistics,contentDetails", "id": cid, "key": key},
    )
    if not d.get("items"):
        raise Exception("채널 정보를 찾을 수 없습니다")
    return _parse_channel_item(d["items"][0])

# ──────────────────────────────────────────────
# 채널 후보 검색  (quota: 최대 101 units → 개선 후 최대 2 units)
# ──────────────────────────────────────────────
def yt_search_channel_candidates(q, key, limit=8) -> list[dict]:
    """
    브랜드 채널 후보 목록 반환.

    quota 최적화:
      • UC…  → channels API 1회 = 1 unit  (search 없음)
      • @handle → channels?forHandle 1회 = 1 unit  (search 없음)
      • 일반 키워드 → search 1회(100) + channels 1회(1) = 101 units
        (기존: search 최대 4회 = 400 units)
    """
    query = q.strip()

    # ── ① 채널 ID 직접 입력 (UC...) ──
    if query.startswith("UC"):
        return [yt_channel_info(query, key)]

    # ── ② @handle 입력 → search 없이 channels API 직행 ──
    if query.startswith("@"):
        d = _yt_get(
            "https://www.googleapis.com/youtube/v3/channels",
            {"part": "snippet,statistics,contentDetails",
             "forHandle": query, "key": key},
        )
        if d.get("items"):
            return [_parse_channel_item(d["items"][0])]
        raise Exception(f"핸들 '{query}'에 해당하는 채널을 찾을 수 없습니다")

    # ── ③ 일반 키워드 → search 1회만 (기존 최대 4회) ──
    d_search = _yt_get(
        "https://www.googleapis.com/youtube/v3/search",
        {"part": "snippet", "type": "channel", "q": query,
         "maxResults": min(limit, 10), "key": key},
    )
    items = d_search.get("items", [])
    if not items:
        raise Exception("채널을 찾을 수 없습니다")

    # search snippet에서 채널 ID 수집 (중복 제거)
    seen, cids = set(), []
    for item in items:
        cid = item["snippet"]["channelId"]
        if cid not in seen:
            seen.add(cid)
            cids.append(cid)
        if len(cids) >= limit:
            break

    # channels API 1회로 전체 정보 + uploads playlist ID 한꺼번에 조회
    d_ch = _yt_get(
        "https://www.googleapis.com/youtube/v3/channels",
        {"part": "snippet,statistics,contentDetails",
         "id": ",".join(cids), "key": key},
    )
    return [_parse_channel_item(it) for it in d_ch.get("items", [])]


def yt_find_channel(q, key) -> str:
    """
    단일 최적 채널 ID 반환 (자동 선택).
    후보 목록에서 공식성 점수가 가장 높은 채널을 선택.
    quota: yt_search_channel_candidates()와 동일
    """
    query  = q.strip()
    q_norm = re.sub(r"\s+", "", query.lower().lstrip("@"))

    candidates = yt_search_channel_candidates(query, key, limit=8)
    if len(candidates) == 1:
        return candidates[0]["id"]

    # search 결과 순위를 보존하기 위해 인덱스로 rank 부여
    rank_map = {ch["id"]: i for i, ch in enumerate(candidates)}
    # channels API raw item이 필요해 재조회 (점수 계산용)
    d_ch = _yt_get(
        "https://www.googleapis.com/youtube/v3/channels",
        {"part": "snippet,statistics",
         "id": ",".join(ch["id"] for ch in candidates), "key": key},
    )
    raw_items = d_ch.get("items", [])
    best = max(raw_items, key=lambda it: _score_channel(it, q_norm, rank_map))
    return best["id"]


def render_channel_candidate_picker(prefix, candidates):
    """검색 후보를 보여주고 선택된 채널 ID를 반환한다."""
    if not candidates:
        return None

    def label(i):
        ch = candidates[i]
        return f"{ch['name']} · 구독자 {fmt_num(ch['subscribers'])} · 영상 {fmt_num(ch['video_count'])}개"

    idx = st.selectbox(
        "공식 채널을 선택하세요",
        range(len(candidates)),
        format_func=label,
        key=f"{prefix}_candidate_select",
    )
    ch = candidates[idx]
    st.markdown(
        f'<div class="guide-hint">선택됨: <b>{ch["name"]}</b><br>'
        f'<span style="color:#888">{ch["description"][:120]}</span></div>',
        unsafe_allow_html=True,
    )
    return ch["id"]

# ──────────────────────────────────────────────
# 영상 목록 수집  (quota: 100 units → 개선 후 2 units)
# ──────────────────────────────────────────────
def yt_fetch_videos(cid, key, max_results=50, uploads_playlist: str = "") -> list[dict]:
    """
    채널의 최근 영상 목록 + 통계 수집.

    quota 최적화:
      search(channelId) 100 units → playlistItems 1 unit 으로 교체.
      채널 info에서 uploads playlist ID를 미리 받아 전달하면 추가 조회 없음.
      playlist ID 없으면 channels API 1회로 확보 (1 unit).

      총 비용: playlistItems(1) + videos(1) = 2 units  (기존 101 units)
    """
    BASE = "https://www.googleapis.com/youtube/v3"

    # ── uploads playlist ID 확보 ──
    pid = uploads_playlist
    if not pid:
        d_ch = _yt_get(f"{BASE}/channels",
            {"part": "contentDetails", "id": cid, "key": key})
        items_ch = d_ch.get("items", [])
        if not items_ch:
            return []
        pid = items_ch[0]["contentDetails"]["relatedPlaylists"].get("uploads", "")
    if not pid:
        return []

    # ── playlistItems로 영상 ID + snippet(제목·날짜·썸네일) 수집 ──
    # pageToken으로 페이지네이션 (한 페이지 최대 50개)
    video_snippets: dict[str, dict] = {}
    next_token = None
    fetched = 0
    while fetched < max_results:
        batch = min(50, max_results - fetched)
        params = {
            "part":       "snippet",
            "playlistId": pid,
            "maxResults": batch,
            "key":        key,
        }
        if next_token:
            params["pageToken"] = next_token
        d_pl = _yt_get(f"{BASE}/playlistItems", params)
        for it in d_pl.get("items", []):
            sn  = it.get("snippet", {})
            rid = sn.get("resourceId", {})
            vid = rid.get("videoId", "")
            if not vid or vid in video_snippets:
                continue
            th = sn.get("thumbnails", {})
            # playlistItems snippet에는 maxres가 없을 수 있음 → hq 우선
            best_th = (
                th.get("maxres", th.get("high", th.get("medium", th.get("default", {}))))
                  .get("url", "")
            )
            # hqdefault URL 직접 조합 (항상 존재, 더 확실함)
            hq_url = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
            video_snippets[vid] = {
                "title":     sn.get("title", ""),
                "published": (sn.get("publishedAt") or "")[:10],
                "thumbnail": best_th or hq_url,
                "thumbnail_hq": hq_url,   # ★ 고화질 썸네일 변수에 저장
            }
            fetched += 1
            if fetched >= max_results:
                break
        next_token = d_pl.get("nextPageToken")
        if not next_token:
            break

    if not video_snippets:
        return []

    # ── videos API로 statistics + contentDetails(duration) 일괄 조회 ──
    # 50개 단위로 분할 요청 (API 제한)
    vids_list = list(video_snippets.keys())
    stats_map: dict[str, dict] = {}
    for i in range(0, len(vids_list), 50):
        chunk = vids_list[i:i+50]
        d_v = _yt_get(f"{BASE}/videos",
            {"part": "statistics,contentDetails",
             "id":   ",".join(chunk), "key": key})
        for v in d_v.get("items", []):
            dur_sec = parse_duration(v["contentDetails"].get("duration", ""))
            stats_map[v["id"]] = {
                "views":    int(v["statistics"].get("viewCount",    0) or 0),
                "likes":    int(v["statistics"].get("likeCount",    0) or 0),
                "comments": int(v["statistics"].get("commentCount", 0) or 0),
                "duration": dur_sec,
            }

    # ── 최종 결과 조합 ──
    result = []
    for vid, sn in video_snippets.items():
        st_ = stats_map.get(vid, {})
        dur = st_.get("duration", 0)
        result.append({
            "id":           vid,
            "title":        sn["title"],
            "published":    sn["published"],
            "thumbnail":    sn["thumbnail"],
            "thumbnail_hq": sn["thumbnail_hq"],   # ★ 변수로 저장해 재사용
            "views":        st_.get("views",    0),
            "likes":        st_.get("likes",    0),
            "comments":     st_.get("comments", 0),
            "duration":     dur,
            "duration_fmt": fmt_dur(dur),
            "verdict":      "unknown",
            "is_longform":  False,
            "is_shortform": False,
        })
    return result

# ══════════════════════════════════════════════
# 채널 분석 로직 (롱폼 기준)
# ══════════════════════════════════════════════
def analyze_channel(videos, bench, domain_name):
    """롱폼만 필터링해서 분석.
    우선순위: ① redirect verdict='longform' ② verdict='unknown'이면 duration>=60 ③ 전체 fallback
    """
    # ① redirect로 명확히 longform 판정된 것
    lf_redirect = [v for v in videos if v.get("verdict") == "longform"]
    # ② verdict=unknown이면 duration으로 보조 판단 (60초 이상)
    lf_dur      = [v for v in videos if v.get("verdict") == "unknown" and v.get("duration",0) >= 60]
    # ③ verdict 자체가 없으면(분류 미실행) duration으로만 판단
    lf_no_verd  = [v for v in videos if "verdict" not in v and v.get("duration",0) >= 60]

    lf = lf_redirect or (lf_redirect + lf_dur) or lf_no_verd
    # shorts/error만 있거나 아무것도 없으면 전체 fallback
    if not lf:
        lf = videos

    views    = [v["views"]    for v in lf]
    likes    = [v["likes"]    for v in lf]
    comments = [v["comments"] for v in lf]

    avg_views    = float(np.mean(views))    if views    else 0
    avg_likes    = float(np.mean(likes))    if likes    else 0
    avg_comments = float(np.mean(comments)) if comments else 0
    er_rate      = (avg_likes + avg_comments) / max(avg_views, 1) * 100
    avg_title    = float(np.mean([len(v["title"]) for v in lf]))

    sorted_v = sorted(lf, key=lambda x: x["views"], reverse=True)
    top3 = sorted_v[:3]
    bot3 = sorted_v[-3:]

    good, bad, act = [], [], []

    # 참여율 언급 제외 (썸네일 전략 보고서는 시각적 요소에만 집중)

    # 제목 길이
    opt = bench["text_len"]
    if abs(avg_title - opt) <= 15:
        good.append(f"평균 제목 길이({avg_title:.0f}자)가 성공 기준({opt:.0f}자)에 근접합니다")
    elif avg_title > opt + 15:
        bad.append(f"제목 평균 {avg_title:.0f}자로 다소 깁니다 (롱폼 기준: {opt:.0f}자)")
        act.append("썸네일 텍스트는 핵심 키워드 위주로 간결하게 구성하세요")
    else:
        bad.append(f"제목 평균 {avg_title:.0f}자로 짧습니다. 핵심 키워드를 더 활용하세요")

    # 조회수 편차
    if len(views) >= 3:
        cv = float(np.std(views)) / max(float(np.mean(views)), 1)
        if cv > 1.0:
            bad.append("롱폼 영상별 조회수 편차가 매우 큽니다. 성과 패턴 분석이 필요합니다")
            act.append("상위 조회수 롱폼 영상의 썸네일 구성 요소를 반복 활용하는 전략을 권장합니다")
        else:
            good.append("롱폼 영상별 조회수가 비교적 안정적입니다. 일관된 콘텐츠 품질을 유지하고 있습니다")

    # 롱폼/숏폼 비율
    n_lf = len([v for v in videos if v.get("verdict") == "longform"])
    n_sf = len([v for v in videos if v.get("verdict") == "shorts"])
    if n_lf > 0:
        good.append(f"분석 기간 내 롱폼 {n_lf}개 · 숏폼 {n_sf}개 업로드 확인")

    # 도메인별 추천
    if domain_name == "FnB":
        act.append(f"2인 이상 인물 구성을 활용하세요 (FnB 롱폼 성공 영상의 {bench['person_cat']['2명+']:.0f}% 적용)")
        act.append("Warm 또는 Neutral 색상 톤을 우선 적용하세요 (식욕·친근감 자극)")
    else:
        act.append("핵심 정보를 명확히 전달하는 텍스트 중심 구성을 강화하세요")
        act.append("Neutral 또는 Cool 색상 톤으로 전문성·신뢰감을 표현하세요")

    return {
        "longform_count":  n_lf,
        "shortform_count": n_sf,
        "avg_views": avg_views, "avg_likes": avg_likes,
        "avg_comments": avg_comments, "er_rate": er_rate,
        "avg_title": avg_title,
        "top3": top3, "bot3": bot3,
        "good": good, "bad": bad, "act": act,
        "longform_videos": lf,
    }

# ══════════════════════════════════════════════
# Gemini API (무료 모델)
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# Pydantic 응답 스키마 정의
# ──────────────────────────────────────────────
class PromptGenResponse(BaseModel):
    """프롬프트 자동 생성 응답 스키마"""
    prompt_en: str = Field(description="영어 이미지 생성 프롬프트 (상세하고 구체적)")
    prompt_ko: str = Field(description="한국어 요약 설명 (1문장)")

class ThumbnailElement(BaseModel):
    """썸네일 구성 요소"""
    main_objects: str        = Field(description="주요 시각 요소")
    text_content: str        = Field(description="텍스트 내용")
    color_feature: str       = Field(description="색상 특징")
    person_composition: str  = Field(description="인물 구성")

class ThumbnailEvaluation(BaseModel):
    """업종 기준 대비 평가"""
    strengths:     str = Field(description="강점")
    weaknesses:    str = Field(description="약점")
    estimated_ctr: str = Field(description="예상 클릭율")

class ThumbnailAnalysisResponse(BaseModel):
    """썸네일 분석 전체 응답 스키마"""
    elements:     ThumbnailElement    = Field(description="썸네일 구성 요소 분석")
    evaluation:   ThumbnailEvaluation = Field(description="업종 기준 대비 평가")
    improvements: list[str]           = Field(description="개선 제안 3가지")

def _to_vertex_schema(model_cls) -> dict:
    """
    Pydantic v2 model_json_schema() → Vertex AI response_schema 용 dict 변환.

    Vertex AI는 두 가지를 허용하지 않음:
      1. 'title' 필드
      2. '$defs' + '$ref' 참조 구조 (중첩 모델에서 자동 생성됨)

    → $ref 를 모두 인라인으로 치환하고 title 을 제거해서 flat dict 반환.
    """
    import copy

    schema = copy.deepcopy(model_cls.model_json_schema())
    defs   = schema.pop("$defs", {})

    def _resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                # "#/$defs/ClassName" → defs["ClassName"] 로 치환
                ref_name = obj["$ref"].split("/")[-1]
                return _resolve(copy.deepcopy(defs.get(ref_name, {})))
            return {k: _resolve(v) for k, v in obj.items() if k != "title"}
        elif isinstance(obj, list):
            return [_resolve(i) for i in obj]
        return obj

    return _resolve(schema)

# 미리 변환해두기 (매 호출마다 재계산 방지)
_PROMPT_SCHEMA   = _to_vertex_schema(PromptGenResponse)
_ANALYSIS_SCHEMA = _to_vertex_schema(ThumbnailAnalysisResponse)

# ──────────────────────────────────────────────
# Vertex AI 클라이언트 싱글턴
# ──────────────────────────────────────────────
_vertex_initialized: dict = {}   # {project_id: True}

_SUPPORTED_LOCATIONS = frozenset({
    'us-central1','us-east1','us-east4','us-east5','us-south1',
    'us-west1','us-west2','us-west3','us-west4',
    'asia-east1','asia-east2','asia-northeast1','asia-northeast2','asia-northeast3',
    'asia-south1','asia-southeast1','asia-southeast2',
    'europe-west1','europe-west2','europe-west3','europe-west4',
    'europe-west6','europe-west8','europe-west9','europe-west12',
    'australia-southeast1','australia-southeast2',
    'northamerica-northeast1','northamerica-northeast2',
    'southamerica-east1','southamerica-west1',
    'me-central1','me-central2','me-west1',
    'africa-south1','europe-central2','europe-north1',
    'europe-southwest1','europe-west12',
})

GEMINI_PROMPT_MODEL = st.secrets.get(
    "GEMINI_PROMPT_MODEL",
    "gemini-2.5-flash-lite",   # gemini-3.1-flash-lite-preview 는 allowlist 필요
)

GEMINI_VISION_MODEL = st.secrets.get(
    "GEMINI_VISION_MODEL",
    "gemini-2.5-flash",
)

IMAGEN_MODEL = st.secrets.get(
    "IMAGEN_MODEL",
    "imagen-4.0-generate-001",
)

# ══════════════════════════════════════════════
# ★ 보고서 설정 (여기서 쉽게 수정 가능) ★
# ══════════════════════════════════════════════
REPORT_CONFIG = {
    # ── 보고서에 포함할 섹션 (True/False로 온/오프) ──
    "show_channel_info":    True,   # 섹션1: 채널 기본 정보 (구독자, 조회수, ER 등)
    "show_thumbnail_analysis": True, # 섹션2: 상위 5개 썸네일 AI 분석
    "show_benchmark":       True,   # 섹션3: 업종 기준 벤치마크 비교
    "show_strategy":        True,   # 섹션4: 권장 썸네일 전략

    # ── 섹션별 세부 항목 ──
    "show_kpi_grid":        True,   # 구독자/조회수/ER/롱폼수 KPI 박스
    "show_tier":            True,   # 채널 티어 표시 (Mega/Macro/Mid/Micro/Nano)
    "show_score_bars":      True,   # AI 평가 점수 분포 바 차트
    "show_ctr_dist":        True,   # 예상 클릭율 분포
    "show_strengths":       True,   # 반복 강점 (공통 패턴)
    "show_issues":          True,   # 반복 개선점 (공통 문제)
    "show_bench_bars":      True,   # 업종 기준 비교 바
    "show_bench_color":     True,   # 권장 밝기/채도/대비/디자인 품질

    # ── 분석 영상 수 ──
    "top_n_videos":         5,      # 상위 몇 개 영상 썸네일 분석할지 (1~10)

    # ── 전략 섹션 항목 수 ──
    "strategy_items_count": 4,      # 권장 전략 항목 수 (최대 4)

    # ── 보고서 타이틀 포맷 ──
    # {channel}: 채널명, {domain}: FnB/IT
    "report_title_format":  "📋 {channel} — {domain} 썸네일 전략 보고서",

    # ── FnB 도메인 전략 항목 (순서대로 최대 strategy_items_count개 사용) ──
    "fnb_strategy": [
        ("색감 우선",      "붉고 선명한 색감 — avg_red/green 수치 높게 유지"),
        ("인물 중심 구도", "인물 등장률·2인 이상 구성으로 시선 집중"),
        ("간결한 텍스트",  "텍스트는 최소화 — 음식 이미지가 주인공"),
        ("브랜드 노출",    "브랜드 로고 노출 비율 유지"),
    ],

    # ── IT 도메인 전략 항목 ──
    "it_strategy": [
        ("텍스트 크기 최우선", "크고 명확한 핵심 키워드 — text_size_level: large"),
        ("밝고 선명하게",      "밝기 기준 이상, Cool/Neutral 계열"),
        ("전문가 인물 활용",   "신뢰감 있는 인물 구도"),
        ("배경 단순화",        "배경 복잡도 낮게 — 핵심 메시지에 집중"),
    ],
}

# ══════════════════════════════════════════════
# ★ 이미지 생성 성능 설정 (여기서 쉽게 수정 가능) ★
# ══════════════════════════════════════════════
IMAGE_GEN_CONFIG = {
    # ── Gemini 텍스트 생성 ──
    "prompt_temperature":       0.6,    # 프롬프트 생성 창의성 (0.0~1.0, 높을수록 창의적)
    "prompt_max_tokens":        700,    # 프롬프트 생성 최대 토큰
    "translate_temperature":    0.1,    # 번역 temperature (낮을수록 정확)
    "translate_max_tokens":     300,    # 번역 최대 토큰

    # ── Gemini Vision 분석 ──
    "vision_temperature":       0.1,    # 비전 분석 temperature
    "vision_max_tokens":        2048,   # 비전 분석 최대 토큰
    "style_desc_max_tokens":    400,    # 스타일 묘사 최대 토큰 (개선 썸네일용)

    # ── Imagen 이미지 생성 ──
    "imagen_aspect_ratio":      "16:9",         # 생성 비율 (16:9 / 1:1 / 4:3)
    "imagen_safety_filter":     "block_few",    # 안전 필터 수준 (block_few / block_some / block_most)
    "imagen_person_generation": "allow_adult",  # 인물 생성 허용 (allow_adult / dont_allow)
    "imagen_sample_count":      1,              # 한 번에 생성할 이미지 수 (1~4)

    # ── 개선 썸네일 스타일 묘사 길이 ──
    "style_desc_trim":          250,    # 스타일 묘사 프롬프트에서 사용할 최대 문자 수
}

def init_vertex(project_id: str, location: str = "us-central1"):
    """Vertex AI 초기화 (중복 방지 + region 검증)"""
    # 잘못된 region 입력 시 us-central1로 폴백
    if location not in _SUPPORTED_LOCATIONS:
        location = "us-central1"
    key = f"{project_id}:{location}"
    if key not in _vertex_initialized:
        vertexai.init(project=project_id, location=location)
        _vertex_initialized[key] = True

def imagen_generate_bytes(
    prompt: str,
    project_id: str,
    location: str,
    model: str,
) -> bytes:
    """Vertex AI Imagen REST API로 이미지 생성. GCP 프로젝트 과금/무료 크레딧 경로를 사용한다."""
    import google.auth
    from google.auth.transport.requests import Request

    if location == "global":
        location = "us-central1"

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())

    endpoint = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/publishers/google/models/{model}:predict"
    )
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount":        IMAGE_GEN_CONFIG["imagen_sample_count"],
            "aspectRatio":        IMAGE_GEN_CONFIG["imagen_aspect_ratio"],
            "safetyFilterLevel":  IMAGE_GEN_CONFIG["imagen_safety_filter"],
            "personGeneration":   IMAGE_GEN_CONFIG["imagen_person_generation"],
        },
    }
    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if not resp.ok:
        raise Exception(f"Imagen API 오류 {resp.status_code}: {resp.text}")

    data = resp.json()
    predictions = data.get("predictions", [])
    if not predictions:
        raise Exception("Imagen API 응답에 이미지가 없습니다")

    pred = predictions[0]
    image_b64 = (
        pred.get("bytesBase64Encoded")
        or pred.get("image", {}).get("bytesBase64Encoded")
    )
    if not image_b64:
        raise Exception("Imagen API 응답에서 이미지 데이터를 찾을 수 없습니다")
    return base64.b64decode(image_b64)
# ──────────────────────────────────────────────
# 1. 텍스트 생성 (gemini-2.5-flash on Vertex AI)
# ──────────────────────────────────────────────
def gemini_generate_text(
    prompt: str,
    project_id: str,
    location: str = "us-central1",
    model: str = GEMINI_PROMPT_MODEL,
    temperature: float = 0.7,
) -> str:
    init_vertex(project_id, location)
    mdl = GenerativeModel(model)
    resp = mdl.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=temperature, max_output_tokens=IMAGE_GEN_CONFIG["vision_max_tokens"]),
    )
    return resp.text.strip()

# ──────────────────────────────────────────────
# 2. 비전 분석 (gemini-2.5-flash on Vertex AI)
# ──────────────────────────────────────────────
def gemini_analyze_image(
    img_url_or_b64: str,
    prompt: str,
    project_id: str,
    location: str = "us-central1",
    is_b64: bool = False,
    model: str = GEMINI_VISION_MODEL,
) -> str:
    init_vertex(project_id, location)

    if is_b64:
        img_bytes = base64.b64decode(img_url_or_b64)
    else:
        resp = requests.get(img_url_or_b64, timeout=10)
        img_bytes = resp.content

    img_part = Part.from_data(data=img_bytes, mime_type="image/jpeg")
    mdl = GenerativeModel(model)
    resp = mdl.generate_content(
        [img_part, prompt],
        generation_config=GenerationConfig(temperature=IMAGE_GEN_CONFIG["vision_temperature"], max_output_tokens=IMAGE_GEN_CONFIG["vision_max_tokens"]),
    )
    return resp.text.strip()

# ──────────────────────────────────────────────
# 3. 프롬프트 자동 생성 — Pydantic 구조화 출력
# ──────────────────────────────────────────────
def gemini_gen_prompt(
    keywords: str,
    domain: str,
    category: str,
    color_tone: str,
    ch_info: Optional[dict],
    project_id: str,
    location: str = "us-central1",
) -> tuple[str, str]:
    """
    Vertex AI + Pydantic 스키마로 구조화된 프롬프트 생성.
    반환: (영어 프롬프트, 한국어 요약)
    """
    ch_ctx = f"채널명: {ch_info['name']}, 구독자: {fmt_num(ch_info['subscribers'])}" if ch_info else ""

    # 카테고리별 필수 시각 요소
    category_rules = {
        "예능/콘텐츠형": (
            "MUST include 1-2 people with exaggerated surprised or excited facial expressions, "
            "open mouth, wide eyes, dynamic body language. People must be the DOMINANT subject. "
            "High energy, vibrant, fun atmosphere."
        ),
        "정보 전달형": (
            "Clean product or subject as the hero shot, takes up 60% of frame. "
            "Simple uncluttered background. Bold, clear visual hierarchy. "
            "Professional and trustworthy feel."
        ),
        "인터뷰/인물형": (
            "MUST include a person's face as the main subject. "
            "Face should be large and clearly visible, direct gaze or thoughtful expression. "
            "Interview-style framing, confident posture."
        ),
        "브랜드 이미지형": (
            "Premium brand aesthetic. Clean, polished, aspirational. "
            "Brand colors dominant. Minimal but impactful composition."
        ),
        "리뷰/비교형": (
            "Products or items clearly displayed side-by-side or held by a person. "
            "Clear visual comparison. Person reacting to products preferred."
        ),
        "제품 홍보형": (
            "Product as absolute hero, ultra-sharp focus, clean solid-color background. "
            "Dynamic lighting to highlight product texture and appeal."
        ),
    }

    # 도메인별 비주얼 스타일
    domain_style = {
        "FnB": (
            "Korean food and beverage brand. "
            "Warm, rich, SATURATED colors (red, orange, golden). "
            "Food looks absolutely delicious and fresh. "
            "Appetizing, vibrant, mouth-watering visual."
        ),
        "IT": (
            "Korean IT tech company. "
            "Cool, clean, modern aesthetic (blue, white, dark navy). "
            "Sharp, precise, professional. Technology-forward feel."
        ),
    }

    color_guidance = {
        "warm (따뜻한)": "dominant warm tones: deep red, golden orange, amber, bright yellow. High saturation, energetic.",
        "cool (차가운)": "dominant cool tones: electric blue, cyan, deep navy, crisp white. Crisp and modern.",
        "neutral (중간)": "balanced neutral palette with one strong accent color. Clean and professional.",
    }

    cat_rule = category_rules.get(category, "Dynamic, eye-catching composition.")
    dom_style = domain_style.get(domain, "Professional Korean brand.")
    col_guide = color_guidance.get(color_tone, "vibrant, saturated colors.")

    # ── 키워드에서 구체적 시각 요소 추출 (고유명사·제품명·캐릭터명 대응) ──
    keyword_analysis_prompt = f"""You are a visual translator for image generation.
Given these Korean keywords for a YouTube thumbnail: "{keywords}"

Extract and expand them into specific visual elements for image generation.
Focus on: what objects/characters/foods to show, their colors, textures, shapes.
For brand collabs (e.g. "스폰지밥 CU 콜라보"): describe SpongeBob character appearance + CU convenience store branding colors (blue/yellow).
For foods (e.g. "크림빵"): describe the cream bun in detail — golden-brown crust, white cream filling visible, soft texture.
For products: describe color, shape, packaging details.

Return ONLY a JSON:
{{
  "visual_subjects": "main visual subjects described in English with appearance details",
  "key_objects": "specific objects/products/characters with color and texture",
  "scene_context": "background or scene that fits the keyword context"
}}"""

    import json as _json
    _kw_mdl = GenerativeModel(GEMINI_PROMPT_MODEL)
    _kw_cfg = GenerationConfig(
        temperature=0.3,
        max_output_tokens=400,
        response_mime_type="application/json",
    )
    visual_elements = {"visual_subjects": keywords, "key_objects": "", "scene_context": ""}
    try:
        _kw_resp = _kw_mdl.generate_content(keyword_analysis_prompt, generation_config=_kw_cfg)
        _kw_data = _json.loads(_kw_resp.text.strip())
        visual_elements.update(_kw_data)
    except Exception:
        pass  # fallback: 원본 키워드 그대로 사용

    system_prompt = f"""You are an expert YouTube thumbnail image generation prompt engineer.

Create a highly detailed, specific Imagen prompt for a YouTube thumbnail.
The result MUST look like a REAL YouTube thumbnail — bold, punchy, eye-catching, high contrast.
Write 150-200 words for maximum visual specificity.

[Channel & Content Context]
{ch_ctx}
Domain: {domain} | Category: {category} | Color tone: {color_tone}
Original Keywords: {keywords}

[Visual Subjects — MUST appear prominently in the image]
Main subjects: {visual_elements["visual_subjects"]}
Key objects/details: {visual_elements["key_objects"]}
Scene context: {visual_elements["scene_context"]}

[Domain Visual Style]
{dom_style}

[Category-Specific MANDATORY Requirements]
{cat_rule}

[Color Guidance]
{col_guide}

[UNIVERSAL THUMBNAIL RULES — ALL MUST BE APPLIED]
1. VIVID, HIGHLY SATURATED colors — NOT muted, NOT moody, NOT dark
2. BRIGHT, WELL-LIT scene — NOT atmospheric, NOT cinematic dark lighting
3. Subject fills 60-80% of frame — EXTREME close-up or tight framing
4. HIGH CONTRAST between subject and background — subject must POP
5. DYNAMIC, energetic composition — NOT static product photography
6. Clean empty space on LEFT or RIGHT side for text overlay
7. Wide horizontal framing (16:9 landscape ratio)
8. Do NOT render any text, letters, words, or numbers inside the image
9. Background must be simple and NOT compete with the main subject
10. Every named object/character/food MUST be visually recognizable and detailed

[CRITICAL ANTI-PATTERNS TO AVOID]
- Do NOT create moody atmospheric lighting
- Do NOT create generic stock photo look
- Do NOT use dark shadows or low-key lighting
- Do NOT place subject far in the background
- Do NOT ignore or generalize the specific keywords — render them literally

Respond ONLY in this JSON format:
{{
  "prompt_en": "detailed English prompt for Imagen, 150-200 words, describing every visual element specifically",
  "prompt_ko": "한국어 1문장 요약"
}}"""

    init_vertex(project_id, location)
    import json as _json
    mdl = GenerativeModel(GEMINI_PROMPT_MODEL)
    cfg = GenerationConfig(
        temperature=IMAGE_GEN_CONFIG["prompt_temperature"],
        max_output_tokens=IMAGE_GEN_CONFIG["prompt_max_tokens"],
        response_mime_type="application/json",
    )
    resp = mdl.generate_content(system_prompt, generation_config=cfg)

    try:
        data = _json.loads(resp.text.strip())
        return data.get("prompt_en",""), data.get("prompt_ko","")
    except Exception:
        text = resp.text
        en_p, ko_p = "", ""
        for line in text.split("\n"):
            if "prompt_en" in line.lower():
                en_p = line.split(":",1)[-1].strip().strip('"')
            elif "prompt_ko" in line.lower():
                ko_p = line.split(":",1)[-1].strip().strip('"')
        return en_p or text[:400], ko_p

def gemini_gen_improvement_prompt(
    user_instruction: str,
    domain: str,
    current_summary: str,
    strengths: list,
    improvement_hints: list,
    project_id: str,
    location: str = "us-central1",
) -> str:
    """기존 썸네일 분석 결과 + 사용자 수정 지시를 바탕으로 Imagen용 개선 프롬프트 생성"""
    init_vertex(project_id, location)

    domain_ctx = (
        "Korean FnB food & beverage brand — vivid warm colors, appetizing, energetic"
        if domain == "FnB"
        else "Korean IT tech company — clean cool tones, modern, professional"
    )

    prompt = f"""You are an expert YouTube thumbnail prompt engineer.
Generate one improved Imagen prompt based on the analysis below.
The result must look like a REAL YouTube thumbnail — bold, punchy, eye-catching.

[Domain]
{domain_ctx}

[Current thumbnail analysis]
{current_summary or "No detailed analysis available."}

[Strengths to preserve]
{", ".join(strengths[:3]) if strengths else "Preserve the strongest visual elements."}

[Improvements to apply]
{", ".join(improvement_hints[:5]) if improvement_hints else "Improve visual impact and click appeal."}

[User instruction]
{user_instruction or "Improve the thumbnail while preserving the original concept."}

[MANDATORY THUMBNAIL RULES]
- VIVID, HIGHLY SATURATED colors — NOT muted or moody
- BRIGHT, WELL-LIT — NOT cinematic dark lighting
- Subject must be LARGE and fill the frame — NOT small or distant
- HIGH CONTRAST between subject and background
- Clean empty space on one side for text overlay
- Do NOT render any text, letters, or numbers in the image
- Wide horizontal framing

Return only the final English prompt (100-150 words). No markdown. No explanation.
"""

    mdl = GenerativeModel(GEMINI_PROMPT_MODEL)
    resp = mdl.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=IMAGE_GEN_CONFIG["prompt_temperature"],
            max_output_tokens=IMAGE_GEN_CONFIG["prompt_max_tokens"],
        ),
    )
    return resp.text.strip()


# ──────────────────────────────────────────────
# 4. 썸네일 분석 — Pydantic 구조화 출력
# ──────────────────────────────────────────────
def gemini_gen_thumbnail_analysis(
    img_url: str,
    title: str,
    views: int,
    domain: str,
    bench: dict,
    project_id: str,
    location: str = "us-central1",
) -> dict:
    """
    Vertex AI Gemini Vision으로 썸네일 이미지 분석.
    - 이미지만 보고 개선점 도출 (텍스트 맥락 최소화)
    - JSON 직접 파싱 반환
    """
    import json as _json

    prompt = f"""You are a professional YouTube thumbnail analyst.
Analyze ONLY what you can visually observe in this thumbnail image.
Do NOT guess or assume information not visible in the image.

Context (for benchmark comparison only):
- Video title: {title}
- Domain: {domain}
- Success benchmarks: person {bench['has_person']}%, multi-person {bench['person_cat']['2명+']}%, text {bench['has_text']}%, brand {bench['brand']}%, brightness {bench['brightness']}, saturation {bench['saturation']}, design quality {bench['design_quality']}/5

CRITICAL: Your ENTIRE response must be a single valid JSON object. No markdown, no code fences, no explanation before or after. Start your response with {{ and end with }}.

Return EXACTLY this JSON structure:
{{
  "elements": {{
    "main_objects": "주요 피사체와 배치 설명 (이미지에서 보이는 것만)",
    "text_overlay": "이미지에 보이는 텍스트 내용 (없으면 '텍스트 없음')",
    "color_palette": "주요 색상, 밝기, 채도 특징",
    "person_count": 0,
    "person_details": "인물 표정, 구도 설명 (인물 없으면 '인물 없음')",
    "brand_elements": "로고, 브랜드 색상, 아이덴티티 요소 (없으면 '없음')"
  }},
  "benchmark_comparison": {{
    "person_score": "벤치마크 {bench['has_person']}% 기준 대비 현재 썸네일 평가",
    "text_score": "벤치마크 {bench['has_text']}% 기준 대비 현재 썸네일 평가",
    "color_score": "권장 밝기/채도 기준 대비 현재 썸네일 평가",
    "design_score": "디자인 품질 {bench['design_quality']}/5 기준 대비 현재 썸네일 평가",
    "overall_ctr": "낮음/보통/높음 + 이유 한 줄"
  }},
  "strengths": [
    "시각적으로 잘 된 점 1",
    "시각적으로 잘 된 점 2"
  ],
  "improvements": [
    {{
      "issue": "이미지에서 발견한 구체적 문제점",
      "action": "구체적 개선 액션 (이미지 기반)",
      "prompt_hint": "이미지 생성 프롬프트에 쓸 영어 표현"
    }},
    {{
      "issue": "문제점 2",
      "action": "개선 액션 2",
      "prompt_hint": "영어 표현 2"
    }},
    {{
      "issue": "문제점 3",
      "action": "개선 액션 3",
      "prompt_hint": "영어 표현 3"
    }}
  ]
}}"""

    init_vertex(project_id, location)

    # ── 이미지 로드 ──
    contents = []
    if img_url:
        try:
            if img_url.startswith("data:image"):
                _, b64_data = img_url.split(",", 1)
                img_bytes = base64.b64decode(b64_data)
            else:
                r = requests.get(img_url, timeout=10)
                img_bytes = r.content
            contents.append(Part.from_data(data=img_bytes, mime_type="image/jpeg"))
        except Exception:
            pass
    contents.append(prompt)

    mdl = GenerativeModel(GEMINI_VISION_MODEL)
    # ※ 멀티모달(이미지+텍스트) 요청에서 response_mime_type="application/json" 은
    #   gemini-2.5-flash 에서 불안정 → 제거하고 프롬프트 레벨에서 JSON 강제
    cfg = GenerationConfig(
        temperature=0.1,
        max_output_tokens=4096,
    )

    # ── 최대 3회 재시도 ──
    raw = ""
    for _attempt in range(3):
        try:
            resp = mdl.generate_content(contents, generation_config=cfg)
            raw  = resp.text.strip()
            if raw:
                break
        except Exception:
            if _attempt == 2:
                raise
            import time as _time
            _time.sleep(2)

    # ── JSON 파싱 — 4단계 방어 ──
    import re as _re
    data = {}

    def _try_parse(text: str):
        try:
            return _json.loads(text)
        except Exception:
            return None

    # 1단계: 직접 파싱
    data = _try_parse(raw) or {}

    if not data:
        # 2단계: 코드블록 제거
        _clean = _re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = _try_parse(_clean) or {}

    if not data:
        # 3단계: 첫 번째 { ... } 블록 추출
        _m = _re.search(r'\{[\s\S]*\}', raw)
        if _m:
            data = _try_parse(_m.group()) or {}

    if not data:
        # 4단계: 불완전 JSON 끝부분 자르고 닫기 시도
        _truncated = raw[:raw.rfind("}") + 1] if "}" in raw else raw
        data = _try_parse(_truncated) or {}

    if data:
        prompt_hints = [i.get("prompt_hint","") for i in data.get("improvements",[]) if i.get("prompt_hint")]
        return {
            "elements":    data.get("elements", {}),
            "benchmark":   data.get("benchmark_comparison", data.get("benchmark", {})),
            "strengths":   data.get("strengths", []),
            "improvements":data.get("improvements", []),
            "prompt_hints":prompt_hints,
            "raw": raw,
        }
    else:
        return {"elements":{}, "benchmark":{}, "strengths":[], "improvements":[], "prompt_hints":[], "raw":raw}

# ──────────────────────────────────────────────
# 4b. 상위 5개 썸네일 일괄 분석 → 통계 집계
# ──────────────────────────────────────────────
def analyze_top5_thumbnails(videos, bench, domain, project_id, location="us-central1"):
    """
    상위 5개 롱폼 영상 썸네일을 Gemini로 분석하고 통계를 집계해서 반환.
    반환: {
        "results": [분석결과 dict, ...],  # 개별 분석
        "stats": {집계 통계},
        "videos": [영상 메타데이터, ...]
    }
    """
    top5 = sorted(videos, key=lambda x: x.get("views", 0), reverse=True)[:5]
    results = []
    for v in top5:
        url = v.get("thumbnail_hq") or v.get("thumbnail", "")
        if not url:
            continue
        try:
            r = gemini_gen_thumbnail_analysis(
                url, v.get("title",""), v.get("views",0),
                domain, bench, project_id, location
            )
            r["video_id"]    = v["id"]
            r["video_title"] = v.get("title","")
            r["views"]       = v.get("views", 0)
            r["thumbnail"]   = v.get("thumbnail","")
            results.append(r)
        except Exception:
            pass

    if not results:
        return {"results": [], "stats": {}, "videos": top5}

    # ── 통계 집계 ──────────────────────────────
    def _has(results, keyword):
        """분석 결과 텍스트에서 키워드 포함 비율"""
        cnt = 0
        for r in results:
            elems = r.get("elements", {})
            text  = " ".join(str(v) for v in elems.values()).lower()
            if keyword.lower() in text:
                cnt += 1
        return round(cnt / len(results) * 100)

    def _multi(results):
        """2인 이상 인물 구성 비율 — person_count 숫자 필드 우선 사용"""
        import re as _re
        cnt = 0
        for r in results:
            elems = r.get("elements") or {}
            # ① person_count 숫자 필드 우선 (프롬프트에서 강제)
            pc = elems.get("person_count")
            if pc is not None:
                try:
                    if int(pc) >= 2:
                        cnt += 1
                    continue
                except (ValueError, TypeError):
                    pass
            # ② fallback: person_details 텍스트에서 숫자 추출
            p = str(elems.get("person_details", "") or elems.get("person_composition", "")).lower()
            nums = _re.findall(r"\d+", p)
            if nums and int(nums[0]) >= 2:
                cnt += 1
            elif any(k in p for k in ["two","multiple","group","both","couple","pair","several"]):
                cnt += 1
        return round(cnt / len(results) * 100) if results else 0

    def _score_dist(results, key):
        """benchmark score 분포 (높음/보통/낮음)"""
        dist = {"높음": 0, "보통": 0, "낮음": 0}
        for r in results:
            bench_ = r.get("benchmark", r.get("benchmark_comparison", {}))
            v = str(bench_.get(key, ""))
            if "높음" in v:   dist["높음"] += 1
            elif "보통" in v: dist["보통"] += 1
            elif "낮음" in v: dist["낮음"] += 1
        total = len(results)
        return {k: round(v/total*100) for k,v in dist.items()}

    def _top_issues(results):
        """반복 등장 개선점 상위 3개"""
        from collections import Counter
        issues = []
        for r in results:
            for imp in r.get("improvements", []):
                if isinstance(imp, dict) and imp.get("issue"):
                    issues.append(imp["issue"])
        return [iss for iss, _ in Counter(issues).most_common(3)]

    def _top_strengths(results):
        """반복 등장 강점 상위 3개"""
        from collections import Counter
        strs = []
        for r in results:
            for s in r.get("strengths", []):
                strs.append(s)
        return [s for s, _ in Counter(strs).most_common(3)]

    stats = {
        "count":           len(results),
        "has_person_pct":  _has(results, "인물"),
        "has_text_pct":    _has(results, "텍스트"),
        "has_brand_pct":   _has(results, "로고"),
        "multi_person_pct": _multi(results),
        "person_score":    _score_dist(results, "person_score"),
        "text_score":      _score_dist(results, "text_score"),
        "color_score":     _score_dist(results, "color_score"),
        "design_score":    _score_dist(results, "design_score"),
        "ctr_dist":        _score_dist(results, "overall_ctr"),
        "top_issues":      _top_issues(results),
        "top_strengths":   _top_strengths(results),
    }

    return {"results": results, "stats": stats, "videos": top5}


# ──────────────────────────────────────────────
# 5a. 이미지 생성 — Vertex AI Imagen 4.0 (새 썸네일)
# ──────────────────────────────────────────────
def gemini_gen_image(
    prompt: str,
    project_id: str,
    location: str = "us-central1",
    model: str = IMAGEN_MODEL,
) -> bytes:
    """Vertex AI Imagen으로 새 이미지 생성. 반환: PNG bytes"""
    init_vertex(project_id, location)

    # 썸네일 특화 스타일 강제 — Imagen이 상업 사진이 아닌 유튜브 썸네일처럼 생성하도록
    enhanced = (
        f"{prompt}. "
        "YouTube thumbnail style: VIVID saturated colors, HIGH CONTRAST, BRIGHT well-lit scene. "
        "Subject fills majority of frame, tight close-up composition. "
        "Simple clean background that does not compete with subject. "
        "Eye-catching, energetic, NOT moody or cinematic. "
        "Wide horizontal format. No text or letters in image."
    )

    raw_bytes = imagen_generate_bytes(enhanced, project_id, location, model)
    from PIL import Image as _PilImage
    pil_img = _PilImage.open(BytesIO(raw_bytes))
    out = BytesIO()
    pil_img.save(out, format="PNG")
    return out.getvalue()

# ──────────────────────────────────────────────
# 5b. 기존 썸네일 스타일 참고 재생성
# Step 1: Gemini Vision으로 기존 썸네일 스타일/구도/색감 상세 분석
# Step 2: 분석 결과 + 수정 지시 → Imagen 4.0으로 고품질 재생성
# ──────────────────────────────────────────────
def gemini_edit_image(
    img_bytes_or_url: bytes | str,
    edit_prompt: str,          # 사용자 수정 지시 (한국어 OK)
    analysis_hints: str,       # 썸네일 분석에서 추출한 영어 힌트
    domain: str,
    project_id: str,
    location: str = "us-central1",
) -> tuple[bytes, str]:
    """
    기존 썸네일을 참고해 개선된 새 썸네일 생성.
    반환: (PNG bytes, 사용된 최종 프롬프트)
    """
    init_vertex(project_id, location)

    # ── 이미지 로드 ──
    if isinstance(img_bytes_or_url, str):
        r = requests.get(img_bytes_or_url, timeout=10)
        img_raw = r.content
    else:
        img_raw = img_bytes_or_url

    domain_ctx = (
        "Korean FnB food & beverage brand, appetizing warm style"
        if domain == "FnB"
        else "Korean IT tech company, professional clean style"
    )

    # ── Step 1: 기존 썸네일 스타일 분석 (구도/색감/인물/분위기 위주) ──
    desc_prompt = f"""Analyze this YouTube thumbnail and describe it for recreating a similar style.
Focus on:
1. COMPOSITION: subject placement, framing, depth
2. COLOR SCHEME: dominant colors, tone (warm/cool/neutral), brightness, saturation
3. SUBJECT: person count, expressions, clothing style, poses
4. BRAND ELEMENTS: logos, brand colors, visual identity
5. ATMOSPHERE: overall mood, energy level, visual style

Be specific and concise. English only. Max 200 words."""

    img_part = Part.from_data(data=img_raw, mime_type="image/jpeg")
    mdl_desc = GenerativeModel(GEMINI_VISION_MODEL)
    desc_resp = mdl_desc.generate_content(
        [img_part, desc_prompt],
        generation_config=GenerationConfig(temperature=IMAGE_GEN_CONFIG["vision_temperature"], max_output_tokens=IMAGE_GEN_CONFIG["style_desc_max_tokens"]),
    )
    style_desc = desc_resp.text.strip()

    # ── Step 2: 수정 지시(한국어) → 영어 번역 후 Imagen 프롬프트 구성 ──
    # Imagen은 영어 전용 모델 — 한국어를 그대로 넣으면 이미지 안에 깨진 글자가 렌더링됨
    if edit_prompt.strip():
        _translate_prompt = (
            "Translate the following Korean YouTube thumbnail improvement instructions "
            "into concise English for an image generation AI. "
            "Return ONLY the English translation, no explanation, no markdown.\n\n"
            f"Korean: {edit_prompt.strip()}"
        )
        _mdl_tr = GenerativeModel(GEMINI_PROMPT_MODEL)
        _tr_resp = _mdl_tr.generate_content(
            _translate_prompt,
            generation_config=GenerationConfig(temperature=IMAGE_GEN_CONFIG["translate_temperature"], max_output_tokens=IMAGE_GEN_CONFIG["translate_max_tokens"]),
        )
        edit_en = _tr_resp.text.strip()
    else:
        edit_en = "improve overall visual quality"

    final_prompt = (
        f"YouTube thumbnail for {domain_ctx}. "
        f"Style reference: {style_desc[:IMAGE_GEN_CONFIG['style_desc_trim']]}. "
        f"Improvements and changes: {edit_en}. "
        f"{('Additional: ' + analysis_hints + '. ') if analysis_hints else ''}"
        f"High quality professional photography, 16:9 aspect ratio, "
        f"Korean brand thumbnail aesthetic. "
        f"Do not render any text or letters inside the image."
    )

    # ── Step 3: Imagen 4.0으로 생성 ──
    raw_bytes = imagen_generate_bytes(final_prompt, project_id, location, IMAGEN_MODEL)
    from PIL import Image as _PIL
    out = BytesIO()
    _PIL.open(BytesIO(raw_bytes)).save(out, format="PNG")
    return out.getvalue(), final_prompt

# ══════════════════════════════════════════════
# 개선점 → 영어 이미지 프롬프트 힌트 변환
# ══════════════════════════════════════════════
def _pts_to_prompt_hint(bad: list, act: list) -> str:
    """
    한국어 개선점/권장사항 리스트 → 이미지 생성 프롬프트용 영어 힌트 문자열 변환.
    키워드 매핑 방식으로 자연스러운 영어 표현으로 치환.
    """
    kw_map = {
        "인물":    "prominent person in frame",
        "얼굴":    "expressive face closeup",
        "클로즈업": "tight face closeup shot",
        "브랜드":  "visible brand logo placement",
        "로고":    "brand logo clearly visible",
        "텍스트":  "clean empty space for text overlay on one side",
        "밝기":    "bright well-lit scene",
        "채도":    "vibrant saturated colors",
        "디자인":  "polished professional design quality",
        "참여율":  "eye-catching engaging composition",
        "후킹":    "strong visual hook element",
        "간결":    "clean minimal composition",
        "색상":    "impactful color contrast",
        "대비":    "high contrast composition",
        "구성":    "dynamic balanced composition",
        "배경":    "clean uncluttered background",
    }
    hints = []
    for pt in (bad or []) + (act or []):
        for ko, en in kw_map.items():
            if ko in pt and en not in hints:
                hints.append(en)
                break
    return ", ".join(hints) if hints else "improved professional composition"

def run_channel_analysis(cid, key, bench, domain_name):
    """
    선택된 채널 ID로 영상 수집, 롱폼/숏폼 분류, 채널 분석까지 수행.
    yt_channel_info()에서 받은 uploads_playlist를 yt_fetch_videos()에 전달해
    channels API 추가 호출을 방지한다.
    quota: channels(1) + playlistItems(ceil(50/50)×1) + videos(ceil(50/50)×1) = 3 units
    """
    ch   = yt_channel_info(cid, key)
    vids = yt_fetch_videos(cid, key, max_results=50,
                           uploads_playlist=ch.get("uploads_playlist", ""))
    vid_ids = [v["id"] for v in vids]
    verdicts = classify_videos_redirect(vid_ids, workers=min(len(vid_ids), 20)) if vid_ids else {}
    for v in vids:
        v["verdict"] = verdicts.get(v["id"], "unknown")
        v["is_longform"] = (v["verdict"] == "longform")
        v["is_shortform"] = (v["verdict"] == "shorts")
    ana = analyze_channel(vids, bench, domain_name)
    return ch, vids, ana
# ══════════════════════════════════════════════
# 차트
# ══════════════════════════════════════════════
def grouped_bar(categories, success_vals, fail_vals, title="", accent="#ff5555", h=220):
    """성공 vs 실패 그룹 막대 차트"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="성공", x=categories, y=success_vals,
        marker_color=accent, opacity=0.85,
        text=[f"{v:.1f}" for v in success_vals],
        textposition="outside", textfont=dict(size=9, color="#4b5563"),
    ))
    fig.add_trace(go.Bar(
        name="실패", x=categories, y=fail_vals,
        marker_color="#444", opacity=0.7,
        text=[f"{v:.1f}" for v in fail_vals],
        textposition="outside", textfont=dict(size=9, color="#888"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#aaa"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30,b=10,l=4,r=4), height=h, barmode="group",
        legend=dict(font=dict(size=9,color="#6b7280"), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, tickfont=dict(size=9,color="#6b7280")),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", tickfont=dict(size=9,color="#6b7280")),
    )
    return fig

def hbar(labels, vals, colors, title="", h=220):
    """수평 막대 차트"""
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors if isinstance(colors,list) else [colors]*len(vals),
        opacity=0.85,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside", textfont=dict(size=9,color="#4b5563"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11,color="#6b7280"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=28,b=4,l=4,r=60), height=h,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=9,color="#4b5563")),
        bargap=0.35,
    )
    return fig

def donut(labels, vals, colors, title="", h=200):
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.55,
        textinfo="percent", textfont=dict(size=10,color="#fff"),
        marker=dict(colors=colors, line=dict(color="#ffffff",width=2)),
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=28,b=4,l=4,r=4), height=h,
        title=dict(text=title, font=dict(size=11,color="#6b7280"), x=0.5),
        legend=dict(font=dict(size=9,color="#6b7280"), bgcolor="rgba(0,0,0,0)", x=1.0, y=0.5),
    )
    return fig

def radar(fnb_r, it_r, labels, h=240):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=fnb_r+[fnb_r[0]], theta=labels+[labels[0]],
        fill="toself", fillcolor="rgba(255,0,0,.12)",
        line=dict(color="#ff0000",width=2), name="FnB 성공"))
    fig.add_trace(go.Scatterpolar(r=it_r+[it_r[0]], theta=labels+[labels[0]],
        fill="toself", fillcolor="rgba(62,166,255,.12)",
        line=dict(color="#3ea6ff",width=2), name="IT 성공"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,showticklabels=False,gridcolor="#e5e7eb",linecolor="#e5e7eb"),
            angularaxis=dict(tickfont=dict(size=10,color="#4b5563"),gridcolor="#e5e7eb",linecolor="#e5e7eb")),
        legend=dict(font=dict(size=10,color="#6b7280"),bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=16,b=16,l=40,r=40), height=h,
    )
    return fig

# ══════════════════════════════════════════════
# 공통: 채널 분석 결과 렌더링
# ══════════════════════════════════════════════
def render_channel_result(prefix, accent, bench, domain_label):
    ch  = st.session_state[f"{prefix}_channel"]
    ana = st.session_state[f"{prefix}_analysis"] or {}
    videos = st.session_state[f"{prefix}_videos"] or []
    tier, tc = get_tier(ch["subscribers"])
    bc = "badge-red" if domain_label=="FnB" else "badge-blue"

    # 채널 헤더
    st.markdown('<div class="yt-card">', unsafe_allow_html=True)
    hc1, hc2 = st.columns([1,5])
    with hc1:
        if ch.get("avatar"):
            try:
                _avatar_resp = requests.get(ch["avatar"], timeout=8, headers=_REDIRECT_HEADERS)
                _avatar_resp.raise_for_status()
                st.image(BytesIO(_avatar_resp.content), width=64)
            except Exception:
                st.markdown(
                    '<div style="width:64px;height:64px;border-radius:50%;background:#e5e7eb;'
                    'display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:22px">▶</div>',
                    unsafe_allow_html=True,
                )
    with hc2:
        st.markdown(
            f'<div style="padding-top:2px">'
            f'<span style="font-size:15px;font-weight:700">{ch["name"]}</span>'
            f'<span class="badge {bc}" style="margin-left:6px">{tier}</span>'
            f'<div style="font-size:11px;color:#6b7280;margin-top:4px">{ch["description"][:100]}...</div>'
            f'</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 통계
    s1,s2,s3,s4 = st.columns(4)
    for col,val,lbl,c in [
        (s1, fmt_num(ch["subscribers"]),    "구독자",         tc),
        (s2, fmt_num(ch["views"]),           "총 조회수",      "#3ea6ff"),
        (s3, f"{ana.get('er_rate',0):.2f}%", "롱폼 참여율(ER)","#ffd600"),
        (s4, str(ana.get("longform_count",0)),"롱폼 영상 수",  "#2ba640"),
    ]:
        col.markdown(f'<div class="stat-box"><div class="stat-val" style="color:{c}">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # 롱폼 영상 목록 — redirect verdict 기반, 숏폼 완전 제외
    _lf_r = [v for v in videos if v.get("verdict") == "longform"]
    _lf_u = [v for v in videos if v.get("verdict") == "unknown" and v.get("duration",0) >= 60]
    _lf_n = [v for v in videos if "verdict" not in v and v.get("duration",0) >= 60]
    if _lf_r:
        lf_videos = _lf_r
    elif _lf_r + _lf_u:
        lf_videos = _lf_r + _lf_u
    elif _lf_n:
        lf_videos = _lf_n
    else:
        lf_videos = [v for v in videos if v.get("verdict") != "shorts"] or videos
    if lf_videos:
        st.markdown(
            f'<div style="font-size:12px;color:#6b7280;margin-bottom:8px">'
            f'📹 롱폼 영상 목록 (총 {len(lf_videos)}개) — 썸네일 클릭 시 AI 분석</div>',
            unsafe_allow_html=True)

        # 5개씩 행으로 표시
        rows = [lf_videos[i:i+5] for i in range(0, min(len(lf_videos), 10), 5)]
        for row in rows:
            cols = st.columns(5)
            for col, v in zip(cols, row):
                with col:
                    selected_key = f"{prefix}_selected_video"
                    is_selected = (st.session_state[selected_key] or {}).get("id") == v["id"]
                    border_color = "#3ea6ff" if is_selected else "#e8eaed"
                    if v.get("thumbnail"):
                        st.image(v["thumbnail"], use_container_width=True)
                    st.markdown(
                        f'<div style="font-size:9px;color:#6b7280;margin-top:3px;line-height:1.3">'
                        f'{v["title"][:32]}...</div>'
                        f'<div style="font-size:9px;color:#6b7280">👁 {fmt_num(v["views"])} '
                        f'· {v["duration_fmt"]}</div>',
                        unsafe_allow_html=True)
                    if st.button("🔍 분석", key=f"ana_{prefix}_{v['id']}", help=v["title"]):
                        st.session_state[selected_key] = v
                        st.session_state[f"{prefix}_thumb_analysis"] = None
                        st.rerun()

    # 선택된 썸네일 분석 패널
    sel = st.session_state.get(f"{prefix}_selected_video")
    if sel:
        st.markdown("<hr style='border-color:#2563eb;margin:12px 0'>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="analysis-modal">'
            f'<div style="font-size:13px;font-weight:600;color:#2563eb;margin-bottom:10px">'
            f'🔍 썸네일 분석 — {sel["title"][:50]}...</div>',
            unsafe_allow_html=True)

        ap1, ap2 = st.columns([1, 2])
        with ap1:
            if sel.get("thumbnail_hq"):
                st.image(sel["thumbnail_hq"], use_container_width=True)
            st.markdown(
                f'<div style="font-size:10px;color:#6b7280;margin-top:4px">'
                f'조회수 {fmt_num(sel["views"])} · 좋아요 {fmt_num(sel["likes"])} '
                f'· 댓글 {fmt_num(sel["comments"])} · {sel["duration_fmt"]}</div>',
                unsafe_allow_html=True)

        with ap2:
            vertex_project  = st.session_state.get("vertex_project","")
            vertex_location = st.session_state.get("vertex_location","us-central1")
            cached = st.session_state.get(f"{prefix}_thumb_analysis")

            if cached and cached.get("video_id") == sel["id"]:
                # 구조화된 분석 결과 카드 렌더링
                # cached = {"video_id":..., "data": result}
                # result  = {"elements":{}, "benchmark":{}, "strengths":[], "improvements":[], "raw":"..."}
                import json as _j, re as _re

                # ── 단계적 파싱 (어떤 구조로 저장됐든 커버) ──
                # 1단계: data 키로 감싸진 경우
                d = cached.get("data") or {}
                # 2단계: data 없으면 cached 자체가 result
                if not d:
                    d = {k: v for k, v in cached.items() if k != "video_id"}

                raw    = d.get("raw", cached.get("text", ""))
                elems  = d.get("elements", {})
                # "benchmark" 또는 "benchmark_comparison" 둘 다 시도
                bench_ = d.get("benchmark") or d.get("benchmark_comparison") or {}
                strns  = d.get("strengths", [])
                imprv  = d.get("improvements", [])

                # 3단계: elems나 bench_ 중 하나라도 비어있으면 raw에서 파싱
                if raw and (not elems or not bench_):
                    _clean = _re.sub(r"```json|```", "", raw.strip()).strip()
                    try:
                        _parsed = _j.loads(_clean)
                        elems  = elems  or _parsed.get("elements", {})
                        bench_ = bench_ or _parsed.get("benchmark_comparison") or _parsed.get("benchmark") or {}
                        strns  = strns  or _parsed.get("strengths", [])
                        imprv  = imprv  or _parsed.get("improvements", [])
                    except Exception:
                        pass

                if elems or bench_ or strns or imprv:
                    # ── 구성 요소 ──
                    st.markdown('<div style="font-size:11px;font-weight:700;color:#2563eb;margin-bottom:6px">📋 썸네일 구성 요소 (이미지 분석)</div>', unsafe_allow_html=True)
                    elem_rows = [
                        ("🔹 주요 피사체", elems.get("main_objects",    elems.get("main_objects","—"))),
                        ("🔹 텍스트",      elems.get("text_overlay",    elems.get("text_content","—"))),
                        ("🔹 색상/밝기",   elems.get("color_palette",   elems.get("color_feature","—"))),
                        ("🔹 인물",        elems.get("person_details",  elems.get("person_composition","—"))),
                        ("🔹 브랜드",      elems.get("brand_elements",  "—")),
                    ]
                    import html as _html
                    rows_html = "".join([
                        f'<div style="display:flex;gap:8px;padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:11px">'
                        f'<span style="color:#888;flex-shrink:0;width:75px">{_html.escape(str(lb))}</span>'
                        f'<span style="color:#374151;line-height:1.5">{_html.escape(str(vl))}</span></div>'
                        for lb,vl in elem_rows if vl and vl != "—"
                    ])
                    st.markdown(f'<div style="background:#ffffff;border-radius:8px;overflow:hidden;margin-bottom:10px">{rows_html}</div>', unsafe_allow_html=True)

                    # ── 벤치마크 비교 점수 ──
                    if bench_:
                        st.markdown('<div style="font-size:11px;font-weight:700;color:#d97706;margin-bottom:6px">📊 업종 기준 비교</div>', unsafe_allow_html=True)
                        bk_cols = st.columns(2)
                        bk_items = [(k,v) for k,v in bench_.items() if k != "overall_ctr"]
                        for i, (k, v) in enumerate(bk_items):
                            col = bk_cols[i % 2]
                            label_map = {"person_score":"🔸 인물","text_score":"🔸 텍스트","color_score":"🔸 색상","design_score":"🔸 디자인"}
                            lbl = label_map.get(k, k)
                            col.markdown(
                                f'<div style="background:#f0f2f5;border-radius:6px;padding:8px;margin-bottom:6px;font-size:10px">'
                                f'<div style="color:#aaa;margin-bottom:3px;font-weight:600">{_html.escape(str(lbl))}</div>'
                                f'<div style="color:#374151;line-height:1.5">{_html.escape(str(v))}</div></div>',
                                unsafe_allow_html=True)
                        ctr = bench_.get("overall_ctr","")
                        if ctr:
                            ctr_color = "#4dd068" if "높음" in ctr else ("#ffe033" if "보통" in ctr else "#ff5555")
                            st.markdown(
                                f'<div style="background:#f0f2f5;border-radius:6px;padding:8px;font-size:11px;margin-bottom:10px">' 
                                f'<span style="color:#aaa;margin-right:8px">🔸예상 클릭율</span>' 
                                f'<span style="color:{ctr_color};font-weight:700">{ctr}</span></div>',
                                unsafe_allow_html=True)

                    # ── 잘 된 점 ──
                    if strns:
                        st.markdown('<div style="font-size:11px;font-weight:700;color:#16a34a;margin-bottom:6px">✅ 시각적 강점</div>', unsafe_allow_html=True)
                        for s in strns:
                            st.markdown(f'<div class="good-point" style="font-size:11px">{_html.escape(str(s))}</div>', unsafe_allow_html=True)

                    # ── 개선 제안 (이미지 기반) ──
                    if imprv:
                        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
                        st.markdown('<div style="font-size:11px;font-weight:700;color:#ea580c;margin-bottom:6px">💡 이미지 기반 개선 제안</div>', unsafe_allow_html=True)
                        for i, tip in enumerate(imprv):
                            if isinstance(tip, dict):
                                issue  = tip.get("issue","")
                                action = tip.get("action","")
                                st.markdown(
                                    f'<div style="background:#f0f2f5;border-left:3px solid #ff9944;border-radius:0 6px 6px 0;padding:8px 12px;margin-bottom:6px">'
                                    f'<div style="font-size:10px;color:#ea580c;font-weight:600;margin-bottom:3px">문제 {i+1}: {_html.escape(str(issue))}</div>'
                                    f'<div style="font-size:11px;color:#1f2937;line-height:1.5">&#8594; {_html.escape(str(action))}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="action-point"><span style="color:#ffe033;font-weight:700;margin-right:6px">{i+1}</span><span style="font-size:11px">{tip}</span></div>',
                                    unsafe_allow_html=True)
                else:
                    # 파싱된 데이터가 전혀 없는 경우 — raw에서 한 번 더 시도
                    import html as _html_mod, json as _j2, re as _re2
                    _attempted = False
                    if raw:
                        _clean2 = _re2.sub(r"```json|```", "", raw.strip()).strip()
                        try:
                            _p2 = _j2.loads(_clean2)
                            elems  = _p2.get("elements", {})
                            bench_ = _p2.get("benchmark_comparison") or _p2.get("benchmark") or {}
                            strns  = _p2.get("strengths", [])
                            imprv  = _p2.get("improvements", [])
                            _attempted = bool(elems or bench_ or strns or imprv)
                        except Exception:
                            pass

                    if _attempted:
                        # 재파싱 성공 → 간단 요약 출력
                        if strns:
                            st.markdown('<div style="font-size:11px;font-weight:700;color:#16a34a;margin-bottom:6px">✅ 시각적 강점</div>', unsafe_allow_html=True)
                            import html as _he
                            for s in strns:
                                st.markdown(f'<div style="background:#f0f2f5;border-left:3px solid #4dd068;border-radius:0 6px 6px 0;padding:6px 10px;margin-bottom:4px;font-size:11px;color:#1f2937">{_he.escape(str(s))}</div>', unsafe_allow_html=True)
                        if imprv:
                            st.markdown('<div style="font-size:11px;font-weight:700;color:#ea580c;margin-bottom:6px;margin-top:8px">💡 개선 제안</div>', unsafe_allow_html=True)
                            for i, tip in enumerate(imprv):
                                if isinstance(tip, dict):
                                    issue  = tip.get("issue", "")
                                    action = tip.get("action", "")
                                    st.markdown(
                                        f'<div style="background:#f0f2f5;border-left:3px solid #ff9944;border-radius:0 6px 6px 0;padding:8px 12px;margin-bottom:6px">'
                                        f'<div style="font-size:10px;color:#ea580c;font-weight:600;margin-bottom:3px">문제 {i+1}: {_he.escape(str(issue))}</div>'
                                        f'<div style="font-size:11px;color:#1f2937">&#8594; {_he.escape(str(action))}</div>'
                                        f'</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div style="font-size:11px;color:#1f2937;padding:4px 0">{i+1}. {_he.escape(str(tip))}</div>', unsafe_allow_html=True)
                    else:
                        # 완전히 파싱 불가 → 재시도 버튼 + raw 텍스트 fallback
                        st.markdown(
                            '<div style="background:#fff7ed;border:1px solid rgba(234,88,12,.2);'
                            'border-radius:8px;padding:12px;font-size:11px;margin-bottom:10px">'
                            '<div style="color:#ea580c;font-weight:600;margin-bottom:4px">⚠️ 분석 결과 파싱 실패</div>'
                            '<div style="color:#6b7280">Gemini 응답을 구조화하지 못했습니다. 재시도해주세요.</div>'
                            '</div>',
                            unsafe_allow_html=True)
                        if raw:
                            with st.expander("📄 원본 응답 보기"):
                                st.text(raw[:1000])
                        if st.button("🔄 재분석", key=f"retry_{prefix}_{sel['id']}", use_container_width=True):
                            st.session_state[f"{prefix}_thumb_analysis"] = None
                            st.rerun()

                # ── 버튼 2개: 개선 제작 / 저장 ──
                _btn1, _btn2 = st.columns(2)
                with _btn1:
                    if st.button("🎨 개선 썸네일 제작하기",
                                 key=f"send_to_gen_{prefix}_{sel['id']}",
                                 use_container_width=True):
                        _hints  = [i.get("prompt_hint","") for i in imprv if isinstance(i,dict) and i.get("prompt_hint")]
                        _issues = [i.get("issue","")       for i in imprv if isinstance(i,dict) and i.get("issue")]
                        st.session_state["thumb_analysis_queue"] = {
                            "video":        sel,
                            "analysis":     d,
                            "domain":       domain_label,
                            "bad_points":   _issues,
                            "act_points":   _hints,
                            "channel":      ch,
                            "prompt_hints": _hints,
                        }
                        st.session_state["jump_to_improve"] = True
                        st.rerun()
                with _btn2:
                    if st.button("💾 분석 결과 저장",
                                 key=f"save_ana_{prefix}_{sel['id']}",
                                 use_container_width=True):
                        _issues_s = [i.get("issue","")  for i in imprv if isinstance(i,dict) and i.get("issue")]
                        _actions_s= [i.get("action","") for i in imprv if isinstance(i,dict) and i.get("action")]
                        _elems_s  = d.get("elements", {})
                        _strns_s  = d.get("strengths", [])
                        _bench_s  = d.get("benchmark_comparison", d.get("benchmark", {}))
                        _ana_md = "\n".join(filter(None, [
                            f"# 썸네일 분석 결과",
                            f"> {sel['title'][:60]}",
                            f"> 조회수 {fmt_num(sel['views'])} | {domain_label}",
                            "",
                            "## 시각적 강점",
                            *[f"- {s}" for s in _strns_s],
                            "",
                            "## 개선 제안",
                            *[f"{idx+1}. {iss} -> {act}" for idx,(iss,act) in enumerate(zip(_issues_s,_actions_s))],
                            "",
                            "## 업종 기준 비교",
                            *[f"- {k}: {v}" for k,v in _bench_s.items()],
                        ]))
                        _good_html = "".join([f'<div class="good-point" style="font-size:11px">{s}</div>' for s in _strns_s])
                        _bad_html  = "".join([f'<div class="bad-point" style="font-size:11px">{iss} &rarr; {act}</div>' for iss,act in zip(_issues_s,_actions_s)])
                        _report_html = (
                            f'<div class="report-container">'
                            f'<div class="report-h1">썸네일 분석 결과</div>'
                            f'<div style="font-size:11px;color:#9ca3af;margin-bottom:12px">{sel["title"][:60]}<br>'
                            f'조회수 {fmt_num(sel["views"])} | {domain_label}</div>'
                            f'<div class="report-h2" style="color:#4dd068">시각적 강점</div>'
                            f'{_good_html}'
                            f'<div class="report-h2" style="color:#ff7020">개선 제안</div>'
                            f'{_bad_html}'
                            f'</div>'
                        )
                        st.session_state.saved_items.append({
                            "id":           int(time.time()*1000),
                            "type":         "guideline",
                            "domain":       domain_label,
                            "title":        f"썸네일 분석 {sel['title'][:25]}",
                            "channel_name": ch.get("name",""),
                            "subscribers":  ch.get("subscribers", 0),
                            "saved_at":     datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "summary":      _ana_md,
                            "report_html":  _report_html,
                            "top_thumbs":   [sel.get("thumbnail","")],
                            "er_rate":      0,
                            "longform_count": 0,
                        })
                        st.success("분석 결과가 저장함에 저장됐어요!")
            else:
                # 분석 결과 없음 → vertex_project 있으면 즉시 자동 분석 실행
                if st.session_state.get("vertex_project",""):
                    with st.spinner("Gemini Vision으로 썸네일 분석 중..."):
                        try:
                            _proj = st.session_state.get("vertex_project","")
                            _loc  = st.session_state.get("vertex_location","us-central1")
                            result = gemini_gen_thumbnail_analysis(
                                sel.get("thumbnail_hq") or sel.get("thumbnail",""),
                                sel["title"], sel["views"], domain_label, bench,
                                _proj, _loc
                            )
                            st.session_state[f"{prefix}_thumb_analysis"] = {
                                "video_id": sel["id"],
                                "data": result,
                            }
                            st.rerun()
                        except Exception as e:
                            st.error(f"분석 실패: {e}")
                            if st.button("🔄 다시 시도", key=f"retry_ana_{prefix}_{sel['id']}"):
                                st.session_state[f"{prefix}_thumb_analysis"] = None
                                st.rerun()
                else:
                    st.markdown(
                        '<div class="api-notice">💡 사이드바에서 Vertex AI 설정이 필요합니다.</div>',
                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#e8eaed'>", unsafe_allow_html=True)

    # ── 맞춤형 전략 보고서 — 버튼 클릭 시 생성 ──
    _report_key = f"{prefix}_report_requested"
    _top5_key   = f"{prefix}_top5_analysis"
    _proj = st.session_state.get("vertex_project","")
    _loc  = st.session_state.get("vertex_location","us-central1")

    if not st.session_state.get(_report_key):
        # 아직 미생성 → 버튼 표시
        if st.button(
            "📋 맞춤형 썸네일 전략 보고서 생성",
            key=f"gen_report_{prefix}",
            use_container_width=True,
        ):
            st.session_state[_report_key] = True
            # top5 분석도 초기화 후 새로 실행
            st.session_state[_top5_key] = None
            st.rerun()
    else:
        # 생성 요청됨 → top5 썸네일 분석 후 보고서 렌더링
        if _proj and not st.session_state.get(_top5_key):
            _lf_vids = (ana.get("longform_videos") or ana.get("top3") or [])
            if _lf_vids:
                with st.spinner(f"상위 {min(5,len(_lf_vids))}개 썸네일 분석 중... (최대 30초)"):
                    _top5_result = analyze_top5_thumbnails(
                        _lf_vids, bench, domain_label, _proj, _loc
                    )
                    st.session_state[_top5_key] = _top5_result

        top5_data = st.session_state.get(_top5_key) or {}

        with st.expander("📋 맞춤형 썸네일 전략 가이드라인 보고서", expanded=True):
            report_html = build_guideline_report(ch, ana, bench, domain_label, accent, top5_data)
            st.markdown(report_html, unsafe_allow_html=True)
            md_text = build_guideline_markdown(ch, ana, bench, domain_label, top5_data)
            col_dl, col_sv, col_re = st.columns([1, 1, 1])
            with col_dl:
                st.download_button(
                    "⬇ 보고서 다운로드 (.md)",
                    data=md_text.encode("utf-8"),
                    file_name=f"{ch['name']}_{domain_label}_가이드라인_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    key=f"dl_report_{prefix}",
                )
            with col_sv:
                if st.button("💾 저장함에 저장", key=f"sv_report_{prefix}"):
                    top_thumbs = [v.get("thumbnail","") for v in ana.get("top3",[]) if v.get("thumbnail")]
                    st.session_state.saved_items.append({
                        "id": int(time.time()*1000),
                        "type": "guideline",
                        "domain": domain_label,
                        "title": f"{ch['name']} {domain_label} 가이드라인",
                        "channel_name": ch["name"],
                        "subscribers": ch["subscribers"],
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "summary": md_text,
                        "report_html": report_html,
                        "top_thumbs": top_thumbs,
                        "er_rate": ana.get("er_rate", 0),
                        "longform_count": ana.get("longform_count", 0),
                    })
                    st.success("저장 완료!")
            with col_re:
                if st.button("🔄 보고서 재생성", key=f"regen_report_{prefix}"):
                    st.session_state[_top5_key] = None
                    st.rerun()

    st.markdown("<hr style='border-color:#e8eaed'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:14px 0 18px 0;
        border-bottom:1px solid #e8eaed;margin-bottom:18px">
        <div style="width:28px;height:20px;background:#ff0000;border-radius:4px;
            display:flex;align-items:center;justify-content:center;font-size:9px;color:#fff">▶</div>
        <span style="font-size:15px;font-weight:700">Tube<span style="color:#ff0000">Strategy</span></span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;color:#6b7280;margin-bottom:8px;letter-spacing:.8px">API 키 설정</div>', unsafe_allow_html=True)
    
    yt_key = st.secrets["YOUTUBE_API_KEY"]
    vertex_project  = st.secrets["GOOGLE_CLOUD_PROJECT"]
    vertex_location = st.secrets["GOOGLE_CLOUD_REGION"]
    
    st.session_state["vertex_project"] = vertex_project
    st.session_state["vertex_location"] = vertex_location

    # 사이드바에서 API 키 직접 입력하던 부분 
    st.markdown("""<div style="font-size:10px;color:#6b7280;line-height:1.6;margin-top:6px">
    Vertex AI 기능:<br>
    · 프롬프트 자동 생성 (gemini-2.5-flash-lite)<br>
    · 썸네일 비전 분석 (gemini-2.5-flash)<br>
    · 이미지 생성 (imagen-4.0)</div>""", unsafe_allow_html=True)

    # CSV 연결 상태 표시
    st.markdown("<hr style='border-color:#e5e7eb;margin:14px 0'>", unsafe_allow_html=True)
    if _csv_fnb:
        st.markdown(
            f'<div style="background:rgba(22,163,74,.06);border:1px solid rgba(22,163,74,.25);'
            f'border-radius:6px;padding:7px 10px;font-size:10px;color:#16a34a">'
            f'📂 데이터 연결됨<br>'
            f'FnB 성공 {_csv_fnb.get("_n_success",0)}개 · IT 성공 {_csv_it.get("_n_success",0)}개</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:rgba(161,98,7,.05);border:1px solid rgba(255,214,0,.25);'
            'border-radius:6px;padding:7px 10px;font-size:10px;color:#d97706">'
            f'⚠️ CSV 미연결 — 하드코딩 기준값 사용<br>'
            f'<span style="color:#888">app.py와 같은 폴더에 all_thumbnail.csv 위치</span></div>',
            unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#e8eaed;margin:14px 0'>", unsafe_allow_html=True)
    n = len(st.session_state.saved_items)
    st.markdown(f'<div style="font-size:11px;color:#6b7280">💾 저장 항목: <span style="color:#ea580c;font-weight:700">{n}개</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 탑바
# ══════════════════════════════════════════════
st.markdown(
    '<div style="display:flex;align-items:center;gap:10px;padding:10px 0;'
    'border-bottom:1px solid #e8eaed;margin-bottom:14px">'
    '<span style="font-size:15px;font-weight:600">썸네일 분석</span>'
    '<span style="font-size:12px;color:#6b7280">| 유튜브 기업 마케팅 채널 전략 대시보드</span>'
    '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: 썸네일 분석
# ══════════════════════════════════════════════
if True:

    tab_guide, tab_save = st.tabs([
        "📋  가이드라인",
        "📁  저장된 리스트",
    ])

    # ═══════ 가이드라인 (FnB / IT 통합) ═══════
    with tab_guide:
        guide_domain = st.radio(
            "업종 선택",
            ["🍔 FnB", "💻 IT"],
            horizontal=True,
            key="guide_domain_sel",
            label_visibility="collapsed",
        )
        _guide_is_fnb = guide_domain.startswith("🍔")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if _guide_is_fnb:
            st.markdown('<div class="sec-title"><span class="tbar" style="background:#ff0000"></span>FnB 기업 맞춤형 썸네일 전략 (롱폼 기준)</div>', unsafe_allow_html=True)

            if not yt_key:
                st.markdown('<div class="api-notice">💡 사이드바에서 YouTube Data API v3 키를 입력해주세요.</div>', unsafe_allow_html=True)

            c1, c2 = st.columns([4,1])
            with c1:
                fnb_q = st.text_input("채널명 또는 채널 ID",
                    placeholder="예: CU씨유튜브 / @CUtube / UCxxxxxxxx",
                    key="fnb_q", label_visibility="collapsed")
            with c2:
                fnb_go = st.button("🔍 채널 분석", key="fnb_go")

            if fnb_go:
                if not yt_key:
                    st.error("사이드바에서 YouTube API 키를 입력해주세요")
                elif not fnb_q.strip():
                    st.error("채널명을 입력해주세요")
                else:
                    with st.spinner("공식 채널 후보를 찾는 중..."):
                        try:
                            st.session_state["fnb_channel_candidates"] = yt_search_channel_candidates(fnb_q, yt_key)
                            st.success("채널 후보를 찾았습니다. 아래에서 공식 채널을 선택한 뒤 분석을 실행하세요.")
                        except Exception as e:
                            st.error(f"오류: {e}")

            fnb_candidates = st.session_state.get("fnb_channel_candidates", [])
            if fnb_candidates:
                selected_cid = render_channel_candidate_picker("fnb", fnb_candidates)
                if st.button("✅ 선택한 FnB 채널 분석", key="fnb_analyze_selected"):
                    with st.spinner("선택한 채널 데이터 수집 중 (롱폼/숏폼 구분 포함)..."):
                        try:
                            ch, vids, ana = run_channel_analysis(selected_cid, yt_key, FNB, "FnB")
                            st.session_state.fnb_channel = ch
                            st.session_state.fnb_videos = vids
                            st.session_state.fnb_analysis = ana
                            st.session_state.fnb_guideline = (
                                f"FnB 채널 '{ch['name']}' 롱폼 분석. "
                                "권장: 예능/콘텐츠형, Warm/Neutral 색상, 2인 이상 인물 구성, 브랜드 로고"
                            )
                            lf_cnt = ana.get("longform_count", 0)
                            sf_cnt = ana.get("shortform_count", 0)
                            st.session_state["fnb_report_requested"] = False
                            st.session_state["fnb_top5_analysis"] = None
                            st.success(f"✅ '{ch['name']}' 분석 완료! — 롱폼 {lf_cnt}개 · 숏폼 {sf_cnt}개 감지")
                        except Exception as e:
                            st.error(f"오류: {e}")

            if st.session_state.fnb_channel:
                render_channel_result("fnb", "#ff0000", FNB, "FnB")

        else:
            st.markdown('<div class="sec-title"><span class="tbar" style="background:#3ea6ff"></span>IT 기업 맞춤형 썸네일 전략 (롱폼 기준)</div>', unsafe_allow_html=True)

            if not yt_key:
                st.markdown('<div class="api-notice">💡 사이드바에서 YouTube Data API v3 키를 입력해주세요.</div>', unsafe_allow_html=True)

            ic1, ic2 = st.columns([4,1])
            with ic1:
                it_q = st.text_input("채널명 또는 채널 ID",
                    placeholder="예: 삼성SDS / @samsungsds / UCxxxxxxxx",
                    key="it_q", label_visibility="collapsed")
            with ic2:
                it_go = st.button("🔍 채널 분석", key="it_go")

            if it_go:
                if not yt_key:
                    st.error("사이드바에서 YouTube API 키를 입력해주세요")
                elif not it_q.strip():
                    st.error("채널명을 입력해주세요")
                else:
                    with st.spinner("공식 채널 후보를 찾는 중..."):
                        try:
                            st.session_state["it_channel_candidates"] = yt_search_channel_candidates(it_q, yt_key)
                            st.success("채널 후보를 찾았습니다. 아래에서 공식 채널을 선택한 뒤 분석을 실행하세요.")
                        except Exception as e:
                            st.error(f"오류: {e}")

            it_candidates = st.session_state.get("it_channel_candidates", [])
            if it_candidates:
                selected_cid = render_channel_candidate_picker("it", it_candidates)
                if st.button("✅ 선택한 IT 채널 분석", key="it_analyze_selected"):
                    with st.spinner("선택한 채널 데이터 수집 중 (롱폼/숏폼 구분 포함)..."):
                        try:
                            ch, vids, ana = run_channel_analysis(selected_cid, yt_key, IT, "IT")
                            st.session_state.it_channel = ch
                            st.session_state.it_videos = vids
                            st.session_state.it_analysis = ana
                            st.session_state.it_guideline = (
                                f"IT 채널 '{ch['name']}' 롱폼 분석. "
                                "권장: 정보 전달형, Neutral/Cool 색상, 전문가 인물, 핵심 텍스트"
                            )
                            lf_cnt = ana.get("longform_count", 0)
                            sf_cnt = ana.get("shortform_count", 0)
                            st.session_state["it_report_requested"] = False
                            st.session_state["it_top5_analysis"] = None
                            st.success(f"✅ '{ch['name']}' 분석 완료! — 롱폼 {lf_cnt}개 · 숏폼 {sf_cnt}개 감지")
                        except Exception as e:
                            st.error(f"오류: {e}")

            if st.session_state.it_channel:
                render_channel_result("it", "#3ea6ff", IT, "IT")



        st.markdown("<hr style='border-color:#e8eaed;margin:20px 0'>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="tbar" style="background:#2ba640"></span>AI 썸네일 제작</div>', unsafe_allow_html=True)

        # 현재 선택된 도메인 가이드라인 자동 적용
        if _guide_is_fnb and st.session_state.fnb_channel and st.session_state.fnb_analysis:
            applied_guide = {
                "text": st.session_state.fnb_guideline or "",
                "bad": st.session_state.fnb_analysis.get("bad",[]),
                "act": st.session_state.fnb_analysis.get("act",[]),
                "domain": "FnB",
            }
        elif not _guide_is_fnb and st.session_state.it_channel and st.session_state.it_analysis:
            applied_guide = {
                "text": st.session_state.it_guideline or "",
                "bad": st.session_state.it_analysis.get("bad",[]),
                "act": st.session_state.it_analysis.get("act",[]),
                "domain": "IT",
            }
        else:
            applied_guide = None
            st.markdown('<div class="api-notice">💡 위에서 채널을 분석하면 가이드라인이 자동으로 적용됩니다.</div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#e5e7eb'>", unsafe_allow_html=True)

        # ─── 서브 탭: 분석 기반 개선 / 새 썸네일 제작 ───
        sub_improve, sub_new = st.tabs(["🔧  분석 기반 개선", "✨  새로운 썸네일 제작"])

        # ──── 분석 기반 개선 탭 ────
        with sub_improve:
            st.markdown(
                '<div style="font-size:12px;color:#9ca3af;margin-bottom:12px">'
                '기존 썸네일을 선택하거나 업로드하면 <strong style="color:#1f2937">Gemini가 자동으로 분석</strong>하고, '
                '수정 지시를 입력하면 개선된 썸네일을 생성합니다.</div>',
                unsafe_allow_html=True)

            # ── 기존 썸네일 소스 선택 ──
            src_mode = st.radio("기존 썸네일 가져오기",
                ["분석한 영상에서 선택", "YouTube URL 입력"],
                horizontal=True, key="imp_src_mode")

            # 분석된 영상 목록에서 선택
            _ref_thumb_url = ""
            _ref_thumb_img = None
            _ref_title = ""
            _ref_analysis = None

            if src_mode == "분석한 영상에서 선택":
                # FnB/IT 분석된 영상 목록 합치기
                all_lf = []
                if st.session_state.fnb_videos:
                    for v in st.session_state.fnb_videos:
                        if v.get("verdict") in ("longform","unknown"):
                            all_lf.append({"label": f"[FnB] {v['title'][:40]}", "v": v, "domain": "FnB"})
                if st.session_state.it_videos:
                    for v in st.session_state.it_videos:
                        if v.get("verdict") in ("longform","unknown"):
                            all_lf.append({"label": f"[IT] {v['title'][:40]}", "v": v, "domain": "IT"})

                queue = st.session_state.get("thumb_analysis_queue")
                if queue:
                    _ref_thumb_url = queue["video"].get("thumbnail_hq","")
                    _ref_title     = queue["video"]["title"]
                    _ref_analysis  = queue.get("analysis")
                    _ref_analysis_domain = queue.get("domain","FnB")
                    if _ref_thumb_url:
                        st.markdown(f'<div style="font-size:11px;color:#2563eb;margin-bottom:8px">📌 FnB/IT 탭에서 전달된 썸네일: {_ref_title[:50]}</div>', unsafe_allow_html=True)
                elif all_lf:
                    sel_idx = st.selectbox("영상 선택", range(len(all_lf)),
                        format_func=lambda i: all_lf[i]["label"], key="imp_vid_sel")
                    _sel_v = all_lf[sel_idx]["v"]
                    _ref_thumb_url = _sel_v.get("thumbnail_hq","")
                    _ref_title     = _sel_v["title"]
                    _ref_analysis_domain = all_lf[sel_idx]["domain"]
                    pfx = "fnb" if _ref_analysis_domain=="FnB" else "it"
                    cached_ana = st.session_state.get(f"{pfx}_thumb_analysis")
                    if cached_ana and cached_ana.get("video_id") == _sel_v["id"]:
                        _ref_analysis = cached_ana.get("data")
                else:
                    st.markdown('<div class="api-notice">💡 FnB/IT 탭에서 채널을 먼저 분석해주세요.</div>', unsafe_allow_html=True)
                    _ref_analysis_domain = "FnB"

            else:  # YouTube URL 입력
                _ref_analysis_domain = st.selectbox("도메인", ["FnB", "IT"], key="imp_url_domain")
                yt_url_input = st.text_input(
                    "YouTube 영상 URL",
                    placeholder="예: https://www.youtube.com/watch?v=xxxxxxxxxxx",
                    key="imp_yt_url",
                )

                def _extract_video_id_from_url(url: str) -> str:
                    """YouTube URL에서 video ID 추출"""
                    url = url.strip()
                    # youtu.be 단축 URL
                    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
                    if m: return m.group(1)
                    # youtube.com/watch?v=
                    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
                    if m: return m.group(1)
                    # youtube.com/shorts/
                    m = re.search(r"/shorts/([A-Za-z0-9_-]{11})", url)
                    if m: return m.group(1)
                    return ""

                def _fetch_yt_thumbnail(video_id: str) -> tuple[str, str]:
                    """YouTube video ID → (thumbnail_url, title)"""
                    # 썸네일 URL: maxresdefault 우선, 없으면 hqdefault
                    thumb_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
                    # maxresdefault 존재 확인
                    try:
                        r = requests.head(thumb_url, timeout=5)
                        if r.status_code != 200:
                            thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                    except Exception:
                        thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

                    # 제목은 YouTube oEmbed API로 가져오기 (API 키 불필요)
                    title = video_id
                    try:
                        oembed = requests.get(
                            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
                            timeout=5,
                        )
                        if oembed.ok:
                            title = oembed.json().get("title", video_id)
                    except Exception:
                        pass
                    return thumb_url, title

                if yt_url_input.strip():
                    _vid_id = _extract_video_id_from_url(yt_url_input)
                    if _vid_id:
                        # URL이 바뀌었을 때만 재조회
                        if st.session_state.get("_imp_last_vid_id") != _vid_id:
                            with st.spinner("썸네일 불러오는 중..."):
                                _thumb_url, _vid_title = _fetch_yt_thumbnail(_vid_id)
                                st.session_state["_imp_last_vid_id"]    = _vid_id
                                st.session_state["_imp_last_thumb_url"] = _thumb_url
                                st.session_state["_imp_last_title"]     = _vid_title
                        _ref_thumb_url = st.session_state.get("_imp_last_thumb_url", "")
                        _ref_title     = st.session_state.get("_imp_last_title", _vid_id)
                        st.markdown(
                            f'<div class="guide-hint">✅ 영상 ID: <b>{_vid_id}</b><br>'
                            f'<span style="font-size:11px">{_ref_title[:60]}</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown('<div class="api-notice">⚠️ 올바른 YouTube URL을 입력해주세요.<br>예: https://www.youtube.com/watch?v=xxxx 또는 https://youtu.be/xxxx</div>', unsafe_allow_html=True)

            # ── 기존 썸네일 미리보기 + 자동 분석 + 수정 지시 ──
            if _ref_thumb_url or _ref_thumb_img:
                col_ref, col_edit = st.columns([1, 2])
                with col_ref:
                    st.markdown('<div style="font-size:11px;color:#9ca3af;margin-bottom:4px">📌 기존 썸네일</div>', unsafe_allow_html=True)
                    if _ref_thumb_url:
                        st.image(_ref_thumb_url, use_container_width=True)
                    elif _ref_thumb_img:
                        st.image(_ref_thumb_img, use_container_width=True)
                    if _ref_title:
                        st.markdown(f'<div style="font-size:9px;color:#9ca3af;margin-top:3px">{_ref_title[:45]}...</div>', unsafe_allow_html=True)

                    # ── 썸네일 자동 분석 버튼 ──
                    has_proj = bool(st.session_state.get("vertex_project",""))
                    if st.button("🔍 썸네일 자동 분석 + 프롬프트 생성", key="imp_auto_analyze",
                                 disabled=not has_proj, use_container_width=True,
                                 help="Gemini Vision이 기존 썸네일을 분석하고 개선 프롬프트를 자동 작성합니다"):
                        with st.spinner("① Gemini Vision 분석 중... ② 프롬프트 자동 생성 중..."):
                            try:
                                _p = st.session_state.get("vertex_project","")
                                _l = st.session_state.get("vertex_location","us-central1")
                                _bench = FNB if _ref_analysis_domain=="FnB" else IT

                                # URL만 사용 (YouTube URL 모드는 항상 URL)
                                _url_for_ana = _ref_thumb_url

                                # ── 분석 실행 ──
                                ana_result = gemini_gen_thumbnail_analysis(
                                    _url_for_ana, _ref_title, 0,
                                    _ref_analysis_domain, _bench, _p, _l
                                )
                                st.session_state["imp_auto_analysis"] = ana_result

                                # ── 분석 결과에서 정보 추출 ──
                                _elems  = ana_result.get("elements", {})
                                _bench_ = ana_result.get("benchmark", {})
                                _strgs  = ana_result.get("strengths", [])
                                _imprv  = ana_result.get("improvements", [])
                                _hints  = ana_result.get("prompt_hints", [])

                                # ① 현재 이미지 현황 요약
                                _current_parts = []
                                if _elems.get("main_objects"):
                                    _current_parts.append(f"main subject: {_elems['main_objects']}")
                                if _elems.get("color_palette"):
                                    _current_parts.append(f"colors: {_elems['color_palette']}")
                                if _elems.get("person_details"):
                                    _current_parts.append(f"person: {_elems['person_details']}")
                                if _elems.get("brand_elements") and _elems["brand_elements"] != "없음":
                                    _current_parts.append(f"brand: {_elems['brand_elements']}")
                                _current_desc = ", ".join(_current_parts[:4]) if _current_parts else ""

                                # ② 강점 (유지할 것)
                                _keep = _strgs[0] if _strgs else ""

                                # ③ 개선 방향 (이미지 생성용 영어 힌트)
                                _improve_parts = []
                                for tip in _imprv[:3]:
                                    if isinstance(tip, dict) and tip.get("prompt_hint"):
                                        _improve_parts.append(tip["prompt_hint"])
                                    elif isinstance(tip, str):
                                        _improve_parts.append(tip)
                                _improve_desc = "; ".join(_improve_parts) if _improve_parts else "improve overall visual quality"

                                # ④ CTR 정보
                                _ctr = _bench_.get("overall_ctr","")

                                # ── 최종 프롬프트 구성 ──
                                _domain_ctx = (
                                    "Korean FnB food & beverage brand, warm appetizing style"
                                    if _ref_analysis_domain == "FnB"
                                    else "Korean IT tech company, professional clean style"
                                )

                                _auto_prompt = (
                                    f"Improved YouTube thumbnail for {_domain_ctx}. "
                                    + (f"Original thumbnail features: {_current_desc}. " if _current_desc else "")
                                    + (f"Preserve strengths: {_keep}. " if _keep else "")
                                    + f"Apply these improvements: {_improve_desc}. "
                                    + f"High quality professional photography, 16:9 aspect ratio, "
                                    + f"Korean brand thumbnail aesthetic, strong visual hook"
                                )

                                st.session_state["imp_prompt"] = _auto_prompt
                                st.rerun()
                            except Exception as e:
                                st.error(f"분석 실패: {e}")

                    # 분석 결과 표시
                    _auto_analysis = st.session_state.get("imp_auto_analysis") or _ref_analysis
                    if _auto_analysis and isinstance(_auto_analysis, dict):
                        imprv_list = _auto_analysis.get("improvements",[])
                        if imprv_list:
                            st.markdown('<div style="font-size:10px;color:#ea580c;font-weight:600;margin:8px 0 4px">💡 분석된 개선점</div>', unsafe_allow_html=True)
                            for tip in imprv_list[:3]:
                                issue  = tip.get("issue","") if isinstance(tip,dict) else str(tip)
                                action = tip.get("action","") if isinstance(tip,dict) else ""
                                st.markdown(
                                    f'<div style="background:#f0f2f5;border-left:3px solid #ff9944;border-radius:0 5px 5px 0;padding:5px 8px;margin-bottom:4px">' 
                                    f'<div style="color:#ff7020;font-size:9px;font-weight:600">{issue}</div>' 
                                    f'<div style="color:#374151;font-size:9px;margin-top:2px">→ {action}</div></div>',
                                    unsafe_allow_html=True)

                with col_edit:
                    st.markdown('<div style="font-size:11px;color:#9ca3af;margin-bottom:4px">✏️ 수정하고 싶은 내용을 직접 입력하세요</div>', unsafe_allow_html=True)
                    imp_keywords = st.text_area(
                        "수정 지시",
                        placeholder="예: 인물 얼굴을 더 크게 클로즈업\n배경을 단순하게 정리\n브랜드 로고 오른쪽 하단에 배치\n전체적으로 더 밝고 따뜻한 색감으로",
                        key="imp_kw", height=120, label_visibility="collapsed")

                    imp_kw_c1, imp_kw_c2 = st.columns([1,1])
                    with imp_kw_c1:
                        imp_auto = st.button("✨ 프롬프트 자동생성", key="imp_auto",
                                            disabled=not st.session_state.get("vertex_project",""),
                                            use_container_width=True)
                    with imp_kw_c2:
                        if st.button("🗑 초기화", key="imp_clear", use_container_width=True):
                            st.session_state["thumb_analysis_queue"] = None
                            st.session_state["imp_auto_analysis"] = None
                            st.session_state["imp_prompt"] = ""
                            st.rerun()

                # 프롬프트 자동생성 버튼 (Gemini + 분석 결과 + 사용자 수정 지시 통합)
                if imp_auto:
                    _proj = st.session_state.get("vertex_project","")
                    _loc  = st.session_state.get("vertex_location","us-central1")
                    _cur_ana = st.session_state.get("imp_auto_analysis") or _ref_analysis

                    try:
                        _elems = (_cur_ana or {}).get("elements", {})
                        _imprv = (_cur_ana or {}).get("improvements", [])
                        _strgs = (_cur_ana or {}).get("strengths", [])

                        current_summary = ", ".join(filter(None, [
                            _elems.get("main_objects", ""),
                            _elems.get("color_palette", ""),
                            _elems.get("person_details", ""),
                            _elems.get("brand_elements", ""),
                        ]))[:400]

                        improvement_hints = []
                        for tip in _imprv[:5]:
                            if isinstance(tip, dict):
                                if tip.get("prompt_hint"):
                                    improvement_hints.append(tip["prompt_hint"])
                                elif tip.get("action"):
                                    improvement_hints.append(tip["action"])
                            elif isinstance(tip, str):
                                improvement_hints.append(tip)

                        with st.spinner("Gemini로 개선 프롬프트 생성 중..."):
                            _new_prompt = gemini_gen_improvement_prompt(
                                user_instruction=imp_keywords.strip(),
                                domain=_ref_analysis_domain,
                                current_summary=current_summary,
                                strengths=_strgs,
                                improvement_hints=improvement_hints,
                                project_id=_proj,
                                location=_loc,
                            )

                        st.session_state["imp_prompt"] = _new_prompt
                        st.rerun()

                    except Exception as e:
                        st.error(f"프롬프트 자동생성 실패: {e}")

                # 프롬프트 초기값 — 분석 결과 있으면 그걸 기반으로, 없으면 기본값
                imp_is_fnb = (_ref_analysis_domain == "FnB")
                imp_dk = _ref_analysis_domain
                _cur_ana_for_default = st.session_state.get("imp_auto_analysis") or _ref_analysis

                if _cur_ana_for_default and isinstance(_cur_ana_for_default, dict):
                    _e = _cur_ana_for_default.get("elements", {})
                    _i = _cur_ana_for_default.get("improvements", [])
                    _s = _cur_ana_for_default.get("strengths", [])
                    _current_d = ", ".join(filter(None, [
                        _e.get("main_objects",""), _e.get("color_palette",""), _e.get("person_details","")
                    ]))[:180]
                    _hints_d = [t.get("action","") for t in _i[:3] if isinstance(t,dict) and t.get("action")]
                    _keep_d  = _s[0] if _s else ""
                    _domain_d = "FnB 푸드&베버리지 브랜드" if imp_is_fnb else "IT 기업"
                    _imp_default = (
                        f"한국 {_domain_d} 유튜브 썸네일 개선. "
                        + (f"현재 구성: {_current_d}. " if _current_d else "")
                        + (f"유지할 강점: {_keep_d}. " if _keep_d else "")
                        + (f"개선 사항: {'; '.join(_hints_d)}. " if _hints_d else "")
                        + ("따뜻하고 식욕을 돋우는 색감, 선명한 음식 사진, 프로페셔널 구도" if imp_is_fnb
                           else "깔끔하고 전문적인 디자인, 밝은 배경, 핵심 메시지 강조")
                    )
                else:
                    _imp_default = (
                        f"한국 {'FnB 브랜드 — 따뜻하고 식욕을 돋우는 색감, 선명한 음식 사진' if imp_is_fnb else 'IT 기업 — 깔끔하고 전문적인 디자인, 밝은 색감'} "
                        f"스타일의 유튜브 썸네일 개선. 인물 구도 강화, 브랜드 로고 배치, 시선을 끄는 비주얼 후킹 요소 추가"
                    )

                if "imp_prompt" not in st.session_state or not st.session_state["imp_prompt"]:
                    st.session_state["imp_prompt"] = _imp_default

                imp_final_prompt = st.text_area(
                    "개선 지시 (한국어로 작성 → 자동 번역 후 생성)",
                    key="imp_prompt", height=100,
                    help="한국어로 자유롭게 작성하세요. 생성 시 Gemini가 영어로 번역해 Imagen에 전달합니다. 이미지 안에 텍스트는 렌더링되지 않습니다.")

                imp_gen_btn = st.button("🔧 개선 썸네일 생성", key="imp_gen",
                                       disabled=not st.session_state.get("vertex_project",""),
                                       use_container_width=True)

                if imp_gen_btn:
                    if not _ref_thumb_url:
                        st.error("썸네일을 먼저 선택하거나 YouTube URL을 입력하세요.")
                    else:
                        _proj = st.session_state.get("vertex_project","")
                        _loc  = st.session_state.get("vertex_location","us-central1")

                        # 분석 힌트 추출
                        _cur_analysis = st.session_state.get("imp_auto_analysis") or _ref_analysis
                        _hints_list = []
                        if _cur_analysis:
                            _hints_list = _cur_analysis.get("prompt_hints",[])
                            if not _hints_list:
                                for tip in _cur_analysis.get("improvements",[]):
                                    if isinstance(tip,dict) and tip.get("prompt_hint"):
                                        _hints_list.append(tip["prompt_hint"])
                        analysis_hints = "; ".join(_hints_list[:3]) if _hints_list else ""

                        edit_instruction = imp_keywords.strip() or imp_final_prompt.strip()

                        with st.spinner("① 기존 썸네일 스타일 분석 중... ② 개선 이미지 생성 중... (최대 60초)"):
                            try:
                                img_bytes, used_prompt = gemini_edit_image(
                                    _ref_thumb_url,
                                    edit_instruction,
                                    analysis_hints,
                                    _ref_analysis_domain,
                                    _proj, _loc,
                                )
                                img = Image.open(BytesIO(img_bytes))
                                st.session_state.generated_thumb = {
                                    "bytes": img_bytes, "image": img,
                                    "prompt": used_prompt,
                                    "domain": _ref_analysis_domain,
                                    "category": "스타일 개선",
                                    "mode": "개선",
                                    "source_thumb": _ref_thumb_url,
                                    "source_title": _ref_title,
                                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                }
                                st.success("✅ 개선 썸네일 생성 완료!")
                                with st.expander("📄 사용된 최종 프롬프트 보기"):
                                    st.code(used_prompt, language="text")
                                st.rerun()
                            except Exception as e:
                                st.error(f"생성 실패: {e}")
            else:
                if src_mode == "분석한 영상에서 선택" and not st.session_state.fnb_videos and not st.session_state.it_videos:
                    pass
                elif src_mode == "YouTube URL 입력":
                    st.markdown('<div class="api-notice">💡 YouTube 영상 URL을 입력하면 썸네일을 자동으로 불러옵니다.</div>', unsafe_allow_html=True)

        # ──── 새로운 썸네일 제작 탭 ────
        with sub_new:
            st.markdown('<div style="font-size:12px;font-weight:600;color:#1f2937;margin-bottom:10px">새로운 컨셉으로 썸네일을 제작합니다</div>', unsafe_allow_html=True)

            g1,g2,g3 = st.columns(3)
            with g1: t_domain   = st.selectbox("도메인",["FnB (식음료)","IT (기술/소프트웨어)"], key="t_domain")
            with g2: t_category = st.selectbox("썸네일 카테고리",["정보 전달형","예능/콘텐츠형","인터뷰/인물형","브랜드 이미지형","리뷰/비교형","제품 홍보형"], key="t_cat")
            with g3: t_color    = st.selectbox("색상 톤",["warm (따뜻한)","cool (차가운)","neutral (중간)"], key="t_color")
            is_fnb = "FnB" in t_domain
            dk = "FnB" if is_fnb else "IT"

            kw1, kw2 = st.columns([4,1])
            with kw1:
                keywords = st.text_input("핵심 키워드", placeholder="예: 신제품 도시락, 여름 한정 메뉴, 콜라보 이벤트", key="keywords", label_visibility="collapsed")
            with kw2:
                auto_btn = st.button("✨ 자동생성", key="auto_btn", disabled=not st.session_state.get("vertex_project",""))

            chips = (["신제품 리뷰","편의점 콜라보","여름 한정","인기 상품 TOP5","매운맛 챌린지","MD 추천"]
                     if is_fnb else ["AI 솔루션","클라우드 전환","신제품 발표","기술 트렌드","전문가 인터뷰","디지털 혁신"])
            st.markdown('<div style="margin-bottom:8px">' +
                " ".join([f'<span style="display:inline-block;background:#f3f4f6;border:1px solid #d1d5db;border-radius:50px;padding:3px 10px;font-size:11px;color:#4b5563;margin:2px">{c}</span>' for c in chips]) +
                '</div>', unsafe_allow_html=True)

            if auto_btn:
                if not keywords.strip():
                    st.error("키워드를 먼저 입력해주세요")
                else:
                    with st.spinner("Gemini 프롬프트 생성 중..."):
                        try:
                            ch_info = st.session_state.fnb_channel if is_fnb else st.session_state.it_channel
                            _proj = st.session_state.get("vertex_project","")
                            _loc  = st.session_state.get("vertex_location","us-central1")
                            en_p, ko_p = gemini_gen_prompt(keywords, dk, t_category, t_color, ch_info, _proj, _loc)
                            # 내부적으로 영어 프롬프트 보관, UI에는 한국어 요약 표시
                            st.session_state["generated_prompt"] = en_p   # 실제 생성에 사용
                            st.session_state["final_prompt_ko"] = ko_p    # UI 표시용
                            st.session_state["_prompt_source"] = "auto"   # 큐/기본값 덮어쓰기 방지
                            st.session_state["_last_fb"] = (dk, t_category)
                            st.success("✅ 프롬프트 생성 완료!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"프롬프트 생성 실패: {e}")

            _fb = {
                ("FnB","정보 전달형"):   "편의점 신제품 소개 썸네일. 음식이 전면에 선명하게 배치되고, 따뜻하고 밝은 조명, 브랜드 로고 포함, 프로페셔널 푸드 사진",
                ("FnB","예능/콘텐츠형"): "한국 FnB 브랜드 예능 콘텐츠 썸네일. 음식에 반응하는 두 명의 인물, 편의점 배경, 따뜻하고 높은 대비, 생동감 있는 구도",
                ("IT","정보 전달형"):    "한국 IT 기업 정보 전달형 썸네일. 차분한 블루 계열 색감, 디지털 배경, 신뢰감 있는 전문적 디자인",
                ("IT","예능/콘텐츠형"):  "한국 IT 브랜드 유튜브 썸네일. 현대적인 사무실에서 테크 전문가, 쿨한 색상, 역동적이고 전문적인 구도",
            }
            fb = _fb.get((dk, t_category), f"한국 {dk} 브랜드 유튜브 썸네일. 시선을 끄는 전문적 구도, 고품질")

            # ── 텍스트박스 초기값 설정 로직 ──
            # 우선순위: ① 자동생성 결과 > ② 분석 큐 > ③ 도메인/카테고리 기본값
            # 자동생성 후에는 _prompt_source="auto"로 표시해 큐/기본값이 덮어쓰지 못하게 보호
            _prompt_source = st.session_state.get("_prompt_source", "")

            if _prompt_source == "auto":
                # 자동생성 결과 보호 — 도메인/카테고리가 바뀔 때만 초기화
                if st.session_state.get("_last_fb") != (dk, t_category):
                    st.session_state["final_prompt_ko"] = fb
                    st.session_state["generated_prompt"] = ""
                    st.session_state["_prompt_source"] = ""
                    st.session_state["_last_fb"] = (dk, t_category)

            else:
                # 분석 큐에서 영상 특징 가져오기 (개선 썸네일 제작하기 버튼으로 전달된 경우)
                _queue = st.session_state.get("thumb_analysis_queue")
                if _queue and _queue.get("analysis"):
                    _qa  = _queue["analysis"]
                    _qe  = _qa.get("elements", {})
                    _qs  = _qa.get("strengths", [])
                    _qi  = _qa.get("improvements", [])
                    _qac = [i.get("action","") for i in _qi if isinstance(i,dict) and i.get("action")]
                    _qvt = _queue.get("video",{}).get("title","")
                    _queue_ko = (
                        f"'{_qvt[:25]}' 영상 특징 기반 신규 썸네일. "
                        + (f"현재 강점: {_qs[0]}. " if _qs else "")
                        + (f"개선 방향: {_qac[0]}. " if _qac else "")
                        + (f"구성: {_qe.get('main_objects','')[:60]}." if _qe.get('main_objects') else "")
                    )
                    if st.session_state.get("_last_fb") != (dk, t_category):
                        st.session_state["final_prompt_ko"] = _queue_ko
                        st.session_state["generated_prompt"] = ""
                        st.session_state["_last_fb"] = (dk, t_category)
                else:
                    if "final_prompt_ko" not in st.session_state or st.session_state.get("_last_fb") != (dk, t_category):
                        st.session_state["final_prompt_ko"] = fb
                        st.session_state["generated_prompt"] = ""
                        st.session_state["_last_fb"] = (dk, t_category)

            final_prompt_ko = st.text_area(
                "썸네일 개념 설명 (한국어 → 자동 번역 후 생성)",
                key="final_prompt_ko", height=100,
                help="한국어로 자유롭게 작성하세요. 분석된 영상의 특징이 자동으로 반영됩니다.")
            final_prompt = final_prompt_ko

            gen_btn = st.button("🎨 새 썸네일 생성하기", key="gen_btn", disabled=not st.session_state.get("vertex_project",""), use_container_width=True)
            if gen_btn:
                if not final_prompt.strip():
                    st.error("프롬프트를 입력해주세요")
                else:
                    _proj = st.session_state.get("vertex_project","")
                    _loc  = st.session_state.get("vertex_location","us-central1")
                    with st.spinner("Vertex AI Imagen으로 이미지 생성 중... (최대 60초)"):
                        try:
                            # 자동생성 버튼으로 만든 영어 프롬프트 있으면 재사용
                            _stored_en = st.session_state.get("generated_prompt","")
                            if _stored_en:
                                _translated = _stored_en
                            else:
                                # 한국어 → 시각 요소 보강 + 영어 변환 (단순 번역 X)
                                _tr_prompt = (
                                    "You are a YouTube thumbnail visual prompt engineer.\n"
                                    "Convert the following Korean thumbnail description into a detailed English image generation prompt.\n"
                                    "- Translate AND expand: add visual details for any specific objects, foods, characters, or brands mentioned.\n"
                                    "- For foods: describe appearance (color, texture, shape) in detail.\n"
                                    "- For characters/IP (e.g. SpongeBob, Pikachu): describe their visual appearance specifically.\n"
                                    "- For brand collabs: mention brand colors and visual identity.\n"
                                    "- Output 80-120 words. Return ONLY the English prompt, no explanation.\n\n"
                                    f"Korean: {final_prompt_ko.strip()}"
                                )
                                _tr_resp = GenerativeModel(GEMINI_PROMPT_MODEL).generate_content(
                                    _tr_prompt,
                                    generation_config=GenerationConfig(
                                        temperature=IMAGE_GEN_CONFIG["translate_temperature"],
                                        max_output_tokens=IMAGE_GEN_CONFIG["translate_max_tokens"],
                                    ),
                                )
                                _translated = _tr_resp.text.strip()
                            ctx = "Korean food beverage brand, warm appetizing" if is_fnb else "Korean IT company, professional corporate"
                            full_p = (
                                f"{_translated}. "
                                f"YouTube thumbnail, {t_category} style, {t_color} color tone, {ctx}. "
                                f"VIVID saturated colors, BRIGHT well-lit, HIGH CONTRAST, "
                                f"subject fills majority of frame, dynamic eye-catching composition, "
                                f"clean background, wide horizontal format. "
                                f"Do not render any text or letters inside the image."
                            )
                            if applied_guide and isinstance(applied_guide, dict):
                                full_p += f" Brand: {applied_guide['text'][:80]}"
                            img_bytes = gemini_gen_image(full_p, _proj, _loc)
                            img = Image.open(BytesIO(img_bytes))
                            st.session_state.generated_thumb = {
                                "bytes": img_bytes, "image": img,
                                "prompt": final_prompt, "domain": dk,
                                "category": t_category, "mode": "신규",
                                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            }
                            st.success("✅ 썸네일 생성 완료!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"생성 실패: {e}")

        # ─── 공통 결과 미리보기 ───
        if st.session_state.generated_thumb:
            t = st.session_state.generated_thumb
            st.markdown("<hr style='border-color:#e5e7eb'>", unsafe_allow_html=True)
            mode_badge = (f'<span class="badge badge-green">{t.get("mode","신규")}</span>'
                          if t.get("mode")=="신규"
                          else f'<span class="badge badge-orange">{t.get("mode","개선")}</span>')
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:#1f2937;margin-bottom:8px">🖼 생성 결과 {mode_badge}</div>', unsafe_allow_html=True)

            # 개선 모드면 before/after 비교
            if t.get("mode") == "개선" and t.get("source_thumb"):
                import base64 as _b64
                _after_b64 = _b64.b64encode(t["bytes"]).decode()
                _src_title = (t.get("source_title") or "")[:35]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
                    f'<div style="flex:1;min-width:0">'
                    f'<div style="font-size:10px;color:#9ca3af;margin-bottom:4px;text-align:center">BEFORE (원본)</div>'
                    f'<img src="{t["source_thumb"]}" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;border:1px solid #e5e7eb">'
                    f'<div style="font-size:9px;color:#9ca3af;margin-top:3px;text-align:center">{_src_title}...</div>'
                    f'</div>'
                    f'<div style="flex-shrink:0;display:flex;align-items:center;justify-content:center;'
                    f'width:36px;height:36px;background:#f3f4f6;border-radius:50%;border:1px solid #d1d5db">'
                    f'<span style="font-size:18px;color:#d97706;line-height:1">&#8594;</span>'
                    f'</div>'
                    f'<div style="flex:1;min-width:0">'
                    f'<div style="font-size:10px;color:#16a34a;margin-bottom:4px;text-align:center">AFTER (AI 개선)</div>'
                    f'<img src="data:image/png;base64,{_after_b64}" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;border:1px solid #2ba640">'
                    f'<div style="font-size:9px;color:#9ca3af;margin-top:3px;text-align:center">{t["generated_at"]}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                p2_col = st.columns([1])[0]
                with p2_col:
                    bc = "badge-red" if t["domain"]=="FnB" else "badge-blue"
                    st.markdown(
                        f'<div class="yt-card" style="margin-top:8px"><div style="font-size:10px;color:#9ca3af;margin-bottom:6px">생성 정보</div>'
                        f'<span class="badge {bc}">{t["domain"]}</span>'
                        f'  <span style="font-size:11px;color:#4b5563">{t["category"]}</span>'
                        f'<div style="font-size:10px;color:#9ca3af;margin-top:6px">{t["generated_at"]} | Vertex AI Imagen</div></div>',
                        unsafe_allow_html=True)
            else:
                p1, p2 = st.columns([3,1])
                with p1:
                    st.image(t["image"], use_container_width=True)
                    st.markdown(f'<div style="font-size:10px;color:#9ca3af;margin-top:4px">{t["generated_at"]} | Vertex AI Imagen</div>', unsafe_allow_html=True)
                with p2:
                    bc = "badge-red" if t["domain"]=="FnB" else "badge-blue"
                    st.markdown(
                        f'<div class="yt-card"><div style="font-size:10px;color:#9ca3af;margin-bottom:8px">생성 정보</div>'
                        f'<span class="badge {bc}">{t["domain"]}</span>'
                        f'<div style="font-size:11px;color:#4b5563;margin-top:6px">{t["category"]}</div></div>',
                        unsafe_allow_html=True)

            # 저장 버튼 (공통)
            sv1, sv2 = st.columns([1,3])
            with sv1:
                buf = BytesIO(); t["image"].save(buf, "PNG")
                st.download_button("⬇ PNG 다운로드", buf.getvalue(),
                    f"thumb_{int(time.time())}.png", "image/png", key="dl_thumb")
            with sv2:
                if st.button("💾 저장함에 저장", key="save_thumb"):
                    buf2 = BytesIO(); t["image"].save(buf2,"PNG")
                    st.session_state.saved_items.append({
                        "id": int(time.time()*1000), "type": "thumbnail",
                        "domain": t["domain"],
                        "title": f"{t.get('mode','신규')} {t['category']} 썸네일",
                        "prompt": t["prompt"],
                        "image_bytes": buf2.getvalue(),
                        "source_thumb": t.get("source_thumb",""),
                        "source_title": t.get("source_title",""),
                        "mode": t.get("mode","신규"),
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
                    st.success("저장 완료!")



    # ═══════ 저장된 리스트 ═══════
    with tab_save:
        st.markdown('<div class="sec-title"><span class="tbar" style="background:#ffe033"></span>저장된 리스트</div>', unsafe_allow_html=True)
        items  = st.session_state.saved_items
        guides = [i for i in items if i["type"]=="guideline"]
        thumbs = [i for i in items if i["type"]=="thumbnail"]

        s1,s2,s3 = st.columns(3)
        s1.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#ffe033">{len(items)}</div><div class="stat-lbl">전체</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#ff5555">{len(guides)}</div><div class="stat-lbl">가이드라인</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#5ab8ff">{len(thumbs)}</div><div class="stat-lbl">썸네일</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if not items:
            st.markdown('<div style="text-align:center;padding:60px 20px;color:#6b7280"><div style="font-size:40px;margin-bottom:10px">📂</div><div style="font-size:14px;color:#9ca3af">저장된 항목이 없습니다</div><div style="font-size:11px;color:#9ca3af;margin-top:6px">FnB/IT 탭 분석 후 가이드라인을 저장하거나,<br>썸네일 제작 탭에서 이미지를 저장하세요</div></div>', unsafe_allow_html=True)
        else:
            flt = st.radio("필터",["전체","가이드라인","썸네일"], horizontal=True, key="sv_flt")
            filtered = items if flt=="전체" else [i for i in items if i["type"]==("guideline" if flt=="가이드라인" else "thumbnail")]
            st.markdown(f'<div style="font-size:11px;color:#9ca3af;margin-bottom:10px">{len(filtered)}개 항목</div>', unsafe_allow_html=True)

            for item in reversed(filtered):
                if item["type"] == "thumbnail":
                    tc1, tc2 = st.columns([2,3])
                    with tc1:
                        if item.get("image_bytes"):
                            st.image(Image.open(BytesIO(item["image_bytes"])), use_container_width=True)
                        # 개선 모드면 원본도 표시
                        if item.get("mode") == "개선" and item.get("source_thumb"):
                            st.markdown('<div style="font-size:9px;color:#9ca3af;margin:4px 0 2px">원본 썸네일</div>', unsafe_allow_html=True)
                            st.image(item["source_thumb"], use_container_width=True)
                            if item.get("source_title"):
                                st.markdown(f'<div style="font-size:9px;color:#9ca3af">{item["source_title"][:30]}...</div>', unsafe_allow_html=True)
                    with tc2:
                        bc = "badge-red" if item["domain"]=="FnB" else "badge-blue"
                        mode_lbl = item.get("mode","신규")
                        st.markdown(
                            f'<div class="yt-card">'
                            f'<div style="font-size:10px;color:#9ca3af;margin-bottom:4px">AI 썸네일 · {mode_lbl}</div>'
                            f'<div style="font-size:13px;font-weight:600;color:#111827;margin-bottom:6px">{item["title"]}</div>'
                            f'<span class="badge {bc}">{item["domain"]}</span>'
                            f'<div style="font-size:11px;color:#9ca3af;margin-top:8px;line-height:1.5">{item.get("prompt","")[:80]}...</div>'
                            f'<div style="font-size:10px;color:#9ca3af;margin-top:6px">💾 {item["saved_at"]}</div>'
                            f'</div>', unsafe_allow_html=True)
                        if item.get("image_bytes"):
                            buf = BytesIO()
                            Image.open(BytesIO(item["image_bytes"])).save(buf,"PNG")
                            st.download_button("⬇ PNG", buf.getvalue(), f"thumb_{item['id']}.png", "image/png", key=f"dl_{item['id']}")

                else:  # guideline
                    bc  = "badge-red" if item["domain"]=="FnB" else "badge-blue"
                    acc = "#ff5555" if item["domain"]=="FnB" else "#5ab8ff"
                    tier_lbl, _ = get_tier(item.get("subscribers",0))

                    with st.expander(f"📋 {item['title']}  ·  {item['saved_at']}", expanded=False):
                        # 헤더 행: 썸네일 스트립 + 메타
                        hdr1, hdr2 = st.columns([3,2])
                        with hdr1:
                            top_thumbs = item.get("top_thumbs", [])
                            if top_thumbs:
                                st.markdown('<div style="font-size:10px;color:#9ca3af;margin-bottom:6px">📹 분석 기준 영상 썸네일</div>', unsafe_allow_html=True)
                                th_cols = st.columns(min(len(top_thumbs), 3))
                                for col, url in zip(th_cols, top_thumbs[:3]):
                                    if url:
                                        col.image(url, use_container_width=True)
                            else:
                                st.markdown(f'<div style="font-size:28px;margin-bottom:4px">{"🍔" if item["domain"]=="FnB" else "💻"}</div>', unsafe_allow_html=True)
                        with hdr2:
                            st.markdown(
                                f'<div style="background:#f3f4f6;border-radius:8px;padding:10px 12px">'
                                f'<div style="font-size:11px;color:#9ca3af;margin-bottom:4px">채널 정보</div>'
                                f'<div style="font-size:13px;font-weight:700;color:#111827">{item.get("channel_name","")}</div>'
                                f'<div style="margin-top:6px"><span class="badge {bc}">{item["domain"]}</span>'
                                f'<span class="badge badge-gray" style="margin-left:4px">{tier_lbl}</span></div>'
                                f'<div style="font-size:11px;color:#6b7280;margin-top:6px">구독자 {fmt_num(item.get("subscribers",0))}'
                                f'  ·  ER {item.get("er_rate",0):.2f}%'
                                f'  ·  롱폼 {item.get("longform_count",0)}개</div>'
                                f'</div>', unsafe_allow_html=True)

                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                        # 보고서 본문
                        if item.get("report_html"):
                            st.markdown(item["report_html"], unsafe_allow_html=True)
                        else:
                            # 레거시 summary 폴백
                            st.markdown(
                                f'<div style="background:#f0f2f5;border-radius:8px;padding:12px;font-size:12px;'
                                f'color:#374151;line-height:1.7;white-space:pre-wrap">{item.get("summary","")}</div>',
                                unsafe_allow_html=True)

                        # 다운로드
                        md = item.get("summary","")
                        if md:
                            dl1, dl2 = st.columns([1,3])
                            with dl1:
                                st.download_button(
                                    "⬇ 보고서 다운로드",
                                    data=md.encode("utf-8"),
                                    file_name=f"{item.get('channel_name','')}_{item['domain']}_가이드라인.md",
                                    mime="text/markdown",
                                    key=f"sv_dl_{item['id']}",
                                )

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            st.markdown("<hr style='border-color:#e5e7eb'>", unsafe_allow_html=True)
            if st.button("🗑 전체 삭제", key="clear_all"):
                st.session_state.saved_items = []
                st.rerun()