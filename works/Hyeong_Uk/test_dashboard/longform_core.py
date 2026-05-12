# ═══════════════════════════════════════════════════════════════════════════
# YouTube Success AI - Long-form 영상 분석 대시보드
# ═══════════════════════════════════════════════════════════════════════════
# [구조 개요]
#  1. 라이브러리 import & 페이지 설정
#  2. CSS 스타일 정의
#  3. 데이터/모델 로드 (캐시)
#  4. 공통 상수 (색상, 한글 매핑 사전)
#  5. 공통 헬퍼 함수
#  6. 상단 가로 네비게이션 (option_menu)
#  7. 페이지별 함수 정의 (page_home, page_domain, ...)
#  8. 라우터 (선택된 메뉴에 따라 함수 호출)
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import textwrap
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# [1] 페이지 기본 설정
# ─────────────────────────────────────────
# layout="wide": 화면 가로 전체 사용
# initial_sidebar_state="collapsed": 사이드바 사용 안 하므로 기본 접힘
st.set_page_config(
    page_title="YouTube Success AI",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# [2] CSS 스타일은 apply_longform_minimal_style()에서 최소만 주입합니다.

# ─────────────────────────────────────────
# [3] 데이터/모델 로드 (Streamlit 캐시 사용)
# ─────────────────────────────────────────
DATA_PATH = "works/Hyeong_Uk/test_dashboard/data"  # 데이터 디렉토리 경로 (프로젝트 구조에 맞게 수정)

@st.cache_data  # 한번 읽은 csv/json은 메모리에 캐싱 → 재실행 시 빠름
def load_data():
    """모든 분석 결과 CSV/JSON 파일을 한 번에 로드"""
    df_all  = pd.read_csv(f"{DATA_PATH}/longform_dashboard_all.csv")
    df_fnb  = pd.read_csv(f"{DATA_PATH}/longform_dashboard_fnb.csv")
    df_it   = pd.read_csv(f"{DATA_PATH}/longform_dashboard_it.csv")
    pred    = pd.read_csv(f"{DATA_PATH}/longform_predictions_by_dataset.csv")
    cat_shap = pd.read_csv(f"{DATA_PATH}/longform_categorical_mean_shap_catboost.csv")
    shap_all = pd.read_csv(f"{DATA_PATH}/longform_shap_grouped_ALL_XGBoost.csv")
    shap_fnb = pd.read_csv(f"{DATA_PATH}/longform_shap_grouped_FnB_CatBoost.csv")
    shap_it  = pd.read_csv(f"{DATA_PATH}/longform_shap_grouped_IT_RandomForest.csv")
    eda_day  = pd.read_csv(f"{DATA_PATH}/longform_eda_success_rate_by_upload_dayofweek.csv")
    eda_hour = pd.read_csv(f"{DATA_PATH}/longform_eda_success_rate_by_upload_hour.csv")
    eda_ctype = pd.read_csv(f"{DATA_PATH}/longform_eda_success_rate_by_cls_content_type.csv")
    eda_cta  = pd.read_csv(f"{DATA_PATH}/longform_eda_success_rate_by_cls_cta_type.csv")
    eda_len  = pd.read_csv(f"{DATA_PATH}/longform_eda_success_rate_by_length_bucket.csv")
    eda_caption = pd.read_csv(f"{DATA_PATH}/longform_eda_success_rate_by_caption.csv")
    with open(f"{DATA_PATH}/longform_eda_stats.json", encoding="utf-8") as f:
        eda_stats = json.load(f)
    with open(f"{DATA_PATH}/longform_strategy_insights.json", encoding="utf-8") as f:
        strategy = json.load(f)
    with open(f"{DATA_PATH}/longform_model_performance.json", encoding="utf-8") as f:
        model_perf = json.load(f)
    with open(f"{DATA_PATH}/longform_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return dict(
        df_all=df_all, df_fnb=df_fnb, df_it=df_it, pred=pred,
        cat_shap=cat_shap, shap_all=shap_all, shap_fnb=shap_fnb, shap_it=shap_it,
        eda_day=eda_day, eda_hour=eda_hour, eda_ctype=eda_ctype, eda_cta=eda_cta,
        eda_len=eda_len, eda_caption=eda_caption,
        eda_stats=eda_stats, strategy=strategy, model_perf=model_perf, metadata=metadata
    )

@st.cache_resource  # 모델 객체는 cache_resource (직렬화 안되는 객체용)
def load_models():
    """학습된 ML 모델 + SHAP explainer 로드"""
    m_fnb = joblib.load(f"{DATA_PATH}/longform_FnB_best_perf_CatBoost.joblib")
    m_it  = joblib.load(f"{DATA_PATH}/longform_IT_best_perf_RandomForest.joblib")
    m_all = joblib.load(f"{DATA_PATH}/longform_ALL_best_perf_CatBoost.joblib")
    shap_fnb = joblib.load(f"{DATA_PATH}/longform_FnB_shap_catboost.joblib")
    shap_it  = joblib.load(f"{DATA_PATH}/longform_IT_shap_catboost.joblib")
    shap_all = joblib.load(f"{DATA_PATH}/longform_ALL_shap_catboost.joblib")
    return dict(m_fnb=m_fnb, m_it=m_it, m_all=m_all,
                shap_fnb=shap_fnb, shap_it=shap_it, shap_all=shap_all)

# 전역 데이터/모델 객체는 탭 렌더링 시 지연 로드합니다.
D = None
M = None


def apply_longform_minimal_style():
    """최종 app.py와 충돌을 줄인 롱폼 탭용 최소 스타일."""
    st.markdown("""
    <style>
    .section-header {
        background: linear-gradient(135deg, #ffffff 0%, #fff7f8 100%);
        color: #111827;
        border: 1px solid #e5e7eb;
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    .section-header h2 { margin: 0; font-size: 22px; font-weight: 900; }
    .section-header p { margin: 6px 0 0; font-size: 13px; color: #6b7280; }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        border-left: 4px solid #ef233c;
        margin-bottom: 12px;
    }
    .metric-title { font-size: 12px; color: #6b7280; font-weight: 700; margin-bottom: 6px; }
    .metric-value { font-size: 26px; font-weight: 900; color: #111827; }
    .metric-sub { font-size: 12px; color: #6b7280; margin-top: 4px; }
    .insight-box { background: #fff1f2; border-left: 4px solid #ef233c; padding: 12px 16px; border-radius: 0 10px 10px 0; font-size: 13px; color: #374151; margin-top: 10px; }
    .insight-green { background: #f0fdf4; border-left-color: #22c55e; color: #166534; }
    .insight-orange { background: #fff7ed; border-left-color: #f97316; color: #9a3412; }
    .concept-box { background:#f9fafb; border:1px solid #e5e7eb; border-radius:14px; padding:12px 16px; margin-bottom:14px; font-size:12px; color:#374151; line-height:1.7; }
    .help-tip { display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px; border-radius:50%; background:#fee2e2; color:#ef233c; font-size:11px; font-weight:900; cursor:help; margin-left:6px; position:relative; vertical-align:middle; }
    .prob-value { font-size: 48px; font-weight: 900; }
    .prob-high { color: #22c55e; }
    .prob-mid { color: #f59e0b; }
    .prob-low { color: #ef4444; }
    </style>
    """, unsafe_allow_html=True)


def ensure_longform_resources(load_model: bool = False):
    """데이터/모델을 탭이 열릴 때만 로드하고, 원본 대시보드의 전역 계산값을 준비합니다."""
    global D, M, df_all, pred, df_recent_30, df_prev_30, recent_30_success, prev_30_success
    if D is None:
        D = load_data()

        # 원본 long_form_app.py.py의 최근 30개 계산 로직을 지연 실행으로 이동
        df_all = D["df_all"].copy()
        pred = D["pred"].copy()
        date_candidates = ["업로드일시", "upload_date", "published_at", "upload_datetime", "created_at"]
        date_col = next((c for c in date_candidates if c in df_all.columns), None)
        if (
            date_col is not None
            and "video_id" in df_all.columns
            and "video_id" in pred.columns
            and "success_probability" in pred.columns
        ):
            pred_prob = (
                pred[["video_id", "success_probability"]]
                .dropna(subset=["video_id", "success_probability"])
                .drop_duplicates(subset=["video_id"])
            )
            df_recent_base = df_all.merge(pred_prob, on="video_id", how="left")
            df_recent_base[date_col] = pd.to_datetime(df_recent_base[date_col], errors="coerce")
            df_recent_base = df_recent_base.dropna(subset=[date_col, "success_probability"])
            df_recent_base = df_recent_base.sort_values(date_col, ascending=False)
            df_recent_30 = df_recent_base.head(30)
            df_prev_30 = df_recent_base.iloc[30:60]
            recent_30_success = df_recent_30["success_probability"].mean()
            prev_30_success = df_prev_30["success_probability"].mean() if len(df_prev_30) > 0 else np.nan
        else:
            df_recent_30 = pd.DataFrame()
            df_prev_30 = pd.DataFrame()
            recent_30_success = np.nan
            prev_30_success = np.nan

    if load_model and M is None:
        M = load_models()
    return D, M


# ─────────────────────────────────────────
# [4-1] 색상 팔레트 (전체 일관성)
# ─────────────────────────────────────────
COL_GREEN  = "#22c55e"   # 성공 / 긍정
COL_RED    = "#ef4444"   # 실패 / 부정
COL_PURPLE = "#6366f1"   # 메인 브랜드 컬러
COL_BLUE   = "#3b82f6"   # 보조 컬러
COL_FNB    = "#22c55e"   # FnB 도메인
COL_IT     = "#6366f1"   # IT 도메인
COL_GRAY   = "#9ca3af"   # 중립 / 비활성

# ─────────────────────────────────────────
# [4-2] 컬럼명 한글 매핑 (사용자 친화적 표시용)
# ─────────────────────────────────────────
# 데이터셋의 영문 컬럼명/값 → 사용자에게 보여줄 한글 라벨로 변환
# 차트 축, 테이블 헤더 등에서 LABEL_MAP[원본명] 또는 k(원본명) 으로 사용

# 4-2-1) 컬럼(Feature) 한글 매핑
LABEL_MAP = {
    # 기본 정보
    "domain": "도메인",
    "title": "영상 제목",
    "row_id": "영상 ID",
    "grade": "성공 등급",
    "success_rate": "성공률",
    "success_probability": "성공 확률",
    "count": "영상 수",

    # 영상 메타데이터
    "description_length": "설명 길이",
    "description_missing_flag": "설명 누락 여부",
    "tags_count": "태그 수",
    "tags_missing_flag": "태그 누락 여부",
    "영상길이(초)": "영상 길이(초)",
    "caption": "자막 사용",
    "definition": "화질",
    "embeddable": "영상 임베드 가능",
    "category_name": "카테고리",
    "has_paid_product_placement": "유료 광고 포함",

    # 업로드 시점 관련
    "upload_year": "업로드 연도",
    "upload_month": "업로드 월",
    "upload_quarter": "업로드 분기",
    "upload_dayofweek": "업로드 요일",
    "upload_hour": "업로드 시간",
    "upload_time_bucket": "업로드 시간대",
    "is_weekend": "주말 여부",

    # 콘텐츠 분류
    "cls_content_type": "콘텐츠 유형",
    "cls_marketing_purpose": "마케팅 목적",
    "cls_cta_type": "CTA 유형",
    "cls_is_series": "시리즈 여부",
    "cls_is_collaboration": "콜라보 여부",
    "length_bucket": "영상 길이 구간",
}

# 4-2-2) length_bucket 값 매핑 (영문 → 한글)
LENGTH_BUCKET_MAP = {
    "short":     "짧은 영상 (30초-3분)",
    "standard":  "표준 길이(3-8분)",
    "mid_ads":   "중간 길이(8-15분)",
    "long_form": "긴 영상 (15분+)",
}

# 4-2-3) upload_time_bucket 값 매핑
TIME_BUCKET_MAP = {
    "morning": "오전 (06-11시)",
    "lunch":   "점심 (11-17시)",
    "evening": "저녁 (17-23시)",
    "night":   "야간 (12-06시)",
}

def k(col):
    """영문 컬럼명을 한글로 변환 (없으면 원본 반환)"""
    return LABEL_MAP.get(col, col)

def map_category_value(feature, value):
    """
    cat_shap의 category 값을 사용자 친화적 한글 라벨로 변환.
    예: length_bucket=mid_ads → 중간 길이(8-15분)
        upload_time_bucket=morning → 오전 (06-11시)
    """
    if pd.isna(value):
        return "-"

    value = str(value)

    if feature == "length_bucket":
        return LENGTH_BUCKET_MAP.get(value, value)

    if feature == "upload_time_bucket":
        return TIME_BUCKET_MAP.get(value, value)

    return value

# ─────────────────────────────────────────
# [4-3] SHAP base_feature 정규화 
# ─────────────────────────────────────────
# 데이터셋마다 base_feature 형식이 다름:
#   - shap_fnb (CatBoost): "upload_year", "cls_content_type"        → 깨끗한 원본 컬럼명
#   - shap_all (XGBoost) : "num__upload_year", "cat__cls_cta_type_정보탐색"
#   - shap_it  (RF)      : 위와 동일 (sklearn ColumnTransformer 통과)
#
# 이를 통일된 형태(원본 컬럼명)로 정규화하고, 같은 컬럼끼리 합산해서 표시
# 예) "cat__cls_cta_type_정보탐색" + "cat__cls_cta_type_방문유도" → "cls_cta_type" 으로 묶음

def normalize_feature(raw_feat):
    """sklearn pipeline의 prefix와 카테고리 값을 제거하고 원본 컬럼명만 추출.
       예) 'num__upload_year'                  → 'upload_year'
           'cat__cls_cta_type_정보탐색'         → 'cls_cta_type'
           'cat__category_name_과학/기술'       → 'category_name'
           'upload_year' (이미 깨끗함)         → 'upload_year'"""
    if not isinstance(raw_feat, str):
        return str(raw_feat)
    # 1) prefix 제거 (num__ 또는 cat__)
    s = raw_feat
    if s.startswith("num__"):
        s = s[5:]
        return s  # 수치형은 그대로 반환
    if s.startswith("cat__"):
        s = s[5:]
        # 2) cat__의 경우, 알려진 컬럼명으로 시작하는지 확인하여 카테고리 값 분리
        # LABEL_MAP의 키들을 길이 내림차순으로 매칭 (긴 것부터 매칭해야 정확)
        for known_col in sorted(LABEL_MAP.keys(), key=len, reverse=True):
            # "cls_cta_type_정보탐색" → known_col이 "cls_cta_type"이면 매칭됨
            if s == known_col:
                return s
            if s.startswith(known_col + "_"):
                return known_col
        # 매칭 안 되면 그대로 반환 (안전장치)
        return s
    # prefix 없으면 그대로
    return s

def k_norm(raw_feat):
    """raw_feat을 정규화한 후 한글로 변환 (k 함수의 강화 버전)"""
    return k(normalize_feature(raw_feat))

def aggregate_shap_by_basefeature(shap_df):
    """SHAP DataFrame의 base_feature를 정규화 후 같은 컬럼끼리 합산.
       반환: 'feature'(원본 컬럼명), 'label'(한글), 'total_mean_abs_shap'(합산값) DataFrame"""
    df = shap_df.copy()
    df["feature"] = df["base_feature"].apply(normalize_feature)
    df["label"]   = df["feature"].apply(k)
    # 같은 feature끼리 |SHAP| 합산 (원-핫 인코딩된 카테고리 값들을 한 컬럼으로 묶음)
    agg = df.groupby(["feature", "label"], as_index=False)["total_mean_abs_shap"].sum()
    return agg.sort_values("total_mean_abs_shap", ascending=False)

def length_label_with_tip(value):
    """영상 길이 구간 표시용 라벨. mid_ads에는 중간 광고 툴팁 추가."""
    label = LENGTH_BUCKET_MAP.get(value, value)

    if value == "mid_ads":
        return f'{label} {tip(TIP_MID_ADS)}'

    return label

# ─────────────────────────────────────────
# [4-4] KPI 계산 헬퍼 
# ─────────────────────────────────────────
def _fmt_pct(x, digits=1):
    """0~1 비율을 퍼센트 문자열로 변환"""
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:.{digits}f}%"

def _fmt_pp(x, digits=1):
    """0~1 비율 차이를 %p 문자열로 변환"""
    if x is None or pd.isna(x):
        return "-"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.{digits}f}%p"

def _find_date_col(df):
    """대시보드 데이터에서 업로드 날짜 컬럼 후보를 자동 탐색"""
    candidates = [
        "upload_date", "published_at", "upload_datetime", "created_at",
        "order_date", "date"
    ]
    return next((c for c in candidates if c in df.columns), None)

def _domain_df(dom_key=None):
    """도메인별 원천 데이터 선택"""
    if dom_key == "FnB":
        return D["df_fnb"].copy()
    if dom_key == "IT":
        return D["df_it"].copy()
    return D["df_all"].copy()

def _prediction_df(dom_key=None):
    """도메인별 예측 결과 데이터 선택"""
    pred = D["pred"].copy()
    if dom_key is not None and "domain" in pred.columns:
        pred["domain"] = pred["domain"].astype(str).str.strip()
        pred = pred[pred["domain"] == dom_key].copy()
    return pred

def calc_avg_success_probability(dom_key=None):
    """평균 성공 확률 계산: 예측 확률이 있으면 사용, 없으면 성공 라벨 평균 사용"""
    pred = _prediction_df(dom_key)
    if "success_probability" in pred.columns and len(pred) > 0:
        return pd.to_numeric(pred["success_probability"], errors="coerce").mean()

    df = _domain_df(dom_key)
    if "grade" in df.columns and len(df) > 0:
        return pd.to_numeric(df["grade"], errors="coerce").mean()

    stats = D.get("eda_stats", {}).get(dom_key, {}) if dom_key else {}
    return stats.get("success_rate", np.nan)

def calc_recent_success_probability(dom_key=None, days=30):
    """최근 N일 평균 성공 확률과 이전 N일 대비 차이 계산"""
    for df, value_col in [
        (_prediction_df(dom_key), "success_probability"),
        (_domain_df(dom_key), "grade"),
    ]:
        if value_col not in df.columns:
            continue
        date_col = _find_date_col(df)
        if date_col is None:
            continue

        temp = df[[date_col, value_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
        temp = temp.dropna(subset=[date_col, value_col])
        if temp.empty:
            continue

        max_date = temp[date_col].max()
        recent_start = max_date - pd.Timedelta(days=days)
        prev_start = max_date - pd.Timedelta(days=days * 2)

        recent = temp[temp[date_col] > recent_start][value_col].mean()
        prev = temp[(temp[date_col] > prev_start) & (temp[date_col] <= recent_start)][value_col].mean()
        diff = recent - prev if not pd.isna(recent) and not pd.isna(prev) else np.nan
        return recent, diff

    return np.nan, np.nan

def make_success_trend(dom_key=None, periods=6):
    """월별/주차별 성공 확률 추이 데이터 생성"""
    for df, value_col in [
        (_prediction_df(dom_key), "success_probability"),
        (_domain_df(dom_key), "grade"),
    ]:
        if value_col not in df.columns:
            continue
        temp = df.copy()
        temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")

        date_col = _find_date_col(temp)
        if date_col is not None:
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col, value_col])
            if temp.empty:
                continue
            temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
            trend = temp.groupby("period", as_index=False)[value_col].mean().sort_values("period").tail(periods)
            trend["label"] = trend["period"].dt.strftime("%m월")
            trend["value_pct"] = trend[value_col] * 100
            return trend[["label", "value_pct"]]

        if "upload_year" in temp.columns and "upload_month" in temp.columns:
            temp = temp.dropna(subset=["upload_year", "upload_month", value_col])
            if temp.empty:
                continue
            temp["period_key"] = temp["upload_year"].astype(int).astype(str) + "-" + temp["upload_month"].astype(int).astype(str).str.zfill(2)
            trend = temp.groupby("period_key", as_index=False)[value_col].mean().sort_values("period_key").tail(periods)
            trend["label"] = trend["period_key"].str[-2:] + "월"
            trend["value_pct"] = trend[value_col] * 100
            return trend[["label", "value_pct"]]

    return pd.DataFrame(columns=["label", "value_pct"])

def get_stat_median(dom_key, stat_keys, df_col=None):
    """eda_stats 우선, 없으면 원천 DF에서 중앙값 계산"""
    stats = D.get("eda_stats", {}).get(dom_key, {})
    if isinstance(stat_keys, str):
        stat_keys = [stat_keys]
    for key in stat_keys:
        if isinstance(stats.get(key), dict) and "median" in stats[key]:
            return stats[key]["median"]

    if df_col is None:
        df_col = stat_keys[0]
    df = _domain_df(dom_key)
    if df_col in df.columns:
        return pd.to_numeric(df[df_col], errors="coerce").median()
    return np.nan

def calc_domain_feature_score(feature_name, dom_key):
    """선택 도메인의 feature별 성공 기여도 크기 계산"""
    shap_df = D["shap_fnb"] if dom_key == "FnB" else D["shap_it"]
    agg = aggregate_shap_by_basefeature(shap_df)
    row = agg[agg["feature"] == feature_name]
    if row.empty:
        return 0.0
    return float(row["total_mean_abs_shap"].iloc[0])

# ─────────────────────────────────────────
# [5] Plotly 공통 스타일
# ─────────────────────────────────────────
def plotly_base():
    """모든 plotly 차트에 공통으로 적용할 기본 레이아웃"""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",  # 차트 외부 배경 투명
        plot_bgcolor="rgba(0,0,0,0)",   # 플롯 영역 배경 투명
        font=dict(family="Noto Sans KR", size=12),
        margin=dict(l=10, r=10, t=30, b=10)
    )

# ─────────────────────────────────────────
# [5] 공통 헬퍼 함수
# ─────────────────────────────────────────
def section_header(num, title, subtitle=""):
    """페이지 상단 섹션 헤더 (보라색 그라데이션 박스)
       num: 페이지 번호, title: 제목, subtitle: 부제목"""
    st.markdown(f"""
    <div class="section-header">
        <h2><span style="background:rgba(255,255,255,0.25);border-radius:50%;width:30px;height:30px;
        display:inline-flex;align-items:center;justify-content:center;font-size:14px;margin-right:10px;">{num}</span>
        {title}</h2>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def metric_card(title, value, sub="", border_color=COL_PURPLE):
    """숫자 강조용 메트릭 카드 (KPI 표시)"""
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{border_color};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 도움말 툴팁 / 개념 설명 박스 헬퍼
# ─────────────────────────────────────────
# 사용자에게 친숙하지 않은 용어를 설명하기 위한 헬퍼들.
# - tip(text): 차트 제목 옆에 ⓘ 아이콘을 표시, hover하면 설명 말풍선이 뜸
# - concept_box(): 페이지 첫 등장 시 '성공 기여도란?' 같은 설명 박스
# - 용어 통일: SHAP / SHAP value → "성공 기여도" 로 표시

def tip(text):
    """ⓘ 아이콘 + hover 툴팁 HTML 문자열 반환.
       title/subheader 옆에 붙여서 사용.
       사용 예: st.markdown(f"### 차트 제목 {tip('이 차트는...')}", unsafe_allow_html=True)"""
    # 줄바꿈은 \n으로 입력 → CSS의 white-space: pre-line이 처리
    safe = text.replace('"', '&quot;').replace("'", "&#39;")
    return f'<span class="help-tip" data-tip="{safe}">i</span>'

def subheader_with_tip(title, tip_text):
    """st.subheader 대체 헬퍼: 제목 옆에 ⓘ 툴팁 포함"""
    st.markdown(
        f'<div style="font-size:1.1rem;font-weight:600;color:#1e1e3f;margin-bottom:8px;">'
        f'{title} {tip(tip_text)}</div>',
        unsafe_allow_html=True
    )

def concept_box(title, body):
    """페이지 첫 등장 시 보여주는 '이게 뭐야?' 설명 박스
       title: 박스 제목 (이모지 포함 가능), body: HTML 가능한 본문"""
    st.markdown(f"""
    <div class="concept-box">
        <b>💡 {title}</b><br>{body}
    </div>""", unsafe_allow_html=True)


# ──── 자주 쓰는 툴팁 텍스트 (한 곳에서 관리) ────
# 같은 개념을 여러 페이지에서 반복 설명하므로, 텍스트를 상수로 정의
TIP_CONTRIBUTION = (
    "성공 기여도란?\n"
    "머신러닝 기반 AI 모델이 분석한 '이 요소가 영상 성공에 얼마나 영향을 줬는지' 점수입니다.\n"
    "+ 값: 성공 확률을 높임\n"
    "- 값: 성공 확률을 낮춤\n"
    "절댓값이 클수록 영향력이 큽니다."
)
TIP_CONTRIBUTION_ABS = (
    "기여도 크기 (절댓값)\n"
    "방향(+/-)에 관계없이 영향이 얼마나 컸는지를 나타냅니다.\n"
    "이 값이 클수록 성공/실패에 결정적인 요소입니다."
)
TIP_WATERFALL = (
    "기여도 흐름 차트란?\n"
    "평균 성공 확률(왼쪽)에서 시작해서, 이 영상의 각 요소가 확률을\n"
    "얼마나 올리거나(초록↑) 내렸는지(빨강↓) 단계별로 보여줍니다.\n"
    "마지막 막대가 최종 예측 확률입니다."
)
TIP_DEPENDENCY = (
    "의존성 분석이란?\n"
    "특성의 '값(예: 브이로그/인터뷰)'에 따라 성공 기여도가\n"
    "어떻게 달라지는지를 비교합니다."
)

TIP_MID_ADS = (
    "중간 길이(8-15분)란?\n"
    "YouTube에서는 수익 창출 영상이 8분 이상이면 영상 중간에 광고(mid-roll)를 삽입할 수 있습니다.\n"
    "따라서 8-15분 구간은 정보 전달량과 수익화 가능성을 함께 고려할 수 있는 길이입니다."
)

# [6] 상단 네비게이션은 app.py의 탭 구조로 대체합니다.

# ═══════════════════════════════════════════════════════════════════════════
# [7-1] 홈 - Executive Dashboard
# ═══════════════════════════════════════════════════════════════════════════
# 전체 KPI, SHAP TOP5, 도메인별 성공률, 트렌드, 추천액션을 한 화면에 요약

# 최근 업로드 관련 전역값은 ensure_longform_resources()에서 계산합니다.
df_all = pd.DataFrame()
pred = pd.DataFrame()
df_recent_30 = pd.DataFrame()
df_prev_30 = pd.DataFrame()
recent_30_success = np.nan
prev_30_success = np.nan

def page_home():
    section_header(1, "메인 홈",
                   "전체 채널의 AI 분석 요약과 핵심 성과 지표를 확인하세요.")

    df_all = D["df_all"]

    # ──── KPI 카드 4개 (상단 가로 배치) ────
    avg_success = calc_avg_success_probability()
    recent_success, recent_diff = calc_recent_success_probability(days=30)
    shap_agg_all = aggregate_shap_by_basefeature(D["shap_all"])
    top_feature_label = shap_agg_all.iloc[0]["label"] if not shap_agg_all.empty else "-"

    fnb_count = len(D["df_fnb"]) if "df_fnb" in D else D["eda_stats"].get("FnB", {}).get("count", 0)
    it_count = len(D["df_it"]) if "df_it" in D else D["eda_stats"].get("IT", {}).get("count", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("전체 평균 성공 확률", _fmt_pct(avg_success), "예측 확률 평균 기준", COL_PURPLE)
    with c2:
        if pd.isna(recent_30_success):
            metric_card(
                "최근 업로드 30개 평균 성공 확률",
                "-",
                "업로드일시 또는 예측 확률 연결 불가",
                COL_GREEN
            )
        else:
            recent_30_text = f"{recent_30_success * 100:.1f}%"

            if pd.isna(prev_30_success):
                sub_text = f"최근 업로드 영상 {len(df_recent_30):,}개 기준"
            else:
                diff = recent_30_success - prev_30_success
                arrow = "▲" if diff >= 0 else "▼"
                sub_text = f"{arrow} 이전 30개 대비 {diff * 100:+.1f}%p"

            metric_card(
                "최근 업로드 30개 평균 성공 확률",
                recent_30_text,
                sub_text,
                COL_GREEN
            )
    with c3:
        metric_card("분석 영상 수", f"{len(df_all):,}개",
                    f"FnB {fnb_count:,}개 / IT {it_count:,}개", COL_BLUE)
    with c4:
        metric_card("가장 중요한 요소 (전체)", top_feature_label, "AI가 분석한 영향력 기준", "#f59e0b")

    # ──── 중단: 성공 기여도 TOP5 / 도메인별 성공률 / 트렌드 ────
    col1, col2, col3 = st.columns([2, 2, 1.5])

    # [좌] 성공 기여도 TOP 5 (전체)
    with col1:
        subheader_with_tip("성공 기여도 TOP 5 (전체)", TIP_CONTRIBUTION_ABS)
        top5 = shap_agg_all.head(5)
        fig = go.Figure(go.Bar(
            x=top5["total_mean_abs_shap"],
            y=top5["label"],
            orientation="h",
            marker_color=COL_PURPLE,
            text=[f"{v:.2f}" for v in top5["total_mean_abs_shap"]],
            textposition="outside"
        ))
        fig.update_layout(**plotly_base(), height=260,
                          xaxis_title="평균 기여도 크기",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # [중] 도메인별 평균 성공 확률 (도넛 차트 2개)
    with col2:
        st.subheader("도메인별 평균 성공 확률")
        fnb_rate = calc_avg_success_probability("FnB")
        it_rate = calc_avg_success_probability("IT")
        fnb_pct = 0 if pd.isna(fnb_rate) else round(fnb_rate * 100, 1)
        it_pct = 0 if pd.isna(it_rate) else round(it_rate * 100, 1)

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
        fig.add_trace(go.Pie(values=[fnb_pct, max(0, 100 - fnb_pct)], labels=["성공", "실패"],
            hole=0.7, marker_colors=[COL_FNB, "#e5e7eb"], textinfo="none", name="FnB"
        ), row=1, col=1)
        fig.add_trace(go.Pie(values=[it_pct, max(0, 100 - it_pct)], labels=["성공", "실패"],
            hole=0.7, marker_colors=[COL_IT, "#e5e7eb"], textinfo="none", name="IT"
        ), row=1, col=2)
        fig.update_layout(**plotly_base(), height=260,
                          annotations=[
                              dict(text=f"FnB<br><b>{fnb_pct:.1f}%</b>", x=0.18, y=0.5, showarrow=False, font_size=14),
                              dict(text=f"IT<br><b>{it_pct:.1f}%</b>",  x=0.82, y=0.5, showarrow=False, font_size=14),
                          ])
        st.plotly_chart(fig, use_container_width=True)
        higher_dom = "FnB" if fnb_pct > it_pct else "IT"
        st.markdown(f'<div class="insight-box">{higher_dom} 도메인이 더 높은 평균 성공 확률을 보입니다.</div>',
                    unsafe_allow_html=True)

    # [우] 최근 트렌드 인사이트
    with col3:
        st.subheader("최근 트렌드 인사이트")
        insights = []

        # 시간대/콘텐츠/CTA별 최근성이 아닌 전체 기준 상위 항목을 데이터에서 계산
        if "eda_hour" in D and not D["eda_hour"].empty:
            h = D["eda_hour"].copy()
            if "success_rate" in h.columns and "upload_hour" in h.columns:
                best_hour = h.groupby("upload_hour")["success_rate"].mean().idxmax()
                insights.append(("🌙", f"{int(best_hour)}시 업로드 성공률 우세"))
        if "eda_ctype" in D and not D["eda_ctype"].empty:
            ct = D["eda_ctype"].copy()
            if "success_rate" in ct.columns and "cls_content_type" in ct.columns:
                best_ct = ct.groupby("cls_content_type")["success_rate"].mean().idxmax()
                insights.append(("🎬", f"{best_ct} 콘텐츠 성공률 우세"))
        if "eda_cta" in D and not D["eda_cta"].empty:
            cta = D["eda_cta"].copy()
            if "success_rate" in cta.columns and "cls_cta_type" in cta.columns:
                best_cta = cta.groupby("cls_cta_type")["success_rate"].mean().idxmax()
                insights.append(("🎯", f"{best_cta} CTA 성공률 우세"))

        for icon, text in insights[:3]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:10px;
            background:white;border-radius:8px;margin-bottom:8px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                <span style="font-size:20px;">{icon}</span>
                <span style="font-size:13px;color:#374151;">{text}</span>
            </div>""", unsafe_allow_html=True)

    # ──── 하단: 성공률 추이 / 도메인별 비교 ────
    col1, col2 = st.columns([2.5, 2.5])

    # [좌] 분석 기간별 성공 확률 추이 (라인 차트)
    with col1:
        st.subheader("분석 기간별 성공 확률 추이")
        trend = make_success_trend(periods=6)
        if trend.empty:
            st.info("성공 확률 추이를 계산할 날짜/월 데이터가 없습니다.")
        else:
            fig = go.Figure(go.Scatter(
                x=trend["label"], y=trend["value_pct"], mode="lines+markers+text",
                line=dict(color=COL_PURPLE, width=2.5),
                marker=dict(size=8, color=COL_PURPLE),
                text=[f"{r:.1f}%" for r in trend["value_pct"]],
                textposition="top center",
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"
            ))
            y_min = max(0, trend["value_pct"].min() - 5)
            y_max = min(100, trend["value_pct"].max() + 5)
            fig.update_layout(**plotly_base(), height=260,
                              yaxis=dict(range=[y_min, y_max], ticksuffix="%"))
            st.plotly_chart(fig, use_container_width=True)

    # [중] 도메인별 핵심 요소 영향도 비교 (그룹 막대)
    with col2:
        st.subheader("도메인별 핵심 요소 영향도 비교")
        feature_items = [
            ("설명<br>길이", "description_length"),
            ("CTA<br>유형", "cls_cta_type"),
            ("콘텐츠<br>유형", "cls_content_type"),
            ("업로드<br>시간", "upload_time_bucket"),
            ("업로드<br>요일", "upload_dayofweek"),
            ("영상<br>길이", "length_bucket"),
            ("카테고리", "category_name"),
            ("자막<br>사용", "caption"),
        ]
        categories = [x[0] for x in feature_items]
        fnb_raw = [calc_domain_feature_score(feat, "FnB") for _, feat in feature_items]
        it_raw  = [calc_domain_feature_score(feat, "IT") for _, feat in feature_items]
        max_val = max(fnb_raw + it_raw + [1e-9])
        fnb_vals = [v / max_val * 100 for v in fnb_raw]
        it_vals  = [v / max_val * 100 for v in it_raw]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="FnB", x=categories, y=fnb_vals,
                             marker_color=COL_FNB, opacity=0.85))
        fig.add_trace(go.Bar(name="IT",  x=categories, y=it_vals,
                             marker_color=COL_IT,  opacity=0.85))
        fig.update_layout(**plotly_base(), height=260, barmode="group",
                          yaxis_title="상대 영향도 지수", yaxis_ticksuffix="점",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# [7-2] 도메인 분석 (FnB / IT 선택)
# ═══════════════════════════════════════════════════════════════════════════
# 변경 사항: 기존 'FnB vs IT 비교' → '도메인 선택 후 해당 도메인의 분석 결과 표시'
#  - 상단에서 라디오 버튼으로 FnB / IT 중 선택
#  - 선택된 도메인 하나에 대한 모든 분석 차트를 표시
def page_domain():
    section_header(2, "도메인별 상세 분석",
                   "도메인을 선택하면 해당 도메인의 성공 전략과 핵심 지표를 자세히 볼 수 있습니다.")

    # ──── 도메인 선택 라디오 버튼 ────
    col_sel, _ = st.columns([1, 4])
    with col_sel:
        domain_choice = st.radio(
            "도메인 선택",
            ["🍽️ FnB (식음료)", "💻 IT"],
            horizontal=True,
            key="domain_select"
        )
    # 내부적으로 사용할 키 (FnB / IT) - 데이터 필터링용
    dom_key   = "FnB" if "FnB" in domain_choice else "IT"
    dom_color = COL_FNB if dom_key == "FnB" else COL_IT
    dom_emoji = "🍽️"   if dom_key == "FnB" else "💻"

    # ──── 탭 구성: 핵심 요약 / CTA / 콘텐츠 / 시간 / 요일 ────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 핵심 요약", "🎯 CTA 분석", "📺 콘텐츠 유형", "⏰ 업로드 시간", "📅 업로드 요일"
    ])

    # ─────── 탭 1: 핵심 요약 (전략 + KPI + SHAP) ───────
    with tab1:
        col1, col2 = st.columns([2, 1])

        # [좌] 주요 성공 전략 카드
        with col1:
            bg = "#f0fdf4" if dom_key == "FnB" else "#eff6ff"
            txt = "#16a34a" if dom_key == "FnB" else "#2563eb"
            txt2 = "#166534" if dom_key == "FnB" else "#1e40af"
            st.markdown(f"""
            <div style="background:{bg};border-radius:12px;padding:18px;">
                <div style="color:{txt};font-weight:700;font-size:16px;margin-bottom:12px;">
                {dom_emoji} {dom_key} 주요 성공 전략</div>
            """, unsafe_allow_html=True)
            strats = D["strategy"][dom_key]["recommended"][:5]
            for s in strats:
                st.markdown(f'<div style="padding:6px 0;font-size:13px;color:{txt2};">✅ {s}</div>',
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # [우] 핵심 성과 지표
        with col2:
            st.subheader(f"{dom_key} 핵심 성과 지표")
            stats = D["eda_stats"][dom_key]
            avg_success_dom = calc_avg_success_probability(dom_key)
            video_len_med = get_stat_median(dom_key, ["영상길이(초)", "video_length_sec"], "영상길이(초)")
            desc_len_med = get_stat_median(dom_key, "description_length", "description_length")
            view_med = get_stat_median(dom_key, ["조회수", "view_count"], "view_count")
            kpis = [
                ("평균 성공 확률", _fmt_pct(avg_success_dom)),
                ("분석 영상 수", f"{stats.get('count', len(_domain_df(dom_key))):,}개"),
                ("영상 길이 중앙값", "-" if pd.isna(video_len_med) else f"{int(video_len_med)}초"),
                ("설명 길이 중앙값", "-" if pd.isna(desc_len_med) else f"{int(desc_len_med)}자"),
                ("조회수 중앙값", "-" if pd.isna(view_med) else f"{int(view_med):,}"),
            ]
            for kpi, v in kpis:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 0;border-bottom:1px solid #f3f4f6;font-size:13px;">
                    <span style="color:#6b7280;">{kpi}</span>
                    <span style="color:{dom_color};font-weight:700;">{v}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # 첫 등장: '성공 기여도'가 뭔지 한 번 설명
        concept_box(
            "성공 기여도란?",
            "AI 모델이 분석한 <b>각 요소의 영향력 점수</b>입니다. "
            "양수(+)는 성공 확률을 높이고, 음수(-)는 낮춥니다. "
            "절댓값이 클수록 결정적인 요소입니다."
        )

        # ──── 성공 기여도 TOP 8 + CTA 기여도 + 콘텐츠 기여도 ────
        col1, col2, col3 = st.columns(3)

        # 성공 기여도 TOP 8 (선택 도메인)
        with col1:
            subheader_with_tip("성공 기여도 TOP 8", TIP_CONTRIBUTION_ABS)
            shap_data = D["shap_fnb"] if dom_key == "FnB" else D["shap_it"]
            # 정규화 + 같은 컬럼끼리 합산
            shap_agg = aggregate_shap_by_basefeature(shap_data)
            top = shap_agg.head(8)
            fig = go.Figure(go.Bar(
                x=top["total_mean_abs_shap"], y=top["label"],
                orientation="h", marker_color=dom_color, opacity=0.85,
                text=[f"{v:.2f}" for v in top["total_mean_abs_shap"]],
                textposition="outside"
            ))
            fig.update_layout(**plotly_base(), height=320,
                              xaxis_title="평균 기여도 크기",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        # CTA 유형별 성공 기여도
        with col2:
            subheader_with_tip("CTA 유형별 성공 기여도", TIP_CONTRIBUTION)
            cta_shap = D["cat_shap"][D["cat_shap"]["feature"]=="cls_cta_type"].copy()
            sub = cta_shap[cta_shap["dataset"]==dom_key].sort_values("mean_shap", ascending=False)
            colors = [COL_GREEN if v >= 0 else COL_RED for v in sub["mean_shap"]]
            fig = go.Figure(go.Bar(
                x=sub["category"], y=sub["mean_shap"],
                marker_color=colors,
                text=[f"{v:+.2f}" for v in sub["mean_shap"]],
                textposition="outside"
            ))
            fig.update_layout(**plotly_base(), height=320,
                              yaxis_title="평균 성공 기여도")
            st.plotly_chart(fig, use_container_width=True)

        # 콘텐츠 유형별 성공 기여도
        with col3:
            subheader_with_tip("콘텐츠 유형별 성공 기여도", TIP_CONTRIBUTION)
            ct_shap = D["cat_shap"][D["cat_shap"]["feature"]=="cls_content_type"].copy()
            sub = ct_shap[ct_shap["dataset"]==dom_key].sort_values("mean_shap", ascending=False)
            colors = [COL_GREEN if v >= 0 else COL_RED for v in sub["mean_shap"]]
            fig = go.Figure(go.Bar(
                x=sub["category"], y=sub["mean_shap"],
                marker_color=colors,
                text=[f"{v:+.2f}" for v in sub["mean_shap"]],
                textposition="outside"
            ))
            fig.update_layout(**plotly_base(), height=320,
                              yaxis_title="평균 성공 기여도")
            st.plotly_chart(fig, use_container_width=True)

        # 인사이트 박스
        msg_fnb = "FnB는 체류 시간과 경험 중심 콘텐츠가 중요하며, 긴 영상과 방문 유도 CTA가 효과적입니다."
        msg_it  = "IT는 정보 전달과 참여 유도 콘텐츠가 중요하며, 중간 길이 영상과 이벤트/기타 CTA가 효과적입니다."
        st.markdown(f"""
        <div class="insight-box">
        💡 <b>핵심 인사이트:</b> {msg_fnb if dom_key == "FnB" else msg_it}
        </div>""", unsafe_allow_html=True)

    # ─────── 탭 2: CTA 분석 ───────
    with tab2:
        st.subheader(f"{dom_key} CTA 유형별 성공률")
        eda_cta = D["eda_cta"].copy()
        sub = eda_cta[eda_cta["domain"]==dom_key].sort_values("success_rate", ascending=False)
        col1, col2 = st.columns([2, 1])
        with col1:
            colors = [dom_color if v >= 0.5 else COL_RED for v in sub["success_rate"]]
            fig = go.Figure(go.Bar(
                x=sub["cls_cta_type"], y=sub["success_rate"]*100,
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in sub["success_rate"]],
                textposition="outside"
            ))
            fig.update_layout(**plotly_base(), height=380,
                              yaxis_ticksuffix="%", yaxis_title="성공률")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**CTA 유형별 영상 수**")
            for _, r in sub.iterrows():
                cnt = int(r.get("count", 0))
                rate = r["success_rate"]
                color = dom_color if rate >= 0.5 else COL_RED
                st.markdown(f"""
                <div style="margin-bottom:6px;font-size:12px;">
                    <div style="display:flex;justify-content:space-between;">
                        <span>{r['cls_cta_type']}</span>
                        <span style="color:{color};font-weight:700;">{rate*100:.1f}% ({cnt})</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ─────── 탭 3: 콘텐츠 유형 ───────
    with tab3:
        st.subheader(f"{dom_key} 콘텐츠 유형별 성공률")
        eda_ct = D["eda_ctype"].copy()
        sub = eda_ct[eda_ct["domain"]==dom_key].sort_values("success_rate", ascending=False)
        colors = [dom_color if v >= 0.5 else COL_RED for v in sub["success_rate"]]
        fig = go.Figure(go.Bar(
            x=sub["cls_content_type"], y=sub["success_rate"]*100,
            marker_color=colors,
            text=[f"{v*100:.0f}%" for v in sub["success_rate"]],
            textposition="outside"
        ))
        fig.update_layout(**plotly_base(), height=420,
                          yaxis_ticksuffix="%", yaxis_title="성공률")
        st.plotly_chart(fig, use_container_width=True)

    # ─────── 탭 4: 업로드 시간 (히트맵 - 잘림 문제 해결) ───────
    with tab4:
        st.subheader(f"{dom_key} 업로드 시간대별 성공률")
        eda_h = D["eda_hour"].copy()
        sub = eda_h[eda_h["domain"]==dom_key].copy()

        # 4시간 단위로 그룹핑하는 헬퍼
        def hour_bucket(h):
            if   0  <= h < 4 : return "00-03시"
            elif 4  <= h < 8 : return "04-07시"
            elif 8  <= h < 12: return "08-11시"
            elif 12 <= h < 16: return "12-15시"
            elif 16 <= h < 20: return "16-19시"
            else:              return "20-23시"

        sub["hour_group"] = sub["upload_hour"].apply(hour_bucket)
        bucket_order = ["00-03시","04-07시","08-11시","12-15시","16-19시","20-23시"]
        pivot = sub.groupby("hour_group")["success_rate"].mean().reindex(bucket_order).fillna(0)

        col1, col2 = st.columns([2, 1])
        with col1:
            # 히트맵: 잘림 방지 위해 height 충분히 + margin 여유 + autosize
            color_scale = "Greens" if dom_key == "FnB" else "Purples"
            fig = go.Figure(go.Heatmap(
                z=[pivot.values],
                x=pivot.index.tolist(),
                y=[dom_key],
                colorscale=color_scale,
                text=[[f"{v*100:.0f}%" for v in pivot.values]],
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=True,
                zmin=0.3, zmax=0.9
            ))
            # 히트맵 잘림 방지: height를 200 이상으로, margin 충분히 확보
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Noto Sans KR", size=12),
                height=240,                                # 기존 100 → 240 (잘림 방지)
                margin=dict(l=40, r=40, t=50, b=80),       # 여백 확보
                title=f"{dom_key} 시간대별 성공률 히트맵",
                xaxis=dict(side="bottom"),
                yaxis=dict(automargin=True)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**시간대별 성공률 순위**")
            ranked = pivot.sort_values(ascending=False)
            for i, (bucket, rate) in enumerate(ranked.items(), 1):
                medal = ["🥇","🥈","🥉","4","5","6"][i-1]
                color = dom_color if rate >= 0.5 else COL_RED
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                padding:6px 0;border-bottom:1px solid #f3f4f6;font-size:13px;">
                    <span>{medal} {bucket}</span>
                    <span style="color:{color};font-weight:700;">{rate*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

    # ─────── 탭 5: 업로드 요일 ───────
    with tab5:
        st.subheader(f"{dom_key} 업로드 요일별 성공률")
        eda_d = D["eda_day"].copy()
        sub = eda_d[eda_d["domain"]==dom_key].copy()
        # 요일 순서 정렬
        day_order = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
        sub = sub.set_index("upload_dayofweek").reindex(day_order).reset_index()
        sub["success_rate"] = sub["success_rate"].fillna(0)
        colors = [dom_color if v >= 0.5 else COL_RED for v in sub["success_rate"]]
        fig = go.Figure(go.Bar(
            x=sub["upload_dayofweek"], y=sub["success_rate"]*100,
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in sub["success_rate"]],
            textposition="outside"
        ))
        fig.update_layout(**plotly_base(), height=380,
                          yaxis_ticksuffix="%", yaxis_title="성공률")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# [7-3] 성공 요인 탐색 (SHAP 분석)
# ═══════════════════════════════════════════════════════════════════════════
def page_shap():
    section_header(3, "영상 성공 요인 탐색기",
                   "영상 특성별로 성공에 미치는 영향(성공 기여도)을 탐색하고, 어떤 요소가 성과를 높이는지 확인하세요.")

    # 페이지 첫 등장 시 '성공 기여도'와 '의존성' 개념 설명
    concept_box(
        "이 페이지는 무엇을 보여주나요?",
        "AI 모델이 분석한 <b>각 특성의 성공 기여도(영향력 점수)</b>를 보여줍니다.<br>"
        "• <b>전체 중요도</b>: 어떤 특성이 가장 중요한지<br>"
        "• <b>의존성 분석</b>: 특성 값(예: '브이로그' vs '인터뷰')에 따라 영향이 어떻게 달라지는지<br>"
        "• <b>특성 분포</b>: 성공/실패 영상의 특성값이 어떻게 다른지"
    )

    tab1, tab2 = st.tabs(["📊 전체 중요도", "📈 특성 분포"])

    # ─────── 탭 1: 전체 중요도 + 선택 특성 의존성 ───────
    with tab1:
        # 상단 선택 영역
        col_sel, col_chart = st.columns([1.1, 2.9])

        feat_map_rev = {
            "CTA 유형":        "cls_cta_type",
            "콘텐츠 유형":     "cls_content_type",
            "마케팅 목적":     "cls_marketing_purpose",
            "업로드 요일":     "upload_dayofweek",
            "업로드 시간대":   "upload_time_bucket",
            "영상 길이 구간":  "length_bucket",
        }

        with col_sel:
            # 1) 도메인 먼저 선택
            selected_domain = st.selectbox(
                "도메인 선택",
                ["FnB", "IT"],
                key="shap_tab1_domain_select"
            )

            # 2) 그 아래 특성 선택
            feature_sel = st.selectbox(
                "특성 선택",
                list(feat_map_rev.keys()),
                key="shap_tab1_feature_select"
            )

            feat_key = feat_map_rev[feature_sel]

            # 선택 도메인/특성에 해당하는 cat_shap 추출
            cat_s = D["cat_shap"].copy()
            sub_feat = cat_s[
                (cat_s["feature"] == feat_key) &
                (cat_s["dataset"] == selected_domain)
            ].copy()

            # 도메인 데이터가 비어 있으면 ALL로 대체
            if sub_feat.empty:
                sub_feat = cat_s[
                    (cat_s["feature"] == feat_key) &
                    (cat_s["dataset"] == "ALL")
                ].copy()

            # category 값 한글 매핑
            if not sub_feat.empty:
                sub_feat["category_label"] = sub_feat["category"].apply(
                    lambda x: map_category_value(feat_key, x)
                )

            # 상위 3개 인사이트: mean_shap 높은 순
            top3 = sub_feat.sort_values("mean_shap", ascending=False).head(3)

            st.markdown(f"""
            <div style="background:#f9f5ff;border-radius:10px;padding:14px;margin-top:8px;">
                <b style="color:{COL_PURPLE};">📌 선택 조건</b><br><br>
                <div style="font-size:13px;color:#374151;line-height:1.7;">
                    <b>도메인:</b> {selected_domain}<br>
                    <b>특성:</b> {feature_sel}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:14px;margin-top:10px;
            box-shadow:0 1px 5px rgba(0,0,0,0.06);">
                <b style="color:{COL_PURPLE};">🏆 {selected_domain} 상위 3개 인사이트</b>
            """, unsafe_allow_html=True)

            if top3.empty:
                st.info("해당 도메인/특성의 성공 기여도 데이터가 없습니다.")
            else:
                for i, (_, row) in enumerate(top3.iterrows(), 1):
                    val = row["mean_shap"]
                    color = COL_GREEN if val >= 0 else COL_RED
                    arrow = "↑" if val >= 0 else "↓"
                    cat_label = row["category_label"]

                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:6px 0;border-bottom:1px solid #f3f4f6;font-size:13px;">
                        <span>{i}. {cat_label}</span>
                        <span style="color:{color};font-weight:700;">{arrow} {val:+.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # 오른쪽: 선택한 특성의 dependency plot
        with col_chart:
            subheader_with_tip(
                f"{selected_domain} - {feature_sel} 의존성 분석",
                TIP_DEPENDENCY
            )

            if sub_feat.empty:
                st.warning("선택한 조건에 해당하는 의존성 분석 데이터가 없습니다.")
            else:
                sub_plot = sub_feat.sort_values("mean_shap", ascending=False).copy()

                colors = [
                    COL_GREEN if v >= 0 else COL_RED
                    for v in sub_plot["mean_shap"]
                ]

                fig = go.Figure(go.Bar(
                    x=sub_plot["category_label"],
                    y=sub_plot["mean_shap"],
                    marker_color=colors,
                    text=[f"{v:+.2f}" for v in sub_plot["mean_shap"]],
                    textposition="outside"
                ))

                fig.update_layout(
                    **plotly_base(),
                    height=380,
                    title=f"{selected_domain} | {feature_sel} 값별 성공 기여도",
                    xaxis_title=feature_sel,
                    yaxis_title="평균 성공 기여도",
                    xaxis=dict(tickangle=-20)
                )

                st.plotly_chart(fig, use_container_width=True)

                best_row = sub_plot.iloc[0]
                worst_row = sub_plot.iloc[-1]

                st.markdown(f"""
                <div class="insight-box">
                💡 <b>해석 가이드:</b><br>
                선택한 <b>{selected_domain}</b> 도메인에서 <b>{feature_sel}</b> 값별로 성공 기여도가 어떻게 달라지는지 보여줍니다.<br>
                가장 긍정적인 값은 <b>{best_row["category_label"]}</b> ({best_row["mean_shap"]:+.2f})이고,
                가장 낮은 값은 <b>{worst_row["category_label"]}</b> ({worst_row["mean_shap"]:+.2f})입니다.
                </div>
                """, unsafe_allow_html=True)

        # ──── 특성별 성공 영향 방향 ────
        st.subheader("특성별 성공 영향 방향")
        st.caption("선택한 도메인에서 각 특성별로 가장 영향력 있는 카테고리와 영향 방향을 표시합니다.")

        cat_shap_df = D["cat_shap"]
        selected_direction_domain = selected_domain

        features_to_show = [
            ("cls_cta_type",          "CTA 유형"),
            ("cls_content_type",      "콘텐츠 유형"),
            ("cls_marketing_purpose", "마케팅 목적"),
            ("upload_dayofweek",      "업로드 요일"),
            ("upload_time_bucket",    "업로드 시간대"),
            ("length_bucket",         "영상 길이 구간"),
            ("category_name",         "카테고리"),
            ("upload_year",           "업로드 연도"),
        ]

        def get_top_category_with_direction(feat_col, dom):
            sub = cat_shap_df[
                (cat_shap_df["feature"] == feat_col) &
                (cat_shap_df["dataset"] == dom)
            ].copy()

            if sub.empty:
                sub = cat_shap_df[
                    (cat_shap_df["feature"] == feat_col) &
                    (cat_shap_df["dataset"] == "ALL")
                ].copy()

            if sub.empty:
                return "-", 0.0, "→"

            sub["abs_shap"] = sub["mean_shap"].abs()
            top_row = sub.sort_values("abs_shap", ascending=False).iloc[0]

            cat_name = map_category_value(feat_col, top_row["category"])
            shap_val = float(top_row["mean_shap"])
            arrow = "↑" if shap_val > 0 else ("↓" if shap_val < 0 else "→")

            return cat_name, shap_val, arrow

        domain_color = COL_FNB if selected_direction_domain == "FnB" else COL_IT
        cols = st.columns(4)

        for i, (feat_col, feat_label) in enumerate(features_to_show):
            cat_name, shap_val, arrow = get_top_category_with_direction(
                feat_col,
                selected_direction_domain
            )

            arrow_color = COL_GREEN if shap_val > 0 else (COL_RED if shap_val < 0 else COL_GRAY)

            with cols[i % 4]:
                st.markdown(
                    f"""
    <div style="background:white;border-radius:14px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:14px;min-height:170px;text-align:center;">
        <div style="font-size:15px;font-weight:800;color:#374151;margin-bottom:14px;">
            {feat_label}
        </div>
        <div style="font-size:12px;color:{domain_color};font-weight:800;margin-bottom:8px;">
            {selected_direction_domain}
        </div>
        <div style="font-size:14px;color:#4b5563;margin-bottom:14px;min-height:22px;">
            {cat_name}
        </div>
        <div style="font-size:26px;color:{arrow_color};font-weight:900;">
            {arrow} <span style="font-size:15px;">{shap_val:+.2f}</span>
        </div>
    </div>
    """,
                    unsafe_allow_html=True
                )

    # ─────── 탭 2:  특성 분포 (수치형 변수 히스토그램) ───────
    with tab2:
        st.subheader("특성 분포 분석")
        df_all = D["df_all"]
        # 수치형 특성 한글 매핑 (셀렉트박스에 한글 표시 → 내부적으로는 영문 컬럼 사용)
        num_feat_options = {
            "설명 길이":   "description_length",
            "영상 길이":   "영상길이(초)",
            "태그 수":     "tags_count",
            "업로드 시간": "upload_hour",
        }
        feat_label_sel = st.selectbox("수치형 특성 선택", list(num_feat_options.keys()))
        num_feat = num_feat_options[feat_label_sel]
        col1, col2 = st.columns(2)
        for i, (dom, color) in enumerate([("FnB", COL_FNB), ("IT", COL_IT)]):
            sub = df_all[df_all["domain"]==dom]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=sub[sub["grade"]==1][num_feat], name="성공",
                                       marker_color=color, opacity=0.7, nbinsx=30))
            fig.add_trace(go.Histogram(x=sub[sub["grade"]==0][num_feat], name="실패",
                                       marker_color=COL_RED, opacity=0.6, nbinsx=30))
            fig.update_layout(**plotly_base(), barmode="overlay",
                              title=f"{dom} - {feat_label_sel} 분포",
                              height=320, xaxis_title=feat_label_sel)
            (col1 if i==0 else col2).plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# [7-4] 영상 진단 리포트 (Waterfall)
# ═══════════════════════════════════════════════════════════════════════════
# 개별 영상의 성공 확률을 SHAP Waterfall 차트로 시각화

def generate_natural_video_recommendations(sel_row, domain, pos_pairs, neg_pairs):
    """
    선택된 영상의 SHAP 긍정/부정 요인과 영상 메타데이터를 바탕으로
    FnB / IT 분석 결과에 맞춘 자연스러운 개선 추천 문장을 생성합니다.
    """

    def safe_get(col, default="-"):
        val = sel_row.get(col, default)
        try:
            if pd.isna(val):
                return default
        except Exception:
            pass
        return val

    def contains_any(text, keywords):
        text = str(text)
        return any(k in text for k in keywords)

    def label_feature(x):
        """컬럼/feature명을 한글 라벨로 변환"""
        if x is None:
            return "-"
        return LABEL_MAP.get(str(x), str(x))

    def label_length(x):
        """length_bucket 값을 한글 라벨로 변환"""
        if x is None:
            return "-"
        return LENGTH_BUCKET_MAP.get(str(x), str(x))

    def label_time(x):
        """upload_time_bucket 값을 한글 라벨로 변환"""
        if x is None:
            return "-"
        return TIME_BUCKET_MAP.get(str(x), str(x))

    # ─────────────────────────────
    # 선택 영상 정보
    # ─────────────────────────────
    desc_len = safe_get("description_length", None)
    cta_type = safe_get("cls_cta_type", "-")
    content_type = safe_get("cls_content_type", "-")
    marketing_purpose = safe_get("cls_marketing_purpose", "-")
    upload_day = safe_get("upload_dayofweek", "-")
    upload_time_bucket = safe_get("upload_time_bucket", "-")
    length_bucket = safe_get("length_bucket", "-")
    category_name = safe_get("category_name", "-")

    # 화면 표시용 매핑값
    upload_time_bucket_label = label_time(upload_time_bucket)
    length_bucket_label = label_length(length_bucket)

    short_label = label_length("short")
    standard_label = label_length("standard")
    mid_ads_label = label_length("mid_ads")
    long_form_label = label_length("long_form")

    morning_label = label_time("morning")
    lunch_label = label_time("lunch")
    evening_label = label_time("evening")
    night_label = label_time("night")

    strongest_neg = neg_pairs[0][0] if len(neg_pairs) > 0 else None
    strongest_pos = pos_pairs[0][0] if len(pos_pairs) > 0 else None

    strongest_neg_label = label_feature(strongest_neg)
    strongest_pos_label = label_feature(strongest_pos)

    recommendations = []

    # ─────────────────────────────
    # 분석 결과 기반 도메인별 기준
    # ※ rules 안의 값은 원본 데이터 값 그대로 둬야 비교가 됩니다.
    # ─────────────────────────────
    DOMAIN_RULES = {
        "FnB": {
            "cta_good": ["방문유도", "기타", "이벤트참여"],
            "cta_bad": ["구매유도", "정보탐색"],
            "content_good": ["웹예능", "기술설명", "에피소드소개", "시설소개", "브이로그"],
            "content_bad": ["제품리뷰", "이벤트/행사", "영양정보", "요리/레시피", "웹드라마", "인터뷰"],
            "purpose_good": ["채용", "브랜드캠페인", "사회공헌/환경", "기업이미지", "서비스활용"],
            "purpose_bad": ["정보제공", "제품홍보", "고객유지"],
            "length_good": ["long_form"],
            "length_bad": ["short", "standard", "mid_ads"],
            "day_good": ["금요일"],
            "day_bad": ["일요일", "수요일", "화요일", "목요일"],
            "time_good": ["morning", "evening"],
            "time_bad": ["lunch", "night"],
        },
        "IT": {
            "cta_good": ["기타", "이벤트참여", "구매유도", "구독유도", "앱다운로드"],
            "cta_bad": ["정보탐색"],
            "content_good": ["브이로그", "인터뷰", "웹드라마", "이벤트/행사", "웹예능", "제품리뷰", "기타", "튜토리얼"],
            "content_bad": ["기술설명", "에피소드소개"],
            "purpose_good": ["채용", "브랜드캠페인", "고객유지", "정보제공", "기업이미지", "제품홍보"],
            "purpose_bad": ["사회공헌/환경", "서비스활용", "고객유입"],
            "length_good": ["mid_ads"],
            "length_bad": ["short", "standard", "long_form"],
            "day_good": ["수요일", "일요일", "월요일", "목요일", "금요일"],
            "day_bad": ["토요일", "화요일"],
            "time_good": ["morning"],
            "time_bad": ["lunch", "night"],
        }
    }

    rules = DOMAIN_RULES.get(domain, DOMAIN_RULES["FnB"])

    # ─────────────────────────────
    # 1. 가장 큰 부정 요인 기반 추천
    # ─────────────────────────────
    if strongest_neg:
        neg_name = str(strongest_neg)

        if contains_any(neg_name, ["CTA"]):
            if domain == "FnB":
                if cta_type in rules["cta_bad"]:
                    recommendations.append(
                        f"이 영상의 CTA는 '{cta_type}'입니다. FnB 분석에서는 '{cta_type}'보다 "
                        "'방문유도', '이벤트참여', '기타' CTA가 더 긍정적으로 작용했습니다. "
                        "단순 구매 요청보다는 매장 방문, 체험, 이벤트 참여처럼 시청자가 부담 없이 행동할 수 있는 문구로 바꿔보세요."
                    )
                else:
                    recommendations.append(
                        f"CTA 유형은 '{cta_type}'입니다. FnB에서는 행동 유도형 CTA의 영향이 특히 크게 나타났으므로, "
                        "설명란과 고정 댓글에 '방문하기', '이벤트 참여하기', '자세히 보기'처럼 행동을 명확히 유도하는 문장을 함께 배치해보세요."
                    )
            else:
                if cta_type in rules["cta_bad"]:
                    recommendations.append(
                        f"이 영상의 CTA는 '{cta_type}'입니다. IT 분석에서는 '정보탐색' CTA가 부정적으로 나타났고, "
                        "'기타', '이벤트참여', '구매유도', '구독유도', '앱다운로드' CTA가 더 긍정적으로 나타났습니다. "
                        "단순히 정보를 더 보게 하기보다 데모 확인, 신청, 다운로드, 이벤트 참여처럼 다음 행동이 분명한 CTA를 테스트해보세요."
                    )
                else:
                    recommendations.append(
                        f"CTA 유형은 '{cta_type}'입니다. IT에서는 시청자가 바로 다음 행동을 할 수 있는 CTA가 긍정적으로 나타났습니다. "
                        "영상 말미뿐 아니라 설명란과 고정 댓글에도 같은 CTA를 반복 노출하면 전환 흐름을 더 명확히 만들 수 있습니다."
                    )

        elif contains_any(neg_name, ["콘텐츠 유형"]):
            if domain == "FnB":
                if content_type in rules["content_bad"]:
                    recommendations.append(
                        f"현재 콘텐츠 유형은 '{content_type}'입니다. FnB 분석에서는 제품리뷰, 이벤트/행사, 영양정보, 요리/레시피보다 "
                        "웹예능, 기술설명, 에피소드소개, 시설소개, 브이로그 유형이 더 긍정적으로 나타났습니다. "
                        "제품 설명 중심이라면 실제 경험 장면, 공간 분위기, 에피소드형 흐름을 더해 콘텐츠를 조금 더 체험형으로 바꿔보세요."
                    )
                else:
                    recommendations.append(
                        f"현재 콘텐츠 유형은 '{content_type}'입니다. FnB에서는 웹예능, 시설소개, 브이로그처럼 경험과 분위기가 드러나는 유형이 긍정적으로 나타났습니다. "
                        "다음 영상에서도 제품 자체보다 사용 장면과 브랜드 경험이 먼저 보이도록 구성해보세요."
                    )
            else:
                if content_type in rules["content_bad"]:
                    recommendations.append(
                        f"현재 콘텐츠 유형은 '{content_type}'입니다. IT 분석에서는 기술설명과 에피소드소개가 상대적으로 부정적으로 나타났고, "
                        "브이로그, 인터뷰, 웹드라마, 이벤트/행사, 웹예능 유형이 더 긍정적으로 나타났습니다. "
                        "기능 설명만 나열하기보다 실제 사용 상황, 인터뷰, 사례 중심의 흐름으로 바꿔보세요."
                    )
                else:
                    recommendations.append(
                        f"현재 콘텐츠 유형은 '{content_type}'입니다. IT에서는 브이로그, 인터뷰, 이벤트/행사처럼 정보가 상황 속에서 전달되는 형식이 긍정적으로 나타났습니다. "
                        "다음 영상에서도 문제 상황과 활용 장면을 함께 보여주는 구성을 유지해보세요."
                    )

        elif contains_any(neg_name, ["마케팅 목적"]):
            if domain == "FnB":
                if marketing_purpose in rules["purpose_bad"]:
                    recommendations.append(
                        f"이 영상의 마케팅 목적은 '{marketing_purpose}'입니다. FnB 분석에서는 정보제공, 제품홍보, 고객유지 목적보다 "
                        "채용, 브랜드캠페인, 사회공헌/환경, 기업이미지 목적이 더 긍정적으로 나타났습니다. "
                        "단순 제품 홍보보다 브랜드가 추구하는 가치, 공간의 분위기, 캠페인 메시지가 드러나도록 목적을 재정리해보세요."
                    )
                else:
                    recommendations.append(
                        f"마케팅 목적은 '{marketing_purpose}'입니다. FnB에서는 브랜드캠페인, 기업이미지, 사회공헌/환경처럼 브랜드 인식과 감성을 만드는 목적이 긍정적으로 나타났습니다. "
                        "영상의 도입부에서 제품보다 브랜드 메시지가 먼저 보이도록 구성하면 좋습니다."
                    )
            else:
                if marketing_purpose in rules["purpose_bad"]:
                    recommendations.append(
                        f"이 영상의 마케팅 목적은 '{marketing_purpose}'입니다. IT 분석에서는 사회공헌/환경, 서비스활용, 고객유입 목적이 상대적으로 부정적으로 나타났고, "
                        "채용, 브랜드캠페인, 고객유지, 정보제공, 기업이미지 목적이 긍정적으로 나타났습니다. "
                        "서비스 사용법만 강조하기보다 브랜드 신뢰, 정보 제공, 고객 유지 관점의 메시지를 강화해보세요."
                    )
                else:
                    recommendations.append(
                        f"마케팅 목적은 '{marketing_purpose}'입니다. IT에서는 채용, 브랜드캠페인, 고객유지, 정보제공 목적이 긍정적으로 나타났습니다. "
                        "영상 초반에 이 영상이 어떤 문제를 해결하고 어떤 신뢰를 줄 수 있는지 더 분명히 보여주세요."
                    )

        elif contains_any(neg_name, ["업로드 요일"]):
            if domain == "FnB":
                recommendations.append(
                    f"현재 업로드 요일은 '{upload_day}'입니다. FnB 분석에서는 금요일 업로드가 긍정적으로 나타났고, "
                    "일요일은 가장 부정적으로 나타났습니다. 비슷한 콘텐츠를 금요일 업로드와 비교 테스트해보는 것이 좋습니다."
                )
            else:
                recommendations.append(
                    f"현재 업로드 요일은 '{upload_day}'입니다. IT 분석에서는 수요일, 일요일, 월요일 업로드가 긍정적으로 나타났고, "
                    "토요일과 화요일은 부정적으로 나타났습니다. 다음 영상은 수요일 또는 일요일 업로드를 우선 테스트해보세요."
                )

        elif contains_any(neg_name, ["업로드 시간", "업로드 시간대"]):
            if domain == "FnB":
                recommendations.append(
                    f"현재 업로드 시간대는 '{upload_time_bucket_label}'입니다. FnB 분석에서는 {lunch_label}, {night_label} 시간대가 부정적으로 나타났고, "
                    f"{morning_label}/{evening_label} 시간대가 상대적으로 더 유리하게 보입니다. 다음 영상은 오전 또는 저녁 시간대와 비교 테스트해보세요."
                )
            else:
                recommendations.append(
                    f"현재 업로드 시간대는 '{upload_time_bucket_label}'입니다. IT 분석에서는 {lunch_label}과 {night_label}이 부정적으로, {morning_label}이 상대적으로 긍정적으로 나타났습니다. "
                    "정보 탐색형 시청자가 집중하기 쉬운 오전 업로드를 테스트해보세요."
                )

        elif contains_any(neg_name, ["영상 길이", "영상 길이 구간"]):
            if domain == "FnB":
                recommendations.append(
                    f"현재 영상 길이 구간은 '{length_bucket_label}'입니다. FnB 분석에서는 {long_form_label}이 강하게 긍정적으로 나타났고, "
                    f"{short_label}, {standard_label}, {mid_ads_label}은 부정적으로 나타났습니다. "
                    "FnB 콘텐츠는 제품 경험, 공간 분위기, 브랜드 스토리를 충분히 보여주는 긴 호흡의 영상으로 확장해보세요."
                )
            else:
                recommendations.append(
                    f"현재 영상 길이 구간은 '{length_bucket_label}'입니다. IT 분석에서는 {mid_ads_label}의 영상이 긍정적으로 나타났고, "
                    f"{long_form_label}은 부정적으로 나타났습니다. 너무 짧게 끝내기보다 핵심 문제, 기능 설명, 활용 예시를 담되 과하게 길어지지 않는 중간 길이로 조정해보세요."
                )

        elif contains_any(neg_name, ["설명 길이"]):
            recommendations.append(
                "설명 길이가 성공 예측에 부정적으로 작용한 것으로 보입니다. "
                "영상 요약, 핵심 키워드, CTA, 관련 링크를 설명란 앞부분에 정리해서 검색성과 행동 유도를 함께 보강해보세요."
            )

        elif contains_any(neg_name, ["카테고리"]):
            recommendations.append(
                f"현재 카테고리는 '{category_name}'입니다. 카테고리 영향이 부정적으로 나타났다면, "
                "영상 제목·설명·태그가 실제 콘텐츠 주제와 더 명확히 연결되도록 정리하고, 같은 도메인의 성공 영상에서 자주 나타난 카테고리 구조를 참고해보세요."
            )

        else:
            recommendations.append(
                f"가장 크게 아쉬운 요인은 '{strongest_neg_label}'입니다. "
                "다음 영상에서는 이 요소를 우선적으로 점검하고, 같은 도메인의 성공 영상과 비교해 차이를 확인해보는 것이 좋습니다."
            )

    # ─────────────────────────────
    # 2. 실제 현재 값이 분석상 불리한 값일 때 보완 추천
    # ─────────────────────────────
    if cta_type in rules["cta_bad"]:
        if domain == "FnB":
            recommendations.append(
                f"현재 CTA '{cta_type}'은 FnB에서 상대적으로 불리하게 나타난 유형입니다. "
                "방문유도나 이벤트참여처럼 오프라인 경험 또는 참여 행동으로 이어지는 CTA를 우선 테스트해보세요."
            )
        else:
            recommendations.append(
                f"현재 CTA '{cta_type}'은 IT에서 상대적으로 불리하게 나타난 유형입니다. "
                "자료 다운로드, 데모 확인, 구독, 이벤트 참여처럼 다음 행동이 분명한 CTA로 바꿔보세요."
            )

    if content_type in rules["content_bad"]:
        if domain == "FnB":
            recommendations.append(
                f"현재 콘텐츠 유형 '{content_type}'은 FnB 분석에서 상대적으로 낮게 나타났습니다. "
                "제품 정보만 전달하기보다 웹예능, 브이로그, 시설소개처럼 경험 중심 포맷으로 전환해보세요."
            )
        else:
            recommendations.append(
                f"현재 콘텐츠 유형 '{content_type}'은 IT 분석에서 상대적으로 낮게 나타났습니다. "
                "기술 설명을 그대로 전달하기보다 인터뷰, 사례, 이벤트/행사형 구성으로 정보 전달 방식을 바꿔보세요."
            )

    if length_bucket in rules["length_bad"]:
        if domain == "FnB":
            recommendations.append(
                f"현재 길이 구간 '{length_bucket_label}'은 FnB에서 부정적으로 나타났습니다. "
                f"FnB는 {long_form_label}의 영향이 가장 긍정적이었으므로, 제품 경험과 브랜드 스토리를 더 충분히 담는 방향을 고려해보세요."
            )
        else:
            recommendations.append(
                f"현재 길이 구간 '{length_bucket_label}'은 IT에서 부정적으로 나타났습니다. "
                f"IT는 {mid_ads_label} 구간이 긍정적으로 나타났으므로, 핵심 정보는 유지하되 너무 짧거나 길지 않게 편집해보세요."
            )

    if upload_time_bucket in rules["time_bad"]:
        if domain == "FnB":
            recommendations.append(
                f"현재 업로드 시간대 '{upload_time_bucket_label}'은 FnB에서 상대적으로 불리하게 나타났습니다. "
                f"{morning_label} 또는 {evening_label} 시간대 업로드를 비교 테스트해보세요."
            )
        else:
            recommendations.append(
                f"현재 업로드 시간대 '{upload_time_bucket_label}'은 IT에서 상대적으로 불리하게 나타났습니다. "
                f"{morning_label} 시간대 업로드를 우선 테스트해보세요."
            )

    # ─────────────────────────────
    # 3. 설명 길이 보완 추천
    # ─────────────────────────────
    try:
        desc_len_num = int(desc_len)

        if desc_len_num < 100:
            recommendations.append(
                f"현재 설명 길이는 약 {desc_len_num}자로 짧은 편입니다. "
                "영상 핵심 요약, 브랜드/제품명, 주요 키워드, CTA 링크를 추가해 최소 100자 이상으로 보강해보세요."
            )
        elif desc_len_num < 300:
            recommendations.append(
                f"현재 설명 길이는 약 {desc_len_num}자입니다. 기본 정보는 있지만, "
                "검색 유입과 CTA 연결을 위해 핵심 키워드와 영상 요약을 조금 더 구체적으로 추가하면 좋습니다."
            )
    except Exception:
        pass

    # ─────────────────────────────
    # 4. 긍정 요인 유지 추천
    # ─────────────────────────────
    if strongest_pos:
        recommendations.append(
            f"반대로 '{strongest_pos_label}'은 이 영상의 긍정 요인으로 작용했습니다. "
            "다음 영상에서도 이 강점은 유지하되, 위의 부정 요인을 함께 개선하는 방향이 좋습니다."
        )

    # ─────────────────────────────
    # 5. 도메인별 마무리 추천
    # ─────────────────────────────
    if domain == "FnB":
        recommendations.append(
            "FnB 영상은 단순 제품 홍보보다 브랜드 경험, 공간 분위기, 캠페인 메시지, 방문 행동 유도가 중요하게 나타났습니다. "
            "다음 영상에서는 제품 자체보다 시청자가 경험하고 싶어지는 장면을 먼저 보여주세요."
        )
    else:
        recommendations.append(
            "IT 영상은 단순 기능 설명보다 시청자가 바로 이해할 수 있는 문제 상황, 활용 장면, 다음 행동 CTA가 중요하게 나타났습니다. "
            "다음 영상에서는 '무엇을 해결해주는지'와 '다음에 무엇을 하면 되는지'를 더 분명히 보여주세요."
        )

    # 중복 제거 후 3개만 반환
    recommendations = list(dict.fromkeys(recommendations))
    return recommendations[:3]


def page_waterfall():
    section_header(5, "영상 진단 리포트",
                   "개별 영상의 성공 가능성과 각 특성이 미치는 영향(성공 기여도)을 한눈에 확인하세요.")

    # 페이지 첫 등장 시 '기여도 흐름 차트' 개념 설명
    concept_box(
        "기여도 흐름 차트(Waterfall)란?",
        "왼쪽의 <b>'평균 성공 확률'</b>에서 시작해, 이 영상의 각 요소가 확률을 "
        "<b>얼마나 올렸는지(초록 ↑)</b> 또는 <b>내렸는지(빨강 ↓)</b> 단계별로 보여줍니다. "
        "오른쪽 끝의 막대가 이 영상의 <b>최종 예측 확률</b>입니다."
    )

    # 영상 목록 구성
    # 영상 목록 구성: 도메인/확률구간/정렬 기준에 따라 선택
    pred = D["pred"].copy()

    # 안전 처리
    pred["domain"] = pred["domain"].astype(str).str.strip()
    pred["title"] = pred["title"].astype(str)
    pred["success_probability"] = pd.to_numeric(
        pred["success_probability"],
        errors="coerce"
    )

    # 확률 결측 제거
    pred = pred.dropna(subset=["success_probability"])

    col_left, col_right = st.columns([1.2, 2.8])

    # ──── 좌측: 도메인 선택 + 확률 구간 필터 + 영상 선택 + 요약 + 확률 ────
    with col_left:
        st.subheader("분석할 영상 선택")

        # 1) 도메인 선택
        selected_domain = st.radio(
            "도메인 선택",
            ["FnB", "IT"],
            horizontal=True,
            key="waterfall_domain_select"
        )

        # 선택 도메인만 필터링
        pred_domain = pred[pred["domain"] == selected_domain].copy()

        if pred_domain.empty:
            st.warning(f"{selected_domain} 도메인에 해당하는 영상이 없습니다.")
            return

        # 2) 성공 확률 구간 필터
        prob_group = st.selectbox(
            "성공 확률 구간",
            [
                "전체",
                "높음: 70% 이상",
                "중간: 40% 이상 ~ 70% 미만",
                "낮음: 40% 미만",
            ],
            key=f"waterfall_prob_group_{selected_domain}"
        )

        if prob_group == "높음: 70% 이상":
            pred_domain = pred_domain[pred_domain["success_probability"] >= 0.7]

        elif prob_group == "중간: 40% 이상 ~ 70% 미만":
            pred_domain = pred_domain[
                (pred_domain["success_probability"] >= 0.4) &
                (pred_domain["success_probability"] < 0.7)
            ]

        elif prob_group == "낮음: 40% 미만":
            pred_domain = pred_domain[pred_domain["success_probability"] < 0.4]

        if pred_domain.empty:
            st.warning(f"{selected_domain} / {prob_group} 구간에 해당하는 영상이 없습니다.")
            return

        # 3) 평균 대비 차이 계산
        domain_avg_prob = pred_domain["success_probability"].mean()

        pred_domain["prob_diff_from_avg"] = (
            pred_domain["success_probability"] - domain_avg_prob
        )
        pred_domain["abs_prob_diff_from_avg"] = pred_domain["prob_diff_from_avg"].abs()

        # 4) 정렬 기준 선택
        sort_option = st.selectbox(
            "영상 정렬 기준",
            [
                "성공 확률 높은 순",
                "성공 확률 낮은 순",
                "평균 대비 차이 큰 순",
                "평균에 가까운 순",
            ],
            key=f"waterfall_sort_option_{selected_domain}_{prob_group}"
        )

        if sort_option == "성공 확률 높은 순":
            pred_sorted = pred_domain.sort_values(
                "success_probability",
                ascending=False
            )

        elif sort_option == "성공 확률 낮은 순":
            pred_sorted = pred_domain.sort_values(
                "success_probability",
                ascending=True
            )

        elif sort_option == "평균 대비 차이 큰 순":
            pred_sorted = pred_domain.sort_values(
                "abs_prob_diff_from_avg",
                ascending=False
            )

        else:
            pred_sorted = pred_domain.sort_values(
                "abs_prob_diff_from_avg",
                ascending=True
            )

        # 5) selectbox 표시용 이름 만들기
        pred_sorted["display"] = (
            pred_sorted["title"].str[:45]
            + " | "
            + (pred_sorted["success_probability"] * 100).round(1).astype(str)
            + "%"
        )

        selected_display = st.selectbox(
            "영상 선택",
            pred_sorted["display"].tolist(),
            key=f"waterfall_video_select_{selected_domain}_{prob_group}_{sort_option}"
        )

        sel_row = pred_sorted[pred_sorted["display"] == selected_display].iloc[0]

        # 영상 요약 정보 카드
        prob = sel_row["success_probability"]
        domain = sel_row["domain"]

        # length_bucket 한글 변환
        len_bucket_kr = LENGTH_BUCKET_MAP.get(sel_row['length_bucket'], sel_row['length_bucket'])

        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="font-size:12px;color:#6b7280;margin-bottom:12px;font-weight:600;">영상 요약</div>
            <div style="font-size:13px;line-height:2;">
                <span style="color:#6b7280;">{k('domain')}</span> &nbsp;&nbsp;&nbsp;
                <span style="font-weight:600;">{domain}</span><br>
                <span style="color:#6b7280;">{k('cls_content_type')}</span> &nbsp;
                <span style="font-weight:600;">{sel_row['cls_content_type']}</span><br>
                <span style="color:#6b7280;">{k('length_bucket')}</span> &nbsp;&nbsp;
                <span style="font-weight:600;">{len_bucket_kr}</span><br>
                <span style="color:#6b7280;">{k('cls_cta_type')}</span> &nbsp;&nbsp;
                <span style="font-weight:600;">{sel_row['cls_cta_type']}</span><br>
                <span style="color:#6b7280;">{k('upload_dayofweek')}</span>
                <span style="font-weight:600;">{sel_row['upload_dayofweek']}</span><br>
                <span style="color:#6b7280;">{k('description_length')}</span> &nbsp;&nbsp;
                <span style="font-weight:600;">{int(sel_row['description_length'])}자</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # 성공 확률 큰 숫자 표시
        prob_pct = int(prob * 100)
        cls = "prob-high" if prob >= 0.7 else ("prob-mid" if prob >= 0.4 else "prob-low")
        label = "성공 가능성 높음 🚀" if prob >= 0.7 else ("중간 ⚡" if prob >= 0.4 else "낮음 ⚠️")
        st.markdown(f"""
        <div style="text-align:center;padding:20px;background:white;border-radius:12px;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-top:12px;">
            <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">예상 성공 확률</div>
            <div class="prob-value {cls}">{prob_pct}%</div>
            <div style="font-size:13px;margin-top:6px;">{label}</div>
        </div>""", unsafe_allow_html=True)

    # ──── 우측: 기여도 흐름 차트 (SHAP Waterfall) ────
    with col_right:
        subheader_with_tip("성공 기여도 흐름 차트", TIP_WATERFALL)
        # 도메인별 SHAP 모델 선택
        m_dict = M["shap_fnb"] if domain == "FnB" else M["shap_it"]
        model = m_dict["model"]
        X_test = m_dict["X_test"]

        # row_id로 X_test에서 해당 영상 행 찾기
        row_id = sel_row["row_id"]
        if row_id < len(X_test):
            sample = X_test.iloc[[row_id]]
        else:
            sample = X_test.iloc[[0]]

        # SHAP 계산 + Waterfall 생성
        labels, values = [], []  # 외부 try/except에서도 접근
        try:
            import shap as shap_lib
            explainer = shap_lib.TreeExplainer(model)
            shap_vals = explainer(sample)
            sv = shap_vals.values[0]
            feats = sample.columns.tolist()

            # 영향 큰 순으로 TOP 8 추출
            # 주의: TreeExplainer의 base_value / shap_value는 모델 내부 점수(raw margin/logit)일 수 있어
            # 그대로 %로 표시하면 -55% 같은 비정상 확률이 나올 수 있습니다.
            # 따라서 화면용 Waterfall은 실제 예측 확률 구간(도메인 평균 → 선택 영상 예측 확률)에 맞게
            # SHAP 방향/상대 크기를 확률 기여도 형태로 정규화해 표시합니다.
            pairs = sorted(zip(feats, sv), key=lambda x: abs(float(x[1])), reverse=True)[:8]
            labels = [k_norm(f) for f, _ in pairs]
            raw_values = [float(v) for _, v in pairs]

            base = calc_avg_success_probability(domain)
            if pd.isna(base):
                base = pred_domain["success_probability"].mean()
            base = float(np.clip(base, 0.0, 1.0))

            final_prob = float(np.clip(sel_row["success_probability"], 0.0, 1.0))
            diff_prob = final_prob - base

            raw_sum = float(np.sum(raw_values))
            if abs(raw_sum) > 1e-9:
                values = [v * diff_prob / raw_sum for v in raw_values]
            else:
                values = [0.0 for _ in raw_values]

            # 너무 작은 값은 차트에서 거의 보이지 않으므로 0으로 정리
            values = [0.0 if abs(v) < 0.0005 else float(v) for v in values]

            x_labels = ["기본<br>(평균)"] + labels + ["최종 예측"]
            measures = ["absolute"] + ["relative"] * len(values) + ["absolute"]
            waterfall_vals = [base] + values + [final_prob]

            fig = go.Figure(go.Waterfall(
                x=x_labels, y=waterfall_vals, measure=measures,
                connector=dict(line=dict(color="#d1d5db")),
                increasing=dict(marker_color=COL_GREEN),
                decreasing=dict(marker_color=COL_RED),
                totals=dict(marker_color=COL_GRAY),
                text=[
                    f"{v*100:+.1f}%p" if 0 < i < len(x_labels)-1 else f"{v*100:.1f}%"
                    for i, v in enumerate(waterfall_vals)
                ],
                textposition="outside"
            ))
            fig.update_layout(
                **plotly_base(),
                height=380,
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            base_pct = int(round(base * 100))
            final_pct = int(round(final_prob * 100))
            diff = final_pct - base_pct
            st.markdown(f"""
            <div class="insight-box">
            이 영상은 도메인 평균 대비 <b>{diff:+d}%p</b> {'높은' if diff>0 else '낮은'} 성공 확률을 보입니다.
            (도메인 평균: {base_pct}% → 선택 영상 예측: {final_pct}%)
            </div>""", unsafe_allow_html=True)

            with st.expander("차트 계산 방식 확인", expanded=False):
                st.caption(
                    "SHAP 원값은 모델 내부 점수 단위일 수 있어 확률처럼 직접 표시하지 않고, "
                    "도메인 평균 성공 확률에서 선택 영상 예측 확률까지의 차이를 기준으로 정규화해 표시했습니다."
                )
        except Exception as e:
            st.info(f"기여도 계산 중 오류: {e}")


    # ──── 하단: 긍정/부정 TOP 5 + 로컬 설명 ────
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    try:
        # 양/음 분리 후 절댓값 큰 순
        pos_pairs = sorted([(l, v) for l, v in zip(labels, values) if v > 0],
                           key=lambda x: x[1], reverse=True)[:5]
        neg_pairs = sorted([(l, v) for l, v in zip(labels, values) if v < 0],
                           key=lambda x: x[1])[:5]

        # 긍정 영향 TOP 5
        with col1:
            st.subheader("긍정 영향 TOP 5")

            if pos_pairs:
                max_v = max(v for _, v in pos_pairs)

                for lbl, v in pos_pairs:
                    pct = int(v / max_v * 100) if max_v > 0 else 0

                    st.markdown(
                        f"""
        <div style="margin-bottom:8px;min-height:42px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span>{lbl}</span>
                <span style="color:{COL_GREEN};font-weight:700;">+{v:.2f}</span>
            </div>
            <div style="background:#e5e7eb;border-radius:4px;height:8px;margin-top:4px;">
                <div style="width:{pct}%;background:{COL_GREEN};height:8px;border-radius:4px;"></div>
            </div>
        </div>
        """,
                        unsafe_allow_html=True
                    )

            else:
                st.markdown(
                    """
        <div style="
            background:#f9fafb;
            border:1px solid #e5e7eb;
            border-radius:10px;
            padding:16px;
            min-height:210px;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            color:#9ca3af;
            font-size:13px;
        ">
            이 영상에서는 뚜렷한 긍정 영향 요인이 없습니다.
        </div>
        """,
                    unsafe_allow_html=True
                )

        # 부정 영향 TOP 5
        with col2:
            st.subheader("부정 영향 TOP 5")

            if neg_pairs:
                max_v = max(abs(v) for _, v in neg_pairs)

                for lbl, v in neg_pairs:
                    pct = int(abs(v) / max_v * 100) if max_v > 0 else 0

                    st.markdown(
                        f"""
        <div style="margin-bottom:8px;min-height:42px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span>{lbl}</span>
                <span style="color:{COL_RED};font-weight:700;">{v:.2f}</span>
            </div>
            <div style="background:#e5e7eb;border-radius:4px;height:8px;margin-top:4px;">
                <div style="width:{pct}%;background:{COL_RED};height:8px;border-radius:4px;"></div>
            </div>
        </div>
        """,
                        unsafe_allow_html=True
                    )

            else:
                st.markdown(
                    """
        <div style="
            background:#f9fafb;
            border:1px solid #e5e7eb;
            border-radius:10px;
            padding:16px;
            min-height:210px;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            color:#9ca3af;
            font-size:13px;
        ">
            이 영상에서는 뚜렷한 부정 영향 요인이 없습니다.
        </div>
        """,
                    unsafe_allow_html=True
                )

        # 로컬 설명 (전략 추천)
        with col3:
            st.subheader("이 영상은 왜?")

            # 도메인별 기본 전략
            strat = D["strategy"].get(domain, {})
            rec = strat.get("recommended", [])
            avoid = strat.get("avoid", [])

            # 선택 영상 맞춤 추천 생성
            video_recs = generate_natural_video_recommendations(
                sel_row=sel_row,
                domain=domain,
                pos_pairs=pos_pairs,
                neg_pairs=neg_pairs
            )

            # 가장 큰 긍정/부정 요인 표시
            strongest_pos_txt = "-"
            strongest_neg_txt = "-"
            if len(pos_pairs) > 0:
                pos_name = LABEL_MAP.get(str(pos_pairs[0][0]), str(pos_pairs[0][0]))
                strongest_pos_txt = f"{pos_name} ({pos_pairs[0][1]:+.2f})"

            if len(neg_pairs) > 0:
                neg_name = LABEL_MAP.get(str(neg_pairs[0][0]), str(neg_pairs[0][0]))
                strongest_neg_txt = f"{neg_name} ({neg_pairs[0][1]:+.2f})"

            st.markdown(
                f"""
        <div style="background:#ffffff;border-radius:10px;padding:12px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06);font-size:12px;line-height:1.7;min-height:120px;">
            <b>📌 영상별 핵심 진단</b><br>
            <span style="color:#16a34a;">가장 큰 긍정 요인</span>: {strongest_pos_txt}<br>
            <span style="color:#dc2626;">가장 큰 부정 요인</span>: {strongest_neg_txt}
        </div>
        """,
                unsafe_allow_html=True
            )

            # 도메인 기본 전략만 col3에 유지
            st.markdown(
                f"""
        <div style="font-size:12px;color:#6b7280;margin-top:12px;margin-bottom:4px;">
            <b>{domain} 도메인 참고 전략</b>
        </div>
        """,
                unsafe_allow_html=True
            )

            for r in rec[:2]:
                st.markdown(
                    f'<div style="font-size:12px;padding:3px 0;">✅ {r}</div>',
                    unsafe_allow_html=True
                )

            for a in avoid[:1]:
                st.markdown(
                    f'<div style="font-size:12px;padding:3px 0;color:#ef4444;">❌ {a}</div>',
                    unsafe_allow_html=True
                )


        # ─────────────────────────────────────────
        # 세 컬럼 아래: 영상 맞춤 개선 추천
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔵 이 영상 맞춤 개선 추천")
        st.caption("선택한 영상의 SHAP 기여도와 도메인별 분석 결과를 바탕으로 개선 방향을 제안합니다.")

        # 추천이 3개보다 적을 경우를 대비
        if not video_recs:
            st.info("이 영상에 대한 맞춤 추천을 생성할 수 없습니다.")
        else:
            rec_cols = st.columns(len(video_recs))

            for i, txt in enumerate(video_recs):
                with rec_cols[i]:
                    with st.container(border=True):
                        st.markdown(f"#### 추천 {i+1}")

                        sentences = re.split(r"(?<=[.!?。])\s+", txt.strip())

                        for sentence in sentences:
                            if sentence.strip():
                                st.markdown(f"- {sentence.strip()}")

    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# [7-5] 운영 전략 가이드
# ═══════════════════════════════════════════════════════════════════════════
# 업로드 요일/시간/CTA/설명길이 4가지 측면에서 최적 전략 가이드
def page_strategy():
    section_header(4, "업로드 전략 추천",
                   "업로드 요일, 시간, CTA 전략 등 영상 성과를 높이는 최적의 운영 전략을 확인하세요.")

    tab1, tab2, tab3, tab4 = st.tabs(["📅 업로드 요일", "⏰ 업로드 시간", "🎯 CTA 전략", "📝 설명 길이"])

    # ─────── 탭 1: 업로드 요일 전략 ───────
    with tab1:
        eda_d = D["eda_day"].copy()
        col1, col2, col3 = st.columns([2, 1, 1.5])

        with col1:
            subheader_with_tip("업로드 요일별 평균 성공 기여도", TIP_CONTRIBUTION)
            domain_filter = st.selectbox("비교 도메인", ["전체","FnB","IT"], key="day_dom")

            # SHAP 기반 요일 영향도 계산
            cat_shap = D["cat_shap"].copy()
            day_shap = cat_shap[cat_shap["feature"]=="upload_dayofweek"].copy()
            sub = day_shap[day_shap["dataset"]=="ALL"] if domain_filter == "전체" \
                  else day_shap[day_shap["dataset"]==domain_filter]

            # 데이터 없으면 EDA로 대체
            if sub.empty:
                sub_eda = eda_d if domain_filter=="전체" else eda_d[eda_d["domain"]==domain_filter]
                sub_g = sub_eda.groupby("upload_dayofweek")["success_rate"].mean().reset_index()
                sub_g["mean_shap"] = sub_g["success_rate"] - sub_g["success_rate"].mean()
                sub = sub_g.rename(columns={"upload_dayofweek":"category"})

            # 요일 순서대로 정렬
            day_order = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
            sub = sub.set_index("category").reindex(day_order).reset_index()
            sub["mean_shap"] = sub["mean_shap"].fillna(0)
            colors = [COL_GREEN if v>=0 else COL_RED for v in sub["mean_shap"]]
            fig = go.Figure(go.Bar(
                x=sub["category"], y=sub["mean_shap"],
                marker_color=colors,
                text=[f"{v:+.2f}" for v in sub["mean_shap"]],
                textposition="outside"
            ))
            fig.update_layout(**plotly_base(), height=300, yaxis_title="평균 성공 기여도")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class="insight-box">금요일 업로드가 평균적으로 가장 긍정적인 영향을 주며,
            일요일 업로드는 성과에 부정적인 영향을 미치는 경향이 있습니다.</div>""", unsafe_allow_html=True)

        with col2:
            st.subheader("추천 순위 (요일)")
            ranks = [("🥇","금요일","+0.42"),("🥈","토요일","+0.15"),("🥉","화요일","+0.08"),
                     ("4","월요일","+0.05"),("5","목요일","+0.01"),("6","수요일","-0.05"),("7","일요일","-0.45")]
            for icon, day, shap_v in ranks:
                color = COL_GREEN if "+" in shap_v else COL_RED
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 0;border-bottom:1px solid #f3f4f6;font-size:13px;">
                    <span>{icon} {day}</span>
                    <span style="color:{color};font-weight:700;">{shap_v}</span>
                </div>""", unsafe_allow_html=True)

        with col3:
            st.subheader("실제 요일별 성공률")
            sub_r = eda_d if domain_filter=="전체" else eda_d[eda_d["domain"]==domain_filter]
            sub_r = sub_r.groupby("upload_dayofweek")["success_rate"].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=sub_r["upload_dayofweek"], y=sub_r["success_rate"]*100,
                marker_color=COL_PURPLE, opacity=0.85,
                text=[f"{v*100:.1f}%" for v in sub_r["success_rate"]],
                textposition="outside"
            ))
            fig.update_layout(**plotly_base(), height=300, yaxis_ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    # ─────── 탭 2: 업로드 시간 (히트맵 - 잘림 방지) ───────
    with tab2:
        st.subheader("업로드 시간대별 평균 성공률")
        eda_h = D["eda_hour"].copy()

        domain_filter_time = st.selectbox(
            "비교 도메인",
            ["전체", "FnB", "IT"],
            key="time_dom"
        )

        col1, col2 = st.columns([2, 1.5])

        with col1:
            time_buckets = ["00-03시", "04-07시", "08-11시", "12-15시", "16-19시", "20-23시"]

            def bucket(h):
                if h < 4:
                    return "00-03시"
                elif h < 8:
                    return "04-07시"
                elif h < 12:
                    return "08-11시"
                elif h < 16:
                    return "12-15시"
                elif h < 20:
                    return "16-19시"
                else:
                    return "20-23시"

            if domain_filter_time == "전체":
                sub_h = eda_h.copy()
                chart_title = "전체 시간대별 성공률"
                y_label = ["전체"]
            else:
                sub_h = eda_h[eda_h["domain"] == domain_filter_time].copy()
                chart_title = f"{domain_filter_time} 시간대별 성공률"
                y_label = [domain_filter_time]

            sub_h["tb"] = sub_h["upload_hour"].apply(bucket)
            g = sub_h.groupby("tb")["success_rate"].mean().reindex(time_buckets).fillna(0.5)

            z = [g.values.tolist()]

            fig = go.Figure(go.Heatmap(
                z=z,
                x=time_buckets,
                y=y_label,
                colorscale="RdYlGn",
                text=[[f"{v*100:.0f}%" for v in g.values]],
                texttemplate="%{text}",
                textfont={"size": 13},
                showscale=True,
                zmin=0.3,
                zmax=0.9
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Noto Sans KR", size=12),
                height=220,
                margin=dict(l=50, r=50, t=50, b=80),
                title=chart_title,
                yaxis=dict(automargin=True)
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            best_time = g.idxmax()
            best_rate = g.max() * 100
            worst_time = g.idxmin()
            worst_rate = g.min() * 100

            if domain_filter_time == "FnB":
                time_tip = "FnB는 제품 경험과 브랜드 분위기가 잘 전달되는 시간대를 중심으로 업로드 테스트를 해보는 것이 좋습니다."
            elif domain_filter_time == "IT":
                time_tip = "IT는 정보 탐색과 업무 맥락에서 집중도가 높은 시간대를 중심으로 업로드 테스트를 해보는 것이 좋습니다."
            else:
                time_tip = "전체 기준으로는 도메인별 차이가 섞여 있으므로, FnB와 IT를 나누어 함께 확인하는 것이 좋습니다."

            st.markdown(f"""
            <div style="background:#fdf4ff;border-radius:12px;padding:14px;">
                <div style="font-weight:700;color:#7c3aed;margin-bottom:8px;">⏰ 추천 업로드 시간</div>
                <div style="font-size:13px;color:#374151;line-height:1.8;">
                    <b>현재 선택 도메인:</b> {domain_filter_time}<br>
                    <b>가장 높은 시간대:</b> {best_time} ({best_rate:.1f}%)<br>
                    <b>가장 낮은 시간대:</b> {worst_time} ({worst_rate:.1f}%)<br><br>
                    {time_tip}
                </div>
            </div>""", unsafe_allow_html=True)

    # ─────── 탭 3: CTA 전략 ───────
    with tab3:
        subheader_with_tip("CTA 유형별 성공 기여도", TIP_CONTRIBUTION)

        eda_c = D["eda_cta"].copy()
        cat_shap = D["cat_shap"].copy()

        domain_filter_cta = st.selectbox(
            "비교 도메인",
            ["전체", "FnB", "IT"],
            key="cta_dom"
        )

        if domain_filter_cta == "전체":
            eda_c_sub = eda_c.copy()
            shap_dataset = "ALL"
            chart_domain_title = "전체"
        else:
            eda_c_sub = eda_c[eda_c["domain"] == domain_filter_cta].copy()
            shap_dataset = domain_filter_cta
            chart_domain_title = domain_filter_cta

        col1, col2, col3 = st.columns([1.5, 1.5, 1])

        # CTA 유형 분포 도넛
        with col1:
            cta_dist = eda_c_sub.groupby("cls_cta_type")["count"].sum().reset_index()

            fig = go.Figure(go.Pie(
                labels=cta_dist["cls_cta_type"],
                values=cta_dist["count"],
                hole=0.5,
                marker_colors=[COL_PURPLE, COL_BLUE, COL_GREEN, "#f59e0b", "#ec4899", COL_GRAY, "#06b6d4"]
            ))

            fig.update_layout(
                **plotly_base(),
                height=280,
                title=f"{chart_domain_title} CTA 유형 분포"
            )

            st.plotly_chart(fig, use_container_width=True)

        # CTA 유형별 SHAP
        with col2:
            cta_shap = cat_shap[cat_shap["feature"] == "cls_cta_type"].copy()
            sub = cta_shap[cta_shap["dataset"] == shap_dataset].sort_values("mean_shap", ascending=False)

            # 혹시 해당 도메인 SHAP이 없으면 ALL로 대체
            if sub.empty:
                sub = cta_shap[cta_shap["dataset"] == "ALL"].sort_values("mean_shap", ascending=False)

            colors = [COL_GREEN if v >= 0 else COL_RED for v in sub["mean_shap"]]

            fig = go.Figure(go.Bar(
                x=sub["category"],
                y=sub["mean_shap"],
                marker_color=colors,
                text=[f"{v:+.2f}" for v in sub["mean_shap"]],
                textposition="outside"
            ))

            fig.update_layout(
                **plotly_base(),
                height=280,
                title=f"{chart_domain_title} CTA 유형별 성공 기여도",
                yaxis_title="평균 성공 기여도"
            )

            st.plotly_chart(fig, use_container_width=True)

            best_cta = sub.iloc[0]["category"] if not sub.empty else "-"
            st.markdown(f"""
            <div class="insight-box">
            현재 선택한 <b>{chart_domain_title}</b> 기준으로는 <b>{best_cta}</b> CTA가 가장 긍정적인 성공 기여도를 보입니다.
            도메인별로 효과적인 CTA 유형이 다를 수 있으므로 전체 기준만 보지 않는 것이 좋습니다.
            </div>""", unsafe_allow_html=True)

        # CTA 유형 성공률 순위
        with col3:
            st.subheader("CTA 성공률 순위")

            cta_rank = (
                eda_c_sub.groupby("cls_cta_type")["success_rate"]
                .mean()
                .sort_values(ascending=False)
            )

            for cta, rate in cta_rank.items():
                bar_w = int(rate * 100)
                color = COL_GREEN if rate > 0.5 else COL_RED

                st.markdown(f"""
                <div style="margin-bottom:8px;font-size:12px;">
                    <div style="display:flex;justify-content:space-between;">
                        <span>{cta}</span><span style="color:{color};">{rate*100:.1f}%</span>
                    </div>
                    <div style="background:#e5e7eb;border-radius:3px;height:6px;margin-top:3px;">
                        <div style="width:{bar_w}%;background:{color};height:6px;border-radius:3px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ─────── 탭 4: 설명 길이 전략 ───────
    with tab4:
        subheader_with_tip("설명 길이별 평균 성공률", TIP_CONTRIBUTION)

        domain_filter_desc = st.selectbox(
            "비교 도메인",
            ["전체", "FnB", "IT"],
            key="desc_dom"
        )

        if domain_filter_desc == "전체":
            df_desc_sub = D["df_all"].copy()
            chart_domain_title = "전체"

        elif domain_filter_desc == "FnB":
            df_desc_sub = D["df_fnb"].copy()
            chart_domain_title = "FnB"

        else:
            df_desc_sub = D["df_it"].copy()
            chart_domain_title = "IT"

        # 안전 처리
        df_desc_sub["description_length"] = pd.to_numeric(
            df_desc_sub["description_length"],
            errors="coerce"
        )

        df_desc_sub["grade"] = pd.to_numeric(
            df_desc_sub["grade"],
            errors="coerce"
        )

        df_desc_sub = df_desc_sub.dropna(subset=["description_length", "grade"])

        if df_desc_sub.empty:
            st.warning(f"{chart_domain_title} 데이터에 설명 길이 또는 성공 여부 값이 없습니다.")
            return

        # 설명 길이 구간 생성
        bins = [0, 50, 100, 200, 300, 500, 1000, np.inf]
        labels = ["0-50", "50-100", "100-200", "200-300", "300-500", "500-1000", "1000+"]

        df_desc_sub["desc_bucket"] = pd.cut(
            df_desc_sub["description_length"],
            bins=bins,
            labels=labels,
            right=False
        )

        desc_summary = (
            df_desc_sub.groupby("desc_bucket", observed=False)
            .agg(
                success_rate=("grade", "mean"),
                count=("grade", "size")
            )
            .reset_index()
        )

        # 평균 대비 차이를 성공 기여도처럼 표시
        avg_success = df_desc_sub["grade"].mean()
        desc_summary["contribution_like"] = desc_summary["success_rate"] - avg_success

        col1, col2 = st.columns([2, 1])

        with col1:
            colors = [
                COL_GREEN if v >= 0 else COL_RED
                for v in desc_summary["contribution_like"]
            ]

            fig = go.Figure(go.Bar(
                x=desc_summary["desc_bucket"].astype(str),
                y=desc_summary["contribution_like"],
                marker_color=colors,
                text=[f"{v:+.2f}" for v in desc_summary["contribution_like"]],
                textposition="outside",
                customdata=np.stack([
                    desc_summary["success_rate"] * 100,
                    desc_summary["count"]
                ], axis=-1),
                hovertemplate=(
                    "설명 길이 구간: %{x}<br>"
                    "평균 대비 차이: %{y:+.2f}<br>"
                    "성공률: %{customdata[0]:.1f}%<br>"
                    "영상 수: %{customdata[1]}개<extra></extra>"
                )
            ))

            fig.update_layout(
                **plotly_base(),
                height=320,
                title=f"{chart_domain_title} 설명 길이별 평균 성공률 차이",
                xaxis_title="설명 길이 (자)",
                yaxis_title="평균 대비 성공률 차이"
            )

            st.plotly_chart(fig, use_container_width=True)

            best_row = desc_summary.sort_values("success_rate", ascending=False).iloc[0]
            best_bucket = best_row["desc_bucket"]
            best_rate = best_row["success_rate"] * 100

            st.markdown(f"""
            <div class="insight-box">
            현재 선택한 <b>{chart_domain_title}</b> 기준으로는 
            <b>{best_bucket}자</b> 구간의 성공률이 가장 높게 나타났습니다.
            단, 구간별 영상 수가 적은 경우에는 해석에 주의해야 합니다.
            </div>""", unsafe_allow_html=True)

        with col2:
            st.subheader("핵심 운영 전략 요약")

            best_row = desc_summary.sort_values("success_rate", ascending=False).iloc[0]
            best_bucket = str(best_row["desc_bucket"])
            best_rate = best_row["success_rate"] * 100
            best_count = int(best_row["count"])

            if domain_filter_desc == "FnB":
                desc_tip = "FnB는 제품 경험, 브랜드 분위기, 방문/이벤트 CTA가 설명란에 함께 드러나도록 구성하는 것이 좋습니다."
            elif domain_filter_desc == "IT":
                desc_tip = "IT는 문제 상황, 기능 요약, 자료/데모 CTA처럼 정보 탐색 흐름을 설명란에 명확히 정리하는 것이 좋습니다."
            else:
                desc_tip = "전체 기준은 FnB와 IT 특성이 섞여 있으므로, 실제 전략은 도메인별 결과를 함께 확인하는 것이 좋습니다."

            tips = [
                ("📝", "추천 설명 길이 구간", f"{best_bucket}자 구간"),
                ("📊", "해당 구간 성공률", f"{best_rate:.1f}%"),
                ("🎬", "해당 구간 영상 수", f"{best_count}개"),
                ("🎯", "도메인별 작성 방향", desc_tip),
                ("🔗", "CTA 배치", "핵심 CTA와 관련 링크는 설명란 앞부분에 배치하세요."),
            ]

            for icon, title, desc in tips:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:12px;margin-bottom:8px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                    <div style="font-size:13px;font-weight:600;color:#374151;">{icon} {title}</div>
                    <div style="font-size:12px;color:#6b7280;margin-top:4px;">{desc}</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# [7-7] AI 맞춤 추천 시뮬레이터 
# ═══════════════════════════════════════════════════════════════════════════
#   - X_test의 dtype을 정확히 매칭하여 sample DataFrame 구성
#   - 사용자 입력을 모든 관련 컬럼에 정확히 반영 (upload_time_bucket 자동 계산 등)
#   - 입력값 다양성을 확보하기 위해 X_test 평균/대표값을 base로 사용

def page_simulator():
    section_header(7, "AI 맞춤 추천 시뮬레이터",
                   "영상 제작 조건을 입력하면 AI가 성공 확률을 예측하고, 최적의 전략을 추천해 드립니다.")

    col1, col2, col3 = st.columns([1.2, 1.5, 2])

    # ─────── 좌측: 영상 정보 입력 폼 ───────
    with col1:
        st.subheader("1. 영상 정보 입력")
        domain_sim = st.selectbox("도메인", ["FnB (식음료)", "IT"])
        content_type = st.selectbox("콘텐츠 유형",
            ["브이로그","인터뷰","제품리뷰","웹예능","기술설명","튜토리얼",
             "에피소드소개","이벤트/행사","시설소개","요리/레시피","웹드라마","기타"])
        cta_type = st.selectbox("CTA 유형",
            ["이벤트참여","방문유도","구매유도","구독유도","정보탐색","기타","앱다운로드"])
        length_bucket_kr = st.selectbox("영상 길이",
            ["짧음(~3분)", "표준 길이(3-8분)", "중간 길이(8-15분)", "긴 영상(15분+)"])
        upload_day = st.selectbox("업로드 요일",
            ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"],
             index=4, key="sim_upload_day")
        upload_hour_label = st.selectbox("업로드 시간",
            ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00","06:00", "07:00", "08:00", "09:00", "10:00", "11:00","12:00", "13:00", "14:00", "15:00", "16:00", "17:00","18:00", "19:00", "20:00", "21:00", "22:00", "23:00",],
            index=20,key="sim_upload_hour")
        day_val = upload_day
        hour_val = int(upload_hour_label.split(":")[0])
        desc_len = st.number_input("설명 길이 (예상)", min_value=0, max_value=5000, value=600, step=50)
        caption_use = st.toggle("자막 사용", value=True)
        # thumbnail_type = st.selectbox("썸네일 타입",
        #     ["인물 중심","음식/제품 중심","텍스트 중심","혼합"])

        st.button("✨ 예측하기", type="primary", use_container_width=True)

    # ─────── 입력값을 모델 입력 형태로 변환 ───────
    dom_key = "FnB" if "FnB" in domain_sim else "IT"

    # 한글 라벨 → 모델 학습 시 사용한 영문 코드로 변환
    length_map = {
        "긴 영상(15분+)":   "long_form",
        "중간 길이(8-15분)":  "mid_ads",
        "표준 길이(3-8분)":  "standard",
        "짧음(~3분)":  "short",
    }
    lb = length_map.get(length_bucket_kr, "mid_ads")

    # 영상 길이(초) 추정값 (length_bucket과 일관성 유지)
    length_seconds_map = {
        "long_form": 600,   # 10분
        "mid_ads":   360,   # 6분
        "standard":  180,   # 3분
        "short":     90,    # 1.5분
    }
    video_length_sec = length_seconds_map[lb]

    # 시간대 자동 계산
    if 6 <= hour_val < 11:
        time_bucket = "morning"
    elif 11 <= hour_val < 17:
        time_bucket = "lunch"
    elif 17 <= hour_val < 23:
        time_bucket = "evening"
    else:
        time_bucket = "night"

    # 주말 여부
    is_weekend_val = 1 if day_val in ["토요일", "일요일"] else 0

    # 도메인별 모델 선택
    m_dict = M["m_fnb"] if dom_key == "FnB" else M["m_it"]
    model  = m_dict["model"]
    X_test = m_dict["X_test"]

    
    # X_test의 dtype을 그대로 가져와 빈 DF 생성
    # 수치형은 X_test의 중앙값/평균, 범주형은 최빈값으로 채움 (도메인 평균 영상)
    # 그 위에 사용자 입력값을 덮어쓰기 → 입력값에 따라 결과가 변함

    sample_dict = {}
    for col in X_test.columns:
        col_dtype = X_test[col].dtype
        if pd.api.types.is_numeric_dtype(col_dtype):
            # 수치형 → 중앙값 사용 (이상치 영향 줄임)
            sample_dict[col] = X_test[col].median()
        else:
            # 범주형/문자열 → 최빈값
            mode_val = X_test[col].mode()
            sample_dict[col] = mode_val.iloc[0] if len(mode_val) > 0 else X_test[col].iloc[0]

    def make_caption_value(caption_use, X_ref):
        if "caption" not in X_ref.columns:
            return int(caption_use)
        dtype = X_ref["caption"].dtype
        if dtype == bool:
            return bool(caption_use)
        if pd.api.types.is_numeric_dtype(dtype):
            return int(caption_use)
        
        # 문자열로 학습된 경우
        unique_vals = X_ref["caption"].dropna().astype(str).unique().tolist()
        if "True" in unique_vals or "False" in unique_vals:
            return "True" if caption_use else "False"
        if "자막 있음" in unique_vals or "자막 없음" in unique_vals:
            return "자막 있음" if caption_use else "자막 없음"
        if "1" in unique_vals or "0" in unique_vals:
            return "1" if caption_use else "0"
        return str(caption_use)

    # ── 사용자 입력값으로 덮어쓰기 (X_test에 해당 컬럼 있을 때만) ──
    caption_value = make_caption_value(caption_use, X_test)
    user_inputs = {
        "description_length": desc_len,
        "description_missing_flag": 0 if desc_len > 0 else 1,
        "caption": caption_value,
        "upload_dayofweek": day_val,
        "upload_hour": hour_val,
        "upload_time_bucket": time_bucket,
        "is_weekend": is_weekend_val,
        "length_bucket": lb,
        "영상길이(초)": video_length_sec,
        "cls_content_type": content_type,
        "cls_cta_type": cta_type,
        "upload_year": 2025,
    }
    for col, val in user_inputs.items():
        if col in X_test.columns:
            sample_dict[col] = val

    # DataFrame 생성 (X_test와 컬럼 순서 동일하게)
    sample = pd.DataFrame([sample_dict])[X_test.columns]

    # 각 컬럼 dtype을 X_test와 일치시키기 (★ 핵심: 모델 입력 호환성)
    for col in X_test.columns:
        try:
            sample[col] = sample[col].astype(X_test[col].dtype)
        except (ValueError, TypeError):
            # 변환 불가능한 경우 기본값으로
            pass

    # ── 모델 예측 ──
    try:
        prob_arr = model.predict_proba(sample)
        prob_val = float(prob_arr[0][1])  # class=1 (성공) 확률
    except Exception as e:
        st.warning(f"예측 중 오류: {e}")
        prob_val = 0.5

    # 비교 그룹 평균 (도메인 평균)
    strat = D["strategy"].get(dom_key, {})
    avg_prob = D["eda_stats"][dom_key]["success_rate"]

    # ─────── 중앙: AI 예측 결과 표시 ───────
    with col2:
        st.subheader("2. AI 예측 결과")
        prob_pct = int(prob_val * 100)
        cls = "prob-high" if prob_val >= 0.7 else ("prob-mid" if prob_val >= 0.4 else "prob-low")
        label = "상위 18% 수준" if prob_val >= 0.7 else \
                ("상위 40% 수준" if prob_val >= 0.4 else "하위 40% 수준")

        st.markdown(f"""
        <div style="text-align:center;background:white;border-radius:14px;padding:24px;
        box-shadow:0 2px 10px rgba(0,0,0,0.08);">
            <div style="font-size:14px;color:#6b7280;margin-bottom:8px;">예상 성공 확률</div>
            <div class="prob-value {cls}">{prob_pct}%</div>
            <div style="font-size:13px;margin-top:6px;color:#6b7280;">{label}</div>
        </div>""", unsafe_allow_html=True)

        # 비교 게이지
        st.markdown("**비교 그룹 평균**")
        st.progress(avg_prob, text=f"{avg_prob*100:.0f}%")
        st.markdown("**선택 조건**")
        st.progress(min(prob_val, 1.0), text=f"{prob_pct}%")

        diff = prob_val - avg_prob
        icon = "✅" if diff > 0 else "⚠️"
        diff_text = f"+{diff*100:.1f}%p 높음" if diff > 0 else f"{diff*100:.1f}%p 낮음"
        st.markdown(f"""
        <div class="insight-box {'insight-green' if diff>0 else 'insight-orange'}">
        {icon} 평균 대비 {diff_text} ({dom_key} 평균 {avg_prob*100:.0f}% 대비)
        </div>""", unsafe_allow_html=True)

    # ─────── 우측: 핵심 성공 요인 (성공 기여도 TOP 5) ───────
    with col3:
        subheader_with_tip("3. 핵심 성공 요인 TOP 5", TIP_CONTRIBUTION)
        try:
            import shap as shap_lib
            shap_m = M["shap_fnb"] if dom_key == "FnB" else M["shap_it"]
            explainer = shap_lib.TreeExplainer(shap_m["model"])

            # SHAP용 sample도 동일한 방식으로 생성
            shap_X = shap_m["X_test"]
            shap_sample_dict = {}
            for col in shap_X.columns:
                col_dtype = shap_X[col].dtype
                if pd.api.types.is_numeric_dtype(col_dtype):
                    shap_sample_dict[col] = shap_X[col].median()
                else:
                    mv = shap_X[col].mode()
                    shap_sample_dict[col] = mv.iloc[0] if len(mv) > 0 else shap_X[col].iloc[0]
            for col, val in user_inputs.items():
                if col in shap_X.columns:
                    shap_sample_dict[col] = val
            shap_sample = pd.DataFrame([shap_sample_dict])[shap_X.columns]
            for col in shap_X.columns:
                try:
                    shap_sample[col] = shap_sample[col].astype(shap_X[col].dtype)
                except (ValueError, TypeError):
                    pass

            shap_vals = explainer(shap_sample)
            sv = shap_vals.values[0]
            feats = shap_X.columns.tolist()

            # 영향 큰 순 TOP 5
            pairs = sorted(zip(feats, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
            top_items = [(k(f), v) for f, v in pairs]  # 한글 라벨

            for feat, val in top_items:
                dir_icon = "↑" if val > 0 else "↓"
                dir_color = COL_GREEN if val > 0 else COL_RED
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                padding:10px;background:white;border-radius:8px;margin-bottom:6px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                    <span style="font-weight:600;font-size:13px;flex:2;">{feat}</span>
                    <span style="color:{dir_color};font-size:20px;flex:0.3;">{dir_icon}</span>
                    <span style="color:{dir_color};font-weight:700;flex:1;text-align:right;">{val:+.2f}</span>
                </div>""", unsafe_allow_html=True)

            # caption 영향 별도 표시
            caption_shap_value = None

            for feat, val in zip(feats, sv):
                if feat == "caption":
                    caption_shap_value = val
                    break

            if caption_shap_value is not None:
                cap_icon = "↑" if caption_shap_value > 0 else "↓"
                cap_color = COL_GREEN if caption_shap_value > 0 else COL_RED
                cap_text = "성공 확률을 높이는 방향" if caption_shap_value > 0 else "성공 확률을 낮추는 방향"

                st.markdown(f"""
                <div style="background:#f0fdf4;border-left:4px solid {cap_color};border-radius:8px;
                padding:10px;margin-top:10px;font-size:13px;">
                    <b>자막 사용 영향</b><br>
                    현재 조건에서 자막 사용 여부는 
                    <span style="color:{cap_color};font-weight:700;">{cap_icon} {caption_shap_value:+.3f}</span>
                    로, {cap_text}으로 작용했습니다.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("현재 SHAP 입력 데이터에 caption 컬럼이 없어 자막 영향도를 표시할 수 없습니다.")


        except Exception as e:
            # SHAP 실패 시 전략 텍스트로 대체
            rec = strat.get("recommended", [])
            for r in rec[:5]:
                st.markdown(
                    f'<div style="padding:8px;background:white;border-radius:8px;margin-bottom:6px;font-size:13px;">✅ {r}</div>',
                    unsafe_allow_html=True
                )

    # ─────── 하단: 전략 추천 카드 ───────
    st.markdown("---")
    st.subheader("4. 전략 추천")
    tips = [
        ("📝 설명 최적화", "설명 길이를 500~800자로 작성하고, 메뉴/위치/가격 등 구체 정보를 포함하세요.", "우선순위: 높음"),
        ("📢 CTA 강화", f"고정 댓글 + 설명 상단에 {cta_type} CTA를 명확하게 배치하세요.", "우선순위: 높음"),
        ("⏰ 업로드 시간 최적화", f"{day_val} {hour_val}시 업로드를 유지하면 더 높은 반응을 기대할 수 있습니다.", "우선순위: 중간"),
        # ("🖼️ 썸네일 개선", f"{thumbnail_type} 중심의 썸네일이 클릭율에 영향을 줍니다.", "우선순위: 중간"),
        ("#️⃣ 자막 & 해시태그", "자막을 사용하고, 자막 + 메뉴 키워드 해시태그를 3~5개 활용하세요.", "우선순위: 낮음"),
        ("🎬 영상 길이 유지", f"{length_bucket_kr} 길이를 유지하면 시청 지속시간 확보에 유리합니다.", "우선순위: 높음"),
    ]
    cols = st.columns(3)
    for i, (title, desc, prio) in enumerate(tips):
        prio_color = COL_GREEN if "높음" in prio else (COL_BLUE if "중간" in prio else COL_GRAY)
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:14px;margin-bottom:10px;
            box-shadow:0 1px 6px rgba(0,0,0,0.07);">
                <div style="font-weight:700;font-size:13px;color:#374151;margin-bottom:6px;">{title}</div>
                <div style="font-size:12px;color:#6b7280;line-height:1.6;">{desc}</div>
                <div style="margin-top:8px;"><span style="font-size:11px;color:{prio_color};
                font-weight:600;">{prio}</span></div>
            </div>""", unsafe_allow_html=True)

    # ─────── 예상 성과 시뮬레이션 ───────
    st.subheader("5. 예상 성과 시뮬레이션")
    stat = D["eda_stats"][dom_key]
    base_views = int(stat["조회수"]["median"])
    pred_views = int(base_views * (1 + prob_val))
    pred_likes = int(stat["좋아요수"]["median"] * (1 + prob_val))
    pred_comments = int(stat["댓글수"]["median"] * (1 + prob_val))

    col_a, col_b, col_c, col_d = st.columns(4)
    sim_data = [
        (col_a, "성공 확률", f"{avg_prob*100:.0f}%", f"{prob_pct}%"),
        (col_b, "조회수 (예상)", f"{base_views:,}", f"{pred_views:,}"),
        (col_c, "좋아요 (예상)", f"{int(stat['좋아요수']['median']):,}", f"{pred_likes:,}"),
        (col_d, "댓글 (예상)", f"{int(stat['댓글수']['median']):,}", f"{pred_comments:,}"),
    ]
    for col, label, base_v, pred_v in sim_data:
        with col:
            # 문자열 → 숫자 파싱
            base_num = float(base_v.replace("%","").replace(",","")) if "%" in base_v \
                       else int(base_v.replace(",",""))
            pred_num = float(pred_v.replace("%","").replace(",","")) if "%" in pred_v \
                       else int(pred_v.replace(",",""))
            fig = go.Figure(go.Bar(
                x=["현재 조건", "전략 적용 시"],
                y=[base_num, pred_num],
                marker_color=[COL_GRAY, COL_PURPLE],
                text=[base_v, pred_v],
                textposition="outside"
            ))
            fig.update_layout(**plotly_base(), height=200, title=label,
                              yaxis_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # AI 종합 코멘트
    improvement = prob_pct - int(avg_prob * 100)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:12px;
    padding:18px;color:white;margin-top:10px;">
        <div style="font-size:16px;font-weight:700;margin-bottom:8px;">🤖 AI 종합 코멘트</div>
        <div style="font-size:14px;line-height:1.7;opacity:0.95;">
            입력하신 조건은 {dom_key} 도메인에서 {'매우 유리한' if prob_val>=0.7 else '보통의'} 조합입니다.<br>
            특히 '{cta_type}' CTA와 '{content_type}' 콘텐츠가 성공에 큰 영향을 줄 것으로 보입니다.<br>
            위 전략을 모두 적용하면 성공 확률이 약 {max(improvement,5)}%p 상승할 것으로 예상됩니다.
        </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# [7-8] 운영 전략 로드맵
# ═══════════════════════════════════════════════════════════════════════════
# 데이터 기반 인사이트를 운영 단계(기획→제작→업로드→배포→분석) 로드맵으로 구성
def page_roadmap():
    section_header(8, "운영 전략 로드맵",
                   "데이터 기반 인사이트를 바탕으로 도메인별 맞춤 운영 전략 로드맵을 제시합니다.")

    # 도메인 선택
    col_sel, _ = st.columns([2, 4])
    with col_sel:
        roadmap_domain = st.selectbox("도메인 선택", ["FnB (식음료)", "IT"], key="rm_dom")
    dom_key = "FnB" if "FnB" in roadmap_domain else "IT"
    strat = D["strategy"][dom_key]

    # ──── 5단계 로드맵 타임라인 ────
    st.subheader(f"채널 운영 전략 로드맵 ({dom_key} 도메인 예시)")
    phases = [
        ("기획", "기간: 1~2주", ["트렌드 리서치","키워드 조사","타깃 페르소나 설정"]),
        ("제작", "기간: 3~4주", ["인기/How-to/리뷰 콘텐츠 중심","핵심 메시지 구성","자막 및 썸네일 제작"]),
        ("업로드", "기간: 지속적", ["야간 시간대 업로드 최적화","요일/수요일 우선","이벤트/할인 CTA 삽입"]),
        ("배포", "기간: 지속적", ["커뮤니티/인스타 공유","SNS 채널 연동","광고/프로모션 진행"]),
        ("분석/개선", "기간: 지속적", ["성과 데이터 분석","성공 기여도 피드백","콘텐츠 개선"]),
    ]
    cols = st.columns(5)
    for i, (phase, period, items) in enumerate(phases):
        with cols[i]:
            items_html = "".join([
                f'<div style="font-size:11px;text-align:left;padding:2px 0;color:#374151;">• {item}</div>'
                for item in items
            ])
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:14px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;min-height:180px;">
                <div style="width:36px;height:36px;background:{COL_PURPLE};border-radius:50%;
                display:flex;align-items:center;justify-content:center;
                font-weight:700;color:white;font-size:14px;margin:0 auto 8px;">{i+1}</div>
                <div style="font-weight:700;font-size:14px;color:#1e1e3f;">{phase}</div>
                <div style="font-size:11px;color:#6b7280;margin-bottom:8px;">{period}</div>
                {items_html}
            </div>""", unsafe_allow_html=True)

    # ──── 체크리스트 / 다음 액션 ────
    st.markdown("---")
    col1, col2 = st.columns([2, 1.5])

    # 도메인별 전략 체크리스트
    with col1:
        st.subheader(f"도메인별 전략 체크리스트 ({dom_key})")
        checklist = [
            ("콘텐츠 주제 최적화", "How-to, 리뷰, 레시피 영상 비중 확대", "완료", COL_GREEN),
            ("업로드 스케줄 최적화", "야간(18시~24시) / 일요일-수요일 집중 업로드", "진행중", COL_BLUE),
            ("썸네일 & 제목 최적화", "클릭 유도형 썸네일 + 키워드 포함 제목", "완료", COL_GREEN),
            ("CTA & 전환 유도", "구독 / 좋아요 / 댓글 유도 문구 삽입", "진행중", COL_BLUE),
            ("커뮤니티 & SNS 연동", "인스타 / 블로그 / 카카오채널 연동 강화", "대기", COL_GRAY),
            ("성과 분석 & 개선", "성공 기여도 분석 기반 콘텐츠 개선 및 A/B 테스트", "대기", COL_GRAY),
        ]
        for title, desc, status, color in checklist:
            status_icon = {"완료":"✅","진행중":"🔄","대기":"⏳"}.get(status,"⏳")
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;padding:10px;
            background:white;border-radius:8px;margin-bottom:6px;
            box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <span style="font-size:16px;">{status_icon}</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#374151;">{title}</div>
                    <div style="font-size:11px;color:#6b7280;margin-top:2px;">{desc}</div>
                </div>
                <span style="margin-left:auto;font-size:11px;color:{color};font-weight:600;
                white-space:nowrap;">{status}</span>
            </div>""", unsafe_allow_html=True)

    # 다음 단계 추천 액션
    with col2:
        st.subheader("다음 단계 추천 액션")
        next_actions = strat.get("recommended", [])[:6]
        for action in next_actions:
            st.markdown(f"""
            <div style="background:white;border-radius:8px;padding:10px;margin-bottom:6px;
            box-shadow:0 1px 4px rgba(0,0,0,0.05);font-size:12px;display:flex;gap:8px;">
                <span>📌</span><span style="color:#374151;">{action}</span>
            </div>""", unsafe_allow_html=True)


# [8] 라우터는 app.py에서 처리합니다.
