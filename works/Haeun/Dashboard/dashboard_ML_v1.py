# ═══════════════════════════════════════════════════════════════════════════
# dashboard_ML.py
# ─ 영상 성과 예측 + 댓글 긍정 비율 예측 시뮬레이터
# ═══════════════════════════════════════════════════════════════════════════
#
# ── 연동 가이드 ──────────────────────────────────────────────────────────
#
# [1] 전달받은 파일 배치
#     아래 파일들이 dashboard_ML.py와 같은 폴더에 있으면 됩니다.
#     실행 위치와 무관하게 동작합니다. (Path(__file__).parent 기준)
#
#     어떤폴더/
#     ├── .streamlit/
#     │   └── secrets.toml                          ← Vertex AI 인증용 (이건 제작하셔야 합니다)
#     │                                               .gitignore에 추가: .streamlit/secrets.toml
#     ├── dashboard_ML.py                           ← 이 파일
#     ├── dashboard_thumbnail.py                    ← 썸네일 분석 페이지
#     ├── dashboard_shortform.py                    ← 숏폼 분석 페이지
#     ├── longform_comment_model.joblib
#     ├── longform_comment_encoding_map.joblib
#     ├── longform_FnB_best_perf_CatBoost.joblib
#     ├── longform_IT_best_perf_RandomForest.joblib
#     ├── longform_FnB_shap_catboost.joblib         ← FnB SHAP 전용
#     ├── longform_IT_shap_catboost.joblib          ← IT SHAP 전용
#     └── test_data(longform_comment).csv
#
# [2] secrets.toml 작성 형식 (.streamlit/secrets.toml)
#     GOOGLE_CLOUD_PROJECT = "your-project-id"
#     GOOGLE_CLOUD_REGION  = "global"
#     GEMINI_MODEL         = "gemini-3.1-flash-lite-preview"
#     GEMINI_MODEL_VISION  = "gemini-2.5-flash"   ← 썸네일 분석(이미지)용
#
# [3] 메인 앱에서 호출 방법
#     from dashboard_ML import page_simulator as page_ml
#     page_ml()
#
# [4] 주의사항
#     st.set_page_config()는 앱 전체에서 한 번만 호출 가능합니다.
#     메인 앱(app.py)에서 이미 설정한 경우, 이 파일 하단의
#     if __name__ == "__main__" 블록만 남기고
#     set_page_config() 호출 블록을 제거하거나 주석 처리하세요.
#
# [5] 추가 설치 패키지
#     uv add pydantic-ai[vertexai] joblib plotly shap catboost
#
# ─────────────────────────────────────────────────────────────────────────

# ── 라이브러리 ────────────────────────────────────────────────────────────
import asyncio
import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# ── 페이지 설정 ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube 영상 성과 예측",
    page_icon="📊",
    layout="wide",
)

# ── Vertex AI 연결 (.streamlit/secrets.toml) ─────────────────────────────
GOOGLE_CLOUD_PROJECT = st.secrets["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_REGION  = st.secrets["GOOGLE_CLOUD_REGION"]
GEMINI_MODEL         = st.secrets.get("GEMINI_PROMPT_MODEL", "gemini-3.1-flash-lite-preview")

provider = GoogleProvider(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_REGION,
)
model_id = GoogleModel(GEMINI_MODEL, provider=provider)

# ── AI 코멘트 에이전트 ────────────────────────────────────────────────────
comment_agent = Agent(
    model_id,
    system_prompt=(
        "당신은 YouTube 마케팅 전문가입니다. "
        "두 AI 모델의 예측 결과를 바탕으로 실무자가 바로 활용할 수 있는 "
        "간결하고 구체적인 업로드 전략 코멘트를 한국어로 작성합니다. "
        "3~4문장으로 핵심만 전달하세요."
    ),
)

def get_ai_comment(prompt: str) -> str:
    """PydanticAI 에이전트를 Streamlit(동기) 환경에서 호출"""
    result = asyncio.run(comment_agent.run(prompt))
    return result.output


# ── 모델 및 데이터 로드 ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent  # dashboard_ML.py 위치 기준 → 실행 위치 무관

@st.cache_resource
def load_all():
    # ── 성과 예측 모델 (도메인별) ─────────────────────────
    fnb_obj = joblib.load(BASE_DIR / "longform_FnB_best_perf_CatBoost.joblib")
    it_obj  = joblib.load(BASE_DIR / "longform_IT_best_perf_RandomForest.joblib")

    perf_models = {
        "FnB": fnb_obj["model"],
        "IT":  it_obj["model"],
    }

    # X_test: DataFrame이면 그대로, numpy array면 피쳐명 추출
    def make_X_test_df(obj, model):
        raw = obj["X_test"]
        if isinstance(raw, pd.DataFrame):
            return raw
        if "feature_cols" in obj:
            cols = obj["feature_cols"]
        elif hasattr(model, "feature_names_"):
            cols = list(model.feature_names_)
        elif hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            cols = None
        return pd.DataFrame(raw, columns=cols)

    perf_X_test = {
        "FnB": make_X_test_df(fnb_obj, fnb_obj["model"]),
        "IT":  make_X_test_df(it_obj,  it_obj["model"]),
    }

    # 성과 모델 도메인 평균 성공률 (test 데이터 기준)
    # [test 데이터를 쓰는 이유]
    # - 학습 후 단순 통계(평균)를 계산하는 것이므로 데이터 누수와 무관
    # - train 기준이면 모델이 이미 학습한 데이터라 평균이 과대 추정될 수 있음
    # - test 데이터는 모델이 본 적 없는 데이터 → 실제 현장에 가까운 공정한 기준값
    perf_avg = {
        "FnB": float(np.array(fnb_obj["y_test"]).mean()),
        "IT":  float(np.array(it_obj["y_test"]).mean()),
    }

    # ── SHAP 전용 모델 (도메인별 별도 파일) ──────────────
    shap_models = {
        "FnB": joblib.load(BASE_DIR / "longform_FnB_shap_catboost.joblib"),
        "IT":  joblib.load(BASE_DIR / "longform_IT_shap_catboost.joblib"),
    }

    # ── 댓글 긍정 비율 모델 ───────────────────────────────
    longform_model   = joblib.load(BASE_DIR / "longform_comment_model.joblib")
    longform_enc_map = joblib.load(BASE_DIR / "longform_comment_encoding_map.joblib")

    longform_test_full = pd.read_csv(BASE_DIR / "test_data(longform_comment).csv")
    longform_X_test    = longform_test_full.drop(columns=["target_feature"])

    # 댓글 모델 도메인별 긍정 비율 평균 (test 데이터 기준, 위와 동일한 이유)
    domain_num_to_str = longform_enc_map["domain"]
    sent_avg = (
        longform_test_full
        .groupby("domain")["target_feature"]
        .mean()
        .rename(index=domain_num_to_str)
        .to_dict()
    )

    return (
        perf_models, perf_X_test, perf_avg,
        shap_models,
        longform_model, longform_X_test, longform_enc_map, sent_avg
    )

(
    perf_models, perf_X_test, PERF_AVG,
    shap_models,
    longform_model, longform_X_test, longform_enc_map, SENT_AVG
) = load_all()

# ── 색상 상수 ─────────────────────────────────────────────────────────────
COL_GREEN  = "#22c55e"
COL_RED    = "#ef4444"
COL_BLUE   = "#3b82f6"
COL_GRAY   = "#9ca3af"
COL_PURPLE = "#8b5cf6"

# ── 피쳐 한글 라벨 매핑 ──────────────────────────────────────────────────
LABEL_MAP = {
    "description_length":        "영상 설명 길이",
    "description_missing_flag":  "영상 설명 누락 여부",
    "tags_count":                "영상 태그 수",
    "tags_missing_flag":         "영상 태그 누락 여부",
    "영상길이(초)":               "영상 길이(초)",
    "caption":                   "자막 사용 여부",
    "definition":                "화질",
    "embeddable":                "외부 임베드 허용",
    "has_paid_product_placement": "유료 광고 포함 여부",
    "upload_year":               "업로드 연도",
    "upload_month":              "업로드 월",
    "upload_quarter":            "업로드 분기",
    "upload_dayofweek":          "업로드 요일",
    "upload_hour":               "업로드 시간",
    "upload_time_bucket":        "업로드 시간대",
    "is_weekend":                "주말 여부",
    "cls_content_type":          "콘텐츠 유형",
    "cls_marketing_purpose":     "마케팅 목적",
    "cls_cta_type":              "고객행동 유도(CTA) 유형",
    "cls_is_series":             "시리즈 여부",
    "cls_is_collaboration":      "콜라보 여부",
    "length_bucket":             "영상 길이",
    "domain":                    "도메인",
    "channel_tier":              "채널 규모",
}

def k(col):
    return LABEL_MAP.get(str(col), str(col))

# ── 커스텀 CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .prob-high { color: #22c55e; }
    .prob-mid  { color: #f59e0b; }
    .prob-low  { color: #ef4444; }
    .insight-box {
        border-radius: 8px; padding: 10px 14px; font-size: 0.88rem; margin-top: 8px;
    }
    .insight-green  { background: #f0fdf4; border-left: 4px solid #22c55e; color: #15803d; }
    .insight-orange { background: #fffbeb; border-left: 4px solid #f59e0b; color: #92400e; }
    .insight-red    { background: #fef2f2; border-left: 4px solid #ef4444; color: #991b1b; }
    .sec-label {
        font-size: 0.75rem; font-weight: 700; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.06em; margin: 1rem 0 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 시뮬레이터 메인
# ═══════════════════════════════════════════════════════════════════════════
def page_simulator():
    st.title("📊 YouTube 영상 AI 분석 대시보드")
    st.caption("제작한 영상의 조건을 입력하면 분석 결과를 기반으로 성공 확률과 댓글 긍정 반응을 예측하고\n영상 제작 및 업로드 전략을 추천합니다.")
    st.divider()

    # ═══════════════════════════════════════════════════════
    # ① 공통 입력 피쳐 (두 모델이 모두 사용하는 피쳐)
    # ═══════════════════════════════════════════════════════
    st.markdown('<div class="sec-label">공통 입력 피쳐</div>', unsafe_allow_html=True)

    c_dom, _ = st.columns([1, 3])
    with c_dom:
        domain_sim = st.selectbox(
            "업종 (도메인)",
            ["FnB (식음료)", "IT"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        content_type = st.selectbox(
            "영상 포맷 (콘텐츠 유형)",
            ["브이로그", "인터뷰", "제품리뷰", "웹예능", "기술설명", "튜토리얼",
             "에피소드소개", "이벤트/행사", "시설소개", "요리/레시피", "웹드라마",
             "광고/CF", "다큐멘터리", "애니메이션", "영양정보", "고객후기", "기타"],
            help="영상이 어떤 형식·포맷으로 만들어졌는지 선택하세요.")
    with c2:
        marketing_purpose = st.selectbox(
            "영상 제작 목적 (마케팅 목적)",
            ["브랜드캠페인", "제품홍보", "고객유입", "고객유지", "기업이미지",
             "채용", "사회공헌/환경", "서비스활용", "정보제공", "기타"],
            help="이 영상을 올리는 목적을 선택하세요.")
    with c3:
        cta_type = st.selectbox(
            "시청자 행동 유도 방식 (CTA)",
            ["구매유도", "구독유도", "이벤트참여", "정보탐색", "앱다운로드", "방문유도", "기타"],
            help="영상에서 시청자에게 유도하는 다음 행동을 선택하세요.")
    with c4:
        length_bucket_kr = st.selectbox(
            "영상 길이",
            ["짧음(~3분)", "표준 길이(3-8분)", "중간 길이(8-15분)", "긴 영상(15분+)"],
            help="업로드할 영상의 총 재생 시간 구간을 선택하세요.")

    c5, c6, c7 = st.columns(3)
    with c5:
        _current_month = datetime.datetime.now().month
        upload_month_val = st.selectbox(
            "업로드 예정 월", list(range(1, 13)),
            format_func=lambda x: f"{x}월", index=_current_month - 1, key="sim_upload_month",
            help="영상을 업로드할 예정 월을 선택하세요. 기본값은 현재 기준입니다.")
    with c6:
        upload_day = st.selectbox(
            "업로드 예정 요일",
            ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"],
            index=4, key="sim_upload_day",
            help="업로드할 예정 요일을 선택하세요.")
    with c7:
        upload_hour_label = st.selectbox(
            "업로드 예정 시각",
            ["00:00","01:00","02:00","03:00","04:00","05:00",
             "06:00","07:00","08:00","09:00","10:00","11:00",
             "12:00","13:00","14:00","15:00","16:00","17:00",
             "18:00","19:00","20:00","21:00","22:00","23:00"],
            index=20, key="sim_upload_hour",
            help="업로드할 예정 시각을 선택하세요. 시청자 활동 시간대에 따라 초기 노출이 달라집니다.")

    c8, c9 = st.columns(2)
    with c8:
        desc_len = st.number_input(
            "영상 설명 길이 (글자 수)", min_value=0, max_value=5000, value=600, step=50,
            help="유튜브 영상 설명란에 작성할 예상 글자 수입니다. 500자 이상 권장합니다.")
    with c9:
        tags_count_val = st.number_input(
            "태그 수", min_value=0, max_value=50, value=10, step=1,
            help="영상에 추가할 태그 개수입니다. 15~30개를 권장합니다.")

    c10, c11, c12 = st.columns(3)
    with c10:
        caption_use = st.toggle("자막 포함", value=True,
            help="영상에 자막(CC)이 들어가면 활성화하세요. 검색 노출과 접근성에 영향을 줍니다.")
    with c11:
        cls_is_series = int(st.toggle("시리즈 영상", value=False,
            help="동일 주제를 연속으로 다루는 시리즈 콘텐츠면 활성화하세요."))
    with c12:
        cls_is_collaboration = int(st.toggle("콜라보 영상", value=False,
            help="다른 크리에이터나 브랜드와 함께 만든 영상이면 활성화하세요."))

    # ═══════════════════════════════════════════════════════
    # ② 고급 옵션 (모델별로 달랐던 피쳐)
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-label">⚙️ 고급 옵션</div>', unsafe_allow_html=True)

    adv1, adv2 = st.columns(2)

    with adv1:
        st.markdown("**📈 성과 모델 전용**")
        # upload_time_bucket: 업로드 시각에서 자동 계산 (UI 미표시)
        # length_bucket: 영상 길이 선택에서 자동 계산 (UI 미표시)
        # description_missing_flag: 설명 길이 0이면 자동으로 1 (UI 미표시)
        # tags_missing_flag: 태그 수 0이면 자동으로 1 (UI 미표시)
        definition = st.selectbox(
            "영상 화질",
            ["hd", "sd"],
            format_func=lambda x: "HD (720p 이상)" if x == "hd" else "SD (720p 미만)",
            help="HD는 720p 이상, SD는 720p 미만입니다. 모델에는 hd/sd 코드로 입력됩니다.")
        upload_year = st.number_input(
            "업로드 연도", min_value=2010, max_value=2030, value=2025,
            help="영상 업로드 예정 연도입니다.")
        embeddable = int(st.toggle("외부 사이트 삽입 허용", value=True,
            help="블로그, 뉴스 등 외부 사이트에서 이 영상을 삽입 재생할 수 있도록 허용합니다."))

    with adv2:
        st.markdown("**💬 댓글 모델 전용**")
        # 채널 규모: 학습 데이터 실제 값 (encoding_map["channel_tier"])
        # {0: 'macro', 1: 'mega', 2: 'micro', 3: 'mid', 4: 'nano', 5: 'pico'}
        CHANNEL_TIER_LABELS = {
            "pico":  "구독자 1천 명 미만",
            "nano":  "구독자 1천~1만 명",
            "micro": "구독자 1만~5만 명",
            "mid":   "구독자 5만~10만 명",
            "macro": "구독자 10만~50만 명",
            "mega":  "구독자 50만 초과",
        }
        channel_tier_val = st.selectbox(
            "채널 구독자 규모",
            list(CHANNEL_TIER_LABELS.keys()),
            format_func=lambda x: CHANNEL_TIER_LABELS[x],
            help="브랜드 유튜브 채널의 구독자 규모를 선택하세요.")
        has_paid_val = int(st.toggle("협찬 / PPL 포함", value=False,
            help="영상 내 유료 광고(PPL, 협찬)가 포함된 경우 활성화하세요."))

    # ═══════════════════════════════════════════════════════
    # 예측 버튼
    # ═══════════════════════════════════════════════════════
    st.divider()
    predict_clicked = st.button("🔮 두 모델 동시 예측", use_container_width=True, type="primary")

    if not predict_clicked:
        return

    # ═══════════════════════════════════════════════════════
    # 입력값 → 모델 입력 형태 변환
    # ═══════════════════════════════════════════════════════
    dom_key  = "FnB" if "FnB" in domain_sim else "IT"
    hour_val = int(upload_hour_label.split(":")[0])
    day_val  = upload_day

    length_map = {
        "긴 영상(15분+)":    "long_form",
        "중간 길이(8-15분)": "mid_ads",
        "표준 길이(3-8분)":  "standard",
        "짧음(~3분)":        "short",
    }
    lb = length_map.get(length_bucket_kr, "mid_ads")

    length_seconds_map = {"long_form": 600, "mid_ads": 360, "standard": 180, "short": 90}
    video_length_sec   = length_seconds_map[lb]

    upload_quarter_val   = (upload_month_val - 1) // 3 + 1
    is_weekend_val       = 1 if day_val in ["토요일", "일요일"] else 0
    day_label_to_int     = {
        "월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3,
        "금요일": 4, "토요일": 5, "일요일": 6,
    }
    upload_dayofweek_val = day_label_to_int[day_val]

    # upload_time_bucket 자동 계산 (업로드 시각에서 4구간으로 분류)
    if 6 <= hour_val < 11:
        time_bucket = "morning"
    elif 11 <= hour_val < 17:
        time_bucket = "lunch"
    elif 17 <= hour_val < 23:
        time_bucket = "evening"
    else:
        time_bucket = "night"

    def make_caption_value(caption_use, X_ref):
        if "caption" not in X_ref.columns:
            return int(caption_use)
        dtype = X_ref["caption"].dtype
        if dtype == bool:
            return bool(caption_use)
        if pd.api.types.is_numeric_dtype(dtype):
            return int(caption_use)
        unique_vals = X_ref["caption"].dropna().astype(str).unique().tolist()
        if "True" in unique_vals or "False" in unique_vals:
            return "True" if caption_use else "False"
        if "자막 있음" in unique_vals or "자막 없음" in unique_vals:
            return "자막 있음" if caption_use else "자막 없음"
        if "1" in unique_vals or "0" in unique_vals:
            return "1" if caption_use else "0"
        return str(caption_use)

    # ── 성과 모델 입력 ────────────────────────────────────
    user_inputs_perf = {
        "description_length":        desc_len,
        "description_missing_flag":  1 if desc_len == 0 else 0,   # 글자 수 0이면 자동으로 1
        "tags_count":                tags_count_val,
        "tags_missing_flag":         1 if tags_count_val == 0 else 0,  # 태그 수 0이면 자동으로 1
        "caption":                   None,
        "upload_dayofweek":          upload_dayofweek_val,
        "upload_hour":               hour_val,
        "upload_month":              upload_month_val,
        "upload_quarter":            upload_quarter_val,
        "upload_time_bucket":        time_bucket,   # 업로드 시간에서 자동 계산
        "is_weekend":                is_weekend_val,
        "length_bucket":             lb,             # 영상 길이 선택에서 자동 계산
        "영상길이(초)":               video_length_sec,
        "cls_content_type":          content_type,
        "cls_marketing_purpose":     marketing_purpose,
        "cls_cta_type":              cta_type,
        "cls_is_series":             cls_is_series,
        "cls_is_collaboration":      cls_is_collaboration,
        "has_paid_product_placement": has_paid_val,
        "definition":                definition,
        "embeddable":                embeddable,
        "upload_year":               upload_year,
    }

    # ── 댓글 모델 입력 (LabelEncoding 역변환) ────────────
    # encoding_map: {col: {숫자: "문자열"}} → 역변환: {"문자열": 숫자}
    reverse_map = {
        col: {v: k for k, v in mapping.items()}
        for col, mapping in longform_enc_map.items()
    }

    def encode(col, raw_val):
        return reverse_map[col].get(raw_val, 0)

    user_inputs_sent = {
        "domain":                    encode("domain", dom_key),
        "영상길이(초)":               video_length_sec,
        "has_paid_product_placement": has_paid_val,
        # channel_tier: 사용자가 선택한 영문 코드값(macro/mega 등)을 그대로 encode
        "channel_tier":              encode("channel_tier", channel_tier_val),
        "upload_month":              upload_month_val,
        "upload_dayofweek":          encode("upload_dayofweek", day_val),
        "upload_hour":               hour_val,
        "upload_quarter":            upload_quarter_val,
        "description_length":        desc_len,
        "tags_count":                tags_count_val,
        "cls_content_type":          encode("cls_content_type", content_type),
        "cls_marketing_purpose":     encode("cls_marketing_purpose", marketing_purpose),
        "cls_cta_type":              encode("cls_cta_type", cta_type),
        "cls_is_series":             cls_is_series,
        "cls_is_collaboration":      cls_is_collaboration,
    }

    # ═══════════════════════════════════════════════════════
    # 성과 모델 예측
    # ═══════════════════════════════════════════════════════
    model  = perf_models[dom_key]
    X_test = perf_X_test[dom_key]

    sample_dict = {}
    for col in X_test.columns:
        if pd.api.types.is_numeric_dtype(X_test[col].dtype):
            sample_dict[col] = X_test[col].median()
        else:
            mv = X_test[col].mode()
            sample_dict[col] = mv.iloc[0] if len(mv) > 0 else X_test[col].iloc[0]

    user_inputs_perf["caption"] = make_caption_value(caption_use, X_test)
    for col, val in user_inputs_perf.items():
        if col in X_test.columns:
            sample_dict[col] = val

    sample = pd.DataFrame([sample_dict])[X_test.columns]
    for col in X_test.columns:
        try:
            sample[col] = sample[col].astype(X_test[col].dtype)
        except (ValueError, TypeError):
            pass

    try:
        prob_arr = model.predict_proba(sample)
        prob_val = float(prob_arr[0][1])
    except Exception as e:
        st.warning(f"성과 모델 예측 오류: {e}")
        prob_val = 0.5

    # ═══════════════════════════════════════════════════════
    # 댓글 긍정 비율 모델 예측
    # ═══════════════════════════════════════════════════════
    sent_sample_dict = {}
    for col in longform_X_test.columns:
        if pd.api.types.is_numeric_dtype(longform_X_test[col].dtype):
            sent_sample_dict[col] = longform_X_test[col].median()
        else:
            mv = longform_X_test[col].mode()
            sent_sample_dict[col] = mv.iloc[0] if len(mv) > 0 else longform_X_test[col].iloc[0]

    for col, val in user_inputs_sent.items():
        if col in longform_X_test.columns:
            sent_sample_dict[col] = val

    sent_sample = pd.DataFrame([sent_sample_dict])[longform_X_test.columns]
    for col in longform_X_test.columns:
        try:
            sent_sample[col] = sent_sample[col].astype(longform_X_test[col].dtype)
        except (ValueError, TypeError):
            pass

    try:
        sent_val = float(longform_model.predict(sent_sample)[0])
    except Exception as e:
        st.warning(f"댓글 모델 예측 오류: {e}")
        sent_val = 0.5

    # ── 결과값 정리 ───────────────────────────────────────
    prob_pct     = int(prob_val * 100)
    sent_pct     = sent_val * 100
    avg_prob     = PERF_AVG[dom_key]
    avg_sent_pct = SENT_AVG.get(dom_key, 0.55) * 100
    diff_perf    = prob_val - avg_prob
    sent_diff    = sent_pct - avg_sent_pct

    # ═══════════════════════════════════════════════════════
    # ③ 예측 결과
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-label">📋 모델 예측 결과</div>', unsafe_allow_html=True)

    res_c1, res_c2, res_c3 = st.columns([1, 1, 1.5])

    # ── 성과 예측 카드 ────────────────────────────────────
    with res_c1:
        cls        = "prob-high" if prob_val >= 0.7 else ("prob-mid" if prob_val >= 0.4 else "prob-low")
        perf_label = "상위 18% 수준" if prob_val >= 0.7 else \
                     ("상위 40% 수준" if prob_val >= 0.4 else "하위 40% 수준")
        perf_icon     = "✅" if diff_perf > 0 else "⚠️"
        diff_perf_str = f"+{diff_perf*100:.1f}%p" if diff_perf > 0 else f"{diff_perf*100:.1f}%p"

        st.markdown(f"""
        <div style="text-align:center;background:white;border-radius:14px;padding:20px;
        box-shadow:0 2px 10px rgba(0,0,0,0.08);">
            <div style="font-size:12px;color:#6b7280;margin-bottom:4px;">📈 영상 성과 예측</div>
            <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">예상 성공 확률</div>
            <div class="prob-value {cls}" style="font-size:2.4rem;font-weight:700;">{prob_pct}%</div>
            <div style="font-size:12px;color:#6b7280;margin-top:4px;">{perf_label}</div>
            <div style="margin-top:10px;">
                <div style="font-size:11px;color:#9ca3af;margin-bottom:3px;">
                    도메인 평균 {avg_prob*100:.0f}%</div>
                <div style="font-size:12px;font-weight:600;
                color:{'#059669' if diff_perf > 0 else '#dc2626'};">
                    {perf_icon} {diff_perf_str}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**성과 확률 비교**")
        st.progress(avg_prob,           text=f"도메인 평균  {avg_prob*100:.0f}%")
        st.progress(min(prob_val, 1.0), text=f"선택 조건    {prob_pct}%")

    # ── 댓글 긍정 비율 카드 ───────────────────────────────
    with res_c2:
        sent_cls      = "prob-high" if sent_pct >= 70 else ("prob-mid" if sent_pct >= 50 else "prob-low")
        sent_label    = "매우 긍정적" if sent_pct >= 70 else ("보통" if sent_pct >= 50 else "주의 필요")
        sent_icon     = "✅" if sent_diff > 0 else "⚠️"
        sent_diff_str = f"+{sent_diff:.1f}%p" if sent_diff > 0 else f"{sent_diff:.1f}%p"

        st.markdown(f"""
        <div style="text-align:center;background:white;border-radius:14px;padding:20px;
        box-shadow:0 2px 10px rgba(0,0,0,0.08);">
            <div style="font-size:12px;color:#6b7280;margin-bottom:4px;">💬 댓글 반응 예측</div>
            <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">예상 긍정 댓글 비율</div>
            <div class="prob-value {sent_cls}" style="font-size:2.4rem;font-weight:700;">{sent_pct:.1f}%</div>
            <div style="font-size:12px;color:#6b7280;margin-top:4px;">{sent_label}</div>
            <div style="margin-top:10px;">
                <div style="font-size:11px;color:#9ca3af;margin-bottom:3px;">
                    도메인 평균 {avg_sent_pct:.0f}%</div>
                <div style="font-size:12px;font-weight:600;
                color:{'#059669' if sent_diff > 0 else '#dc2626'};">
                    {sent_icon} {sent_diff_str}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**긍정 댓글 비율 비교**")
        st.progress(avg_sent_pct / 100,       text=f"도메인 평균  {avg_sent_pct:.0f}%")
        st.progress(min(sent_pct / 100, 1.0), text=f"선택 조건    {sent_pct:.1f}%")

    # ── 핵심 성공 요인 TOP 5 (SHAP) ───────────────────────
    with res_c3:
        st.markdown("**🔑 핵심 성공 요인 TOP 5**")
        try:
            import shap as shap_lib

            # 도메인별 SHAP 전용 파일 사용 (FnB/IT 모두 지원)
            shap_obj  = shap_models[dom_key]
            shap_model = shap_obj["model"]
            shap_X     = shap_obj["X_test"]

            if not isinstance(shap_X, pd.DataFrame):
                if "feature_cols" in shap_obj:
                    shap_X = pd.DataFrame(shap_X, columns=shap_obj["feature_cols"])
                elif hasattr(shap_model, "feature_names_"):
                    shap_X = pd.DataFrame(shap_X, columns=list(shap_model.feature_names_))
                else:
                    shap_X = pd.DataFrame(shap_X)

            # SHAP sample 생성
            shap_sample_dict = {}
            for col in shap_X.columns:
                if pd.api.types.is_numeric_dtype(shap_X[col].dtype):
                    shap_sample_dict[col] = shap_X[col].median()
                else:
                    mv = shap_X[col].mode()
                    shap_sample_dict[col] = mv.iloc[0] if len(mv) > 0 else shap_X[col].iloc[0]

            for col, val in user_inputs_perf.items():
                if col in shap_X.columns:
                    shap_sample_dict[col] = val

            shap_sample = pd.DataFrame([shap_sample_dict])[shap_X.columns]
            for col in shap_X.columns:
                try:
                    shap_sample[col] = shap_sample[col].astype(shap_X[col].dtype)
                except (ValueError, TypeError):
                    pass

            explainer = shap_lib.TreeExplainer(shap_model)
            shap_vals = explainer(shap_sample)
            sv        = shap_vals.values[0]
            feats     = shap_X.columns.tolist()

            pairs     = sorted(zip(feats, sv), key=lambda x: abs(x[1]), reverse=True)[:5]

            for feat, val in pairs:
                dir_icon  = "↑" if val > 0 else "↓"
                dir_color = COL_GREEN if val > 0 else COL_RED
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                padding:10px;background:white;border-radius:8px;margin-bottom:6px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                    <span style="font-weight:500;font-size:13px;flex:2;">{k(feat)}</span>
                    <span style="color:{dir_color};font-size:18px;flex:0.3;">{dir_icon}</span>
                    <span style="color:{dir_color};font-weight:700;flex:1;text-align:right;">{val:+.2f}</span>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.info(f"SHAP 분석 오류: {e}")

    # ── 종합 판정 ─────────────────────────────────────────
    both_good     = diff_perf > 0 and sent_diff > 0
    both_bad      = diff_perf <= 0 and sent_diff <= 0
    overall_icon  = "✅" if both_good else ("⚠️" if both_bad else "🔶")
    overall_color = "insight-green" if both_good else ("insight-red" if both_bad else "insight-orange")
    overall_text  = (
        "성과·댓글 반응 모두 도메인 평균을 상회합니다. 업로드를 적극 권장합니다." if both_good else
        "성과·댓글 반응 모두 평균 이하입니다. 아래 전략 개선 후 재검토를 권장합니다." if both_bad else
        "성과와 댓글 반응 중 하나가 평균 이하입니다. 하단 전략 카드를 참고하세요."
    )
    st.markdown(f"""
    <div class="insight-box {overall_color}" style="margin-top:10px;">
        {overall_icon} {overall_text}
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # ④ AI 종합 코멘트 (PydanticAI + Vertex AI Gemini)
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="sec-label">🤖 AI 종합 요약</div>', unsafe_allow_html=True)

    prompt = f"""
아래는 YouTube 영상 업로드 조건과 두 AI 모델의 예측 결과입니다.

[입력 조건]
- 도메인: {dom_key}
- 콘텐츠 유형: {content_type}
- 마케팅 목적: {marketing_purpose}
- CTA 유형: {cta_type}
- 영상 길이: {length_bucket_kr}
- 업로드: {day_val} {hour_val}시
- 설명 길이: {desc_len}자 / 태그 수: {tags_count_val}개
- 자막: {'있음' if caption_use else '없음'}
- 채널 규모: {channel_tier_val}
- 시리즈: {'예' if cls_is_series else '아니오'} / 콜라보: {'예' if cls_is_collaboration else '아니오'} / 유료광고: {'예' if has_paid_val else '아니오'}

[예측 결과]
- 영상 성과 예측: 성공 확률 {prob_pct}% (도메인 평균 {avg_prob*100:.0f}% 대비 {'+' if diff_perf > 0 else ''}{diff_perf*100:.1f}%p)
- 긍정 댓글 비율 예측: {sent_pct:.1f}% (도메인 평균 {avg_sent_pct:.0f}% 대비 {'+' if sent_diff > 0 else ''}{sent_diff:.1f}%p)

위 조건을 종합해 실무자가 바로 활용할 수 있는 업로드 전략 코멘트를 작성해주세요.
"""

    with st.spinner("AI 코멘트 생성 중..."):
        try:
            ai_comment = get_ai_comment(prompt)
        except Exception as e:
            ai_comment = f"AI 코멘트 생성 중 오류가 발생했습니다: {e}"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:12px;
    padding:18px;color:white;margin-top:4px;">
        <div style="font-size:16px;font-weight:700;margin-bottom:10px;">🤖 AI 종합 코멘트</div>
        <div style="font-size:14px;line-height:1.8;opacity:0.95;">{ai_comment}</div>
    </div>""", unsafe_allow_html=True)


# ── 진입점 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    page_simulator()