import re
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from html import escape
from textwrap import dedent
from collections import Counter
import math

# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="YouTube Shorts Analysis Dashboard",
    page_icon="▶️",
    layout="wide",
)

# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent   # works/Hyeong_Uk/shorts_video_analysis
PROJECT_ROOT = BASE_DIR.parent.parent.parent  # final_project 루트 근처 추정

# 결과 CSV 후보들
RESULT_CANDIDATES = [
    BASE_DIR / "results" / "result_sample_shorts_all_for_video_agent_fixed.csv",
    BASE_DIR / "results" / "result_sample_shorts_all_for_video_agent.csv",

    BASE_DIR / "data" / "result_sample_shorts_all_for_video_agent_fixed.csv",
    BASE_DIR / "data" / "result_sample_shorts_all_for_video_agent.csv",

    BASE_DIR / "result_sample_shorts_all_for_video_agent_fixed.csv",
    BASE_DIR / "result_sample_shorts_all_for_video_agent.csv",

    PROJECT_ROOT / "works" / "Hyeong_Uk" / "shorts_video_analysis" / "results" / "result_sample_shorts_all_for_video_agent_fixed.csv",
    PROJECT_ROOT / "works" / "Hyeong_Uk" / "shorts_video_analysis" / "results" / "result_sample_shorts_all_for_video_agent.csv",

    PROJECT_ROOT / "works" / "Hyeong_Uk" / "shorts_video_analysis" / "data" / "result_sample_shorts_all_for_video_agent_fixed.csv",
    PROJECT_ROOT / "works" / "Hyeong_Uk" / "shorts_video_analysis" / "data" / "result_sample_shorts_all_for_video_agent.csv",
]

# SHAP 결과 후보들
SHAP_ORIGINAL_CANDIDATES = [
    BASE_DIR / "eda_outputs" / "shorts_shap_importance_original_feature.csv",
    BASE_DIR / "data" / "shorts_shap_importance_original_feature.csv",
    PROJECT_ROOT / "works" / "Hyeong_Uk" / "shorts_video_analysis" / "eda_outputs" / "shorts_shap_importance_original_feature.csv",
]

SHAP_ONEHOT_CANDIDATES = [
    BASE_DIR / "eda_outputs" / "shorts_shap_importance_onehot.csv",
    BASE_DIR / "data" / "shorts_shap_importance_onehot.csv",
    PROJECT_ROOT / "works" / "Hyeong_Uk" / "shorts_video_analysis" / "eda_outputs" / "shorts_shap_importance_onehot.csv",
]


# =========================================================
# 1. 유틸 함수
# =========================================================
def find_existing_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_data
def load_main_data():
    file_path = find_existing_file(RESULT_CANDIDATES)
    if file_path is None:
        return None, None

    df = pd.read_csv(file_path)

    # 기본 정리
    if "domain" in df.columns:
        df["domain"] = df["domain"].astype(str).str.replace("F&B", "FnB")
        df["domain"] = df["domain"].replace({"fnb": "FnB", "it": "IT", "FnB": "FnB", "IT": "IT"})

    if "success_label" in df.columns:
        df["success_label"] = df["success_label"].astype(str).str.lower().replace({
            "성공": "success",
            "실패": "fail"
        })

    # 숫자형 컬럼 변환
    numeric_candidates = [
        "person_ratio", "face_ratio", "text_ratio",
        "avg_brightness", "avg_blue", "avg_green", "avg_red"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, file_path


@st.cache_data
def load_shap_original():
    file_path = find_existing_file(SHAP_ORIGINAL_CANDIDATES)
    if file_path is None:
        return None, None
    df = pd.read_csv(file_path)
    return df, file_path


@st.cache_data
def load_shap_onehot():
    file_path = find_existing_file(SHAP_ONEHOT_CANDIDATES)
    if file_path is None:
        return None, None
    df = pd.read_csv(file_path)
    return df, file_path


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    if len(series.dropna()) == 0:
        return np.nan
    return series.mean()


def format_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.1%}"


def format_num(x, digit=3):
    if pd.isna(x):
        return "-"
    return f"{x:.{digit}f}"


def render_guide_card(title, guide_list):
    bullet_html = "".join([f"<li>{b}</li>" for b in guide_list])
    st.markdown(
        f"""
        <div class="guide-card">
            <div class="guide-title">{title}</div>
            <ul class="guide-list">
                {bullet_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_thumbnail(row):
    thumb = row.get("thumbnail", "")
    if pd.isna(thumb):
        thumb = ""
    return str(thumb)


def get_video_url(row):
    url = row.get("final_url", "")
    if pd.isna(url):
        url = ""
    return str(url)


def get_channel_name(row):
    if "채널명" in row and pd.notna(row["채널명"]):
        return str(row["채널명"])
    if "channel_title" in row and pd.notna(row["channel_title"]):
        return str(row["channel_title"])
    return "-"


def get_video_title(row):
    if "title" in row and pd.notna(row["title"]):
        return str(row["title"])
    return "제목 없음"

def truncate_text(text, max_len=36):
    text = "" if pd.isna(text) else str(text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def pick_representative_videos(df, domain, n=3):
    """
    도메인별 대표 성공 영상 추출
    """
    temp = df.copy()

    if "domain" in temp.columns:
        temp = temp[temp["domain"] == domain]

    if "success_label" in temp.columns:
        temp = temp[temp["success_label"] == "success"]

    if len(temp) == 0:
        return temp

    temp = temp.copy()
    temp["person_ratio"] = pd.to_numeric(temp.get("person_ratio"), errors="coerce").fillna(0)
    temp["face_ratio"] = pd.to_numeric(temp.get("face_ratio"), errors="coerce").fillna(0)
    temp["text_ratio"] = pd.to_numeric(temp.get("text_ratio"), errors="coerce").fillna(0)

    if domain == "FnB":
        temp["rep_score"] = (
            temp["person_ratio"] * 0.35
            + temp["face_ratio"] * 0.35
            + (temp.get("first_3sec", "").astype(str).eq("인물")).astype(int) * 0.15
            + (temp.get("motion_graphic", "").astype(str).eq("보조적")).astype(int) * 0.10
            + (temp.get("editing_pace", "").astype(str).isin(["빠름", "매우 빠름"])).astype(int) * 0.05
        )
    else:
        temp["rep_score"] = (
            temp["text_ratio"] * 0.20
            + (temp.get("motion_graphic", "").astype(str).eq("핵심요소")).astype(int) * 0.30
            + (temp.get("first_3sec", "").astype(str).eq("텍스트")).astype(int) * 0.20
            + (temp.get("video_format", "").astype(str).isin(["기술설명", "제품리뷰", "튜토리얼"])).astype(int) * 0.20
            + (temp.get("editing_pace", "").astype(str).isin(["빠름", "매우 빠름"])).astype(int) * 0.10
        )

    # 너무 같은 채널만 뽑히는 걸 줄이기 위해 순위 후 채널 중복 최소화
    temp = temp.sort_values("rep_score", ascending=False)

    selected_rows = []
    used_channels = set()

    for _, row in temp.iterrows():
        channel = get_channel_name(row)
        if channel not in used_channels:
            selected_rows.append(row)
            used_channels.add(channel)
        if len(selected_rows) >= n:
            break

    # 부족하면 그냥 추가
    if len(selected_rows) < n:
        used_titles = set([get_video_title(r) for r in selected_rows])
        for _, row in temp.iterrows():
            title = get_video_title(row)
            if title not in used_titles:
                selected_rows.append(row)
                used_titles.add(title)
            if len(selected_rows) >= n:
                break

    if len(selected_rows) == 0:
        return temp.head(0)

    return pd.DataFrame(selected_rows)


def make_domain_feature_summary(df):
    rows = []
    for domain in ["FnB", "IT"]:
        temp = df[df["domain"] == domain].copy()
        if len(temp) == 0:
            continue

        success = temp[temp["success_label"] == "success"]
        fail = temp[temp["success_label"] == "fail"]

        rows.append({
            "domain": domain,
            "metric": "인물 비율 평균",
            "success": safe_mean(success["person_ratio"]) if "person_ratio" in temp.columns else np.nan,
            "fail": safe_mean(fail["person_ratio"]) if "person_ratio" in temp.columns else np.nan,
        })
        rows.append({
            "domain": domain,
            "metric": "얼굴 비율 평균",
            "success": safe_mean(success["face_ratio"]) if "face_ratio" in temp.columns else np.nan,
            "fail": safe_mean(fail["face_ratio"]) if "face_ratio" in temp.columns else np.nan,
        })
        rows.append({
            "domain": domain,
            "metric": "텍스트 비율 평균",
            "success": safe_mean(success["text_ratio"]) if "text_ratio" in temp.columns else np.nan,
            "fail": safe_mean(fail["text_ratio"]) if "text_ratio" in temp.columns else np.nan,
        })

        if "first_3sec" in temp.columns:
            if domain == "FnB":
                succ_ratio = success["first_3sec"].astype(str).eq("인물").mean()
                fail_ratio = fail["first_3sec"].astype(str).eq("인물").mean()
                rows.append({
                    "domain": domain,
                    "metric": "첫 3초 인물 비율",
                    "success": succ_ratio,
                    "fail": fail_ratio,
                })
            else:
                succ_ratio = success["first_3sec"].astype(str).eq("텍스트").mean()
                fail_ratio = fail["first_3sec"].astype(str).eq("텍스트").mean()
                rows.append({
                    "domain": domain,
                    "metric": "첫 3초 텍스트 비율",
                    "success": succ_ratio,
                    "fail": fail_ratio,
                })

        if "motion_graphic" in temp.columns:
            if domain == "FnB":
                succ_ratio = success["motion_graphic"].astype(str).eq("보조적").mean()
                fail_ratio = fail["motion_graphic"].astype(str).eq("보조적").mean()
                rows.append({
                    "domain": domain,
                    "metric": "모션그래픽 보조적 활용 비율",
                    "success": succ_ratio,
                    "fail": fail_ratio,
                })
            else:
                succ_ratio = success["motion_graphic"].astype(str).eq("핵심요소").mean()
                fail_ratio = fail["motion_graphic"].astype(str).eq("핵심요소").mean()
                rows.append({
                    "domain": domain,
                    "metric": "모션그래픽 핵심 활용 비율",
                    "success": succ_ratio,
                    "fail": fail_ratio,
                })

    return pd.DataFrame(rows)

# =========================================================
# 도메인 인사이트 데이터 기반 계산 함수
# =========================================================

def pct_float(x):
    if pd.isna(x):
        return 0.0
    return round(float(x) * 100, 1)


def get_success_subset(df, domain):
    temp = df.copy()
    temp["success_label"] = temp["success_label"].astype(str).str.lower()
    return temp[
        (temp["domain"] == domain)
        & (temp["success_label"] == "success")
    ].copy()


def build_domain_candidate_metrics(df, domain):
    temp = df.copy()
    temp["success_label"] = temp["success_label"].astype(str).str.lower()

    success_df = temp[
        (temp["domain"] == domain)
        & (temp["success_label"] == "success")
    ].copy()

    fail_df = temp[
        (temp["domain"] == domain)
        & (temp["success_label"] == "fail")
    ].copy()

    metrics = []

    numeric_map = {
        "person_ratio": "인물 비율",
        "face_ratio": "얼굴 비율",
        "text_ratio": "텍스트 비율",
    }

    for col, label in numeric_map.items():
        if col in success_df.columns and col in fail_df.columns:
            s = pd.to_numeric(success_df[col], errors="coerce").dropna()
            f = pd.to_numeric(fail_df[col], errors="coerce").dropna()

            if len(s) > 0 and len(f) > 0:
                success_mean = s.mean()
                fail_mean = f.mean()
                diff = success_mean - fail_mean

                metrics.append({
                    "type": "numeric",
                    "feature": col,
                    "label": label,
                    "success_value": float(success_mean),
                    "fail_value": float(fail_mean),
                    "diff": float(diff),
                    "display_value": float(success_mean),
                })

    categorical_candidates = [
        ("first_3sec", "인물", "첫 3초 인물"),
        ("first_3sec", "텍스트", "첫 3초 텍스트"),
        ("motion_graphic", "보조적", "모션그래픽 보조"),
        ("motion_graphic", "핵심요소", "모션그래픽 핵심"),
        ("video_format", "기술설명", "기술설명 포맷"),
        ("video_format", "웹예능", "웹예능 포맷"),
        ("video_format", "광고/CF", "광고/CF 포맷"),
        ("video_format", "튜토리얼", "튜토리얼 포맷"),
        ("editing_pace", "빠름", "빠른 편집"),
        ("editing_pace", "매우 빠름", "매우 빠른 편집"),
    ]

    for col, value, label in categorical_candidates:
        if col in success_df.columns and col in fail_df.columns:
            s_ratio = success_df[col].astype(str).eq(value).mean()
            f_ratio = fail_df[col].astype(str).eq(value).mean()
            diff = s_ratio - f_ratio

            metrics.append({
                "type": "categorical",
                "feature": col,
                "value": value,
                "label": label,
                "success_value": float(s_ratio),
                "fail_value": float(f_ratio),
                "diff": float(diff),
                "display_value": float(s_ratio),
            })

    metrics_df = pd.DataFrame(metrics)

    if len(metrics_df) == 0:
        return metrics_df

    metrics_df["abs_diff"] = metrics_df["diff"].abs()

    # 성공 쪽에서 더 높은 항목을 우선 보여주기
    metrics_df = metrics_df.sort_values(
        ["diff", "abs_diff"],
        ascending=[False, False]
    )

    return metrics_df


def to_short_insight_text(row):
    feature = row.get("feature", "")
    value = str(row.get("value", ""))

    # 숫자형
    if feature in ["person_ratio", "face_ratio"]:
        return "인물/얼굴 비율 높음"

    if feature == "text_ratio":
        return "텍스트 비율 높음"

    # 범주형
    if feature == "first_3sec" and value == "인물":
        return "첫 3초 인물"
    if feature == "first_3sec" and value == "텍스트":
        return "첫 3초 텍스트"

    if feature == "motion_graphic" and value == "보조적":
        return "모션그래픽 보조 활용"
    if feature == "motion_graphic" and value == "핵심요소":
        return "모션그래픽 핵심 활용"

    if feature == "video_format" and value == "기술설명":
        return "기술설명형 포맷"
    if feature == "video_format" and value == "웹예능":
        return "웹예능형 포맷"
    if feature == "video_format" and value == "광고/CF":
        return "광고/CF 포맷"
    if feature == "video_format" and value == "튜토리얼":
        return "튜토리얼 포맷"

    if feature == "editing_pace" and value == "빠름":
        return "빠른 편집"
    if feature == "editing_pace" and value == "매우 빠름":
        return "매우 빠른 편집"

    # fallback
    return str(row.get("label", "핵심 특징"))


def build_domain_bullets(df, domain, n=3):
    metrics_df = build_domain_candidate_metrics(df, domain)

    if len(metrics_df) == 0:
        return ["데이터 부족"]

    positive_df = metrics_df[metrics_df["diff"] > 0].copy()

    if len(positive_df) == 0:
        positive_df = metrics_df.copy()

    positive_df["short_text"] = positive_df.apply(to_short_insight_text, axis=1)

    # 수치 기준 내림차순 정렬
    positive_df = positive_df.sort_values(
        ["display_value", "abs_diff"],
        ascending=[False, False]
    )

    bullets = []
    seen = set()

    for _, row in positive_df.iterrows():
        text = row["short_text"]

        # 중복 문구 제거
        if text not in seen:
            bullets.append(text)
            seen.add(text)

        if len(bullets) >= n:
            break

    return bullets


def build_domain_strengths(df, domain, n=5):
    metrics_df = build_domain_candidate_metrics(df, domain)

    if len(metrics_df) == 0:
        return []

    positive_df = metrics_df[metrics_df["diff"] > 0].copy()

    if len(positive_df) == 0:
        positive_df = metrics_df.copy()

    # 수치 기준 내림차순
    positive_df = positive_df.sort_values(
        ["display_value", "abs_diff"],
        ascending=[False, False]
    )

    topn = positive_df.head(n)

    strengths = []

    for _, row in topn.iterrows():
        strengths.append({
            "label": row["label"],
            "value": float(row["display_value"]),
        })

    return strengths


def extract_domain_keywords(df, domain, top_n=8):
    temp = get_success_subset(df, domain)

    if len(temp) == 0:
        return []

    text_sources = []

    for col in ["title", "video_format", "first_3sec", "motion_graphic", "background_style", "reason"]:
        if col in temp.columns:
            text_sources.extend(temp[col].dropna().astype(str).tolist())

    text = " ".join(text_sources)

    tokens = re.findall(r"[가-힣A-Za-z0-9+#]{2,}", text)

    stopwords = {
        "영상", "쇼츠", "shorts", "youtube", "유튜브",
        "성공", "실패", "분석", "기업", "채널", "도메인",
        "first", "3sec", "motion", "graphic", "format", "style",
        "보조적", "핵심요소", "인물", "텍스트", "장면", "기타",
        "사용", "활용", "구성", "중심", "요소", "비율",
        "나타남", "높음", "낮음", "많음", "있음",
    }

    tokens = [
        token for token in tokens
        if token.lower() not in stopwords
        and len(token) >= 2
    ]

    counter = Counter(tokens)

    return [word for word, _ in counter.most_common(top_n)]


def render_domain_insight_card(df, domain):
    color = "#ef233c" if domain == "FnB" else "#2563eb"
    fill_class = "strength-fill-red" if domain == "FnB" else "strength-fill-blue"
    chip_class = "keyword-chip-red" if domain == "FnB" else "keyword-chip-blue"
    icon_bg = "#fff1f2" if domain == "FnB" else "#eff6ff"
    icon_symbol = "🍴" if domain == "FnB" else "🖥️"
    check_bg = "#ef233c" if domain == "FnB" else "#2563eb"

    bullets = build_domain_bullets(df, domain, n=3)
    strengths = build_domain_strengths(df, domain, n=5)
    keywords = extract_domain_keywords(df, domain, top_n=8)

    if len(keywords) == 0:
        keywords = ["키워드 없음"]

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="insight-head">
                <div class="insight-icon-box" style="background:{icon_bg};">{icon_symbol}</div>
                <div>{domain} 도메인 인사이트</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns([1.25, 1.0, 0.9])

        with c1:
            for bullet in bullets:
                st.markdown(
                    f"""
                    <div class="insight-bullet">
                        <div class="insight-check" style="background:{check_bg};">✓</div>
                        <div class="insight-bullet-text">{escape(str(bullet))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with c2:
            st.markdown(
                '<div class="insight-section-title">특징 강도 (상위 지표)</div>',
                unsafe_allow_html=True
            )

            for item in strengths:
                width_pct = max(5, min(100, item["value"] * 100))

                st.markdown(
                    f"""
                    <div class="strength-item">
                        <div class="strength-row">
                            <span>{escape(str(item["label"]))}</span>
                            <span>{item["value"]:.2f}</span>
                        </div>
                        <div class="strength-bar">
                            <div class="{fill_class}" style="width:{width_pct}%"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with c3:
            st.markdown(
                '<div class="insight-section-title">주요 키워드</div>',
                unsafe_allow_html=True
            )

            chips_html = "".join([
                f'<span class="{chip_class}">{escape(str(keyword))}</span>'
                for keyword in keywords
            ])

            st.markdown(
                f'<div class="keyword-wrap">{chips_html}</div>',
                unsafe_allow_html=True
            )

# =========================================================
# AI 가이드라인 생성 함수 추가
# =========================================================

def pct(value):
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def get_ratio(series, target_value):
    if series is None or len(series) == 0:
        return np.nan
    return series.astype(str).eq(target_value).mean()


def get_top_shap_features(shap_original_df, domain, top_n=5):
    if shap_original_df is None or len(shap_original_df) == 0:
        return []

    temp = shap_original_df[shap_original_df["domain"] == domain].copy()

    if len(temp) == 0:
        return []

    temp = temp.sort_values("mean_abs_shap", ascending=False).head(top_n)

    return temp["original_feature"].tolist()


def build_rule_based_ai_guideline(df, shap_original_df, selected_domain="전체"):
    """
    API 호출 없이, 현재 분석 결과를 기반으로 가이드라인 문장을 생성하는 함수.
    - df: 숏츠 분석 결과 데이터프레임
    - shap_original_df: SHAP 원래 변수 기준 중요도 데이터프레임
    - selected_domain: 전체 / FnB / IT
    """

    domains = ["FnB", "IT"] if selected_domain == "전체" else [selected_domain]

    html_blocks = []

    for domain in domains:
        domain_df = df[df["domain"] == domain].copy()

        if len(domain_df) == 0:
            continue

        success_df = domain_df[domain_df["success_label"] == "success"].copy()
        fail_df = domain_df[domain_df["success_label"] == "fail"].copy()

        if len(success_df) == 0:
            continue

        # -----------------------------
        # 공통 수치 계산
        # -----------------------------
        success_person = pd.to_numeric(success_df.get("person_ratio"), errors="coerce").mean()
        fail_person = pd.to_numeric(fail_df.get("person_ratio"), errors="coerce").mean()

        success_face = pd.to_numeric(success_df.get("face_ratio"), errors="coerce").mean()
        fail_face = pd.to_numeric(fail_df.get("face_ratio"), errors="coerce").mean()

        success_text = pd.to_numeric(success_df.get("text_ratio"), errors="coerce").mean()
        fail_text = pd.to_numeric(fail_df.get("text_ratio"), errors="coerce").mean()

        first_person_ratio = get_ratio(success_df.get("first_3sec"), "인물") if "first_3sec" in success_df.columns else np.nan
        first_text_ratio = get_ratio(success_df.get("first_3sec"), "텍스트") if "first_3sec" in success_df.columns else np.nan

        motion_support_ratio = get_ratio(success_df.get("motion_graphic"), "보조적") if "motion_graphic" in success_df.columns else np.nan
        motion_core_ratio = get_ratio(success_df.get("motion_graphic"), "핵심요소") if "motion_graphic" in success_df.columns else np.nan

        tech_format_ratio = get_ratio(success_df.get("video_format"), "기술설명") if "video_format" in success_df.columns else np.nan

        top_shap_features = get_top_shap_features(shap_original_df, domain, top_n=5)
        top_shap_text = ", ".join(top_shap_features) if top_shap_features else "SHAP 결과 없음"

        # -----------------------------
        # 도메인별 가이드라인 생성
        # -----------------------------
        if domain == "FnB":
            guide_items = []

            if success_person > fail_person:
                guide_items.append(
                    f"성공 영상의 평균 person_ratio가 {success_person:.3f}로 실패 영상보다 높게 나타났습니다. "
                    f"따라서 제품만 보여주기보다 사람이 제품을 먹거나 사용하는 장면을 초반에 배치하는 구성이 좋습니다."
                )

            if success_face > fail_face:
                guide_items.append(
                    f"성공 영상의 평균 face_ratio가 {success_face:.3f}로 높게 나타났습니다. "
                    f"얼굴, 표정, 반응이 보이는 컷을 활용해 경험형 콘텐츠로 구성하는 것이 적합합니다."
                )

            if not pd.isna(first_person_ratio):
                guide_items.append(
                    f"성공 FnB 숏츠에서 첫 3초 인물 등장 비율은 {pct(first_person_ratio)}입니다. "
                    f"초반에는 텍스트 설명보다 인물의 행동이나 반응으로 시선을 잡는 전략을 우선 고려할 수 있습니다."
                )

            if not pd.isna(motion_support_ratio):
                guide_items.append(
                    f"성공 FnB 숏츠에서 모션그래픽 보조 활용 비율은 {pct(motion_support_ratio)}입니다. "
                    f"모션그래픽은 영상의 주인공이 아니라 자막, 포인트 강조, 리듬감 보완 용도로 사용하는 것이 좋습니다."
                )

            guide_items.append(
                "콘텐츠 흐름은 ‘상황 제시 → 제품 경험 → 반응/후기 → 짧은 메시지’ 구조로 설계하는 것을 추천합니다."
            )

            domain_summary = (
                f"<b>FnB 분석 기반 가이드라인</b><br>"
                f"<span style='color:#6b7280;'>SHAP 상위 변수: {escape(top_shap_text)}</span><br><br>"
            )

            guide_html = "".join([f"<li>{escape(item)}</li>" for item in guide_items])

            html_blocks.append(
                dedent(f"""
                <div style="margin-bottom:18px;">
                    <div style="font-size:18px; font-weight:900; color:#ef233c; margin-bottom:8px;">FnB</div>
                    <div style="font-size:13px; color:#374151; line-height:1.7; margin-bottom:8px;">
                        {domain_summary}
                    </div>
                    <ul style="line-height:1.8; margin-top:6px;">
                        {guide_html}
                    </ul>
                </div>
                """).strip()
            )

        elif domain == "IT":
            guide_items = []

            if not pd.isna(motion_core_ratio):
                guide_items.append(
                    f"성공 IT 숏츠에서 모션그래픽 핵심 활용 비율은 {pct(motion_core_ratio)}입니다. "
                    f"기능 흐름, 서비스 구조, 기술 개념을 말로만 설명하기보다 그래픽으로 시각화하는 구성이 적합합니다."
                )

            if not pd.isna(first_text_ratio):
                guide_items.append(
                    f"성공 IT 숏츠에서 첫 3초 텍스트 도입 비율은 {pct(first_text_ratio)}입니다. "
                    f"초반에는 문제 상황, 핵심 기능, 비교 포인트를 짧은 문장으로 먼저 제시하는 전략을 추천합니다."
                )

            if not pd.isna(tech_format_ratio):
                guide_items.append(
                    f"성공 IT 숏츠에서 기술설명형 포맷 비율은 {pct(tech_format_ratio)}입니다. "
                    f"기능 소개, 사용법, 전후 비교, 리스트형 설명처럼 구조화된 정보 전달 포맷을 우선 테스트해볼 수 있습니다."
                )

            if success_text >= fail_text:
                guide_items.append(
                    f"성공 영상의 평균 text_ratio는 {success_text:.3f}입니다. "
                    f"다만 텍스트를 많이 넣는 것 자체보다, 핵심 메시지를 짧고 명확하게 배치하는 것이 중요합니다."
                )

            guide_items.append(
                "콘텐츠 흐름은 ‘문제 제시 → 기능/해결책 제시 → 모션그래픽 설명 → 핵심 이점 요약’ 구조를 추천합니다."
            )

            domain_summary = (
                f"<b>IT 분석 기반 가이드라인</b><br>"
                f"<span style='color:#6b7280;'>SHAP 상위 변수: {escape(top_shap_text)}</span><br><br>"
            )

            guide_html = "".join([f"<li>{escape(item)}</li>" for item in guide_items])

            html_blocks.append(
                dedent(f"""
                <div style="margin-bottom:18px;">
                    <div style="font-size:18px; font-weight:900; color:#2563eb; margin-bottom:8px;">IT</div>
                    <div style="font-size:13px; color:#374151; line-height:1.7; margin-bottom:8px;">
                        {domain_summary}
                    </div>
                    <ul style="line-height:1.8; margin-top:6px;">
                        {guide_html}
                    </ul>
                </div>
                """).strip()
            )

    if not html_blocks:
        return """
        <b>가이드라인 생성 불가</b><br>
        현재 선택된 조건에서 충분한 데이터를 찾지 못했습니다.
        """

    return dedent("".join(html_blocks)).strip()

# =========================================================
# 추가 UI 렌더링 함수
# =========================================================

def kpi_message_card(label, value, desc, icon="▶", tone="default"):
    icon_class = ""
    value_class = ""

    if tone == "red":
        value_class = "red"
    elif tone == "blue":
        icon_class = "blue"
        value_class = "blue"
    elif tone == "yellow":
        icon_class = "yellow"

    return (
        f'<div class="kpi-card-v2">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value {value_class}">{escape(str(value))}</div>'
        f'<div class="desc">{escape(desc)}</div>'
        f'<div class="kpi-icon-circle {icon_class}">{icon}</div>'
        f'</div>'
    )


def mini_bar(label, value, color="red"):
    value = float(value)
    width = max(3, min(100, value * 100))
    fill_class = "blue" if color == "blue" else ""

    return (
        f'<div class="mini-bar-row">'
        f'<div class="mini-bar-label">{escape(label)}</div>'
        f'<div class="mini-bar-bg">'
        f'<div class="mini-bar-fill {fill_class}" style="width:{width}%"></div>'
        f'</div>'
        f'<div class="mini-bar-val">{value:.2f}</div>'
        f'</div>'
    )


def keyword_chips(words, color="red"):
    chip_class = "blue" if color == "blue" else ""
    return "".join(
        [f'<span class="keyword-chip {chip_class}">{escape(str(w))}</span>' for w in words]
    )


def check_items(items, color="red"):
    dot_class = "blue" if color == "blue" else ""
    html = ""

    for item in items:
        html += (
            f'<li>'
            f'<span class="check-dot {dot_class}">✓</span>'
            f'<span>{escape(str(item))}</span>'
            f'</li>'
        )

    return html


def render_domain_panel(domain):
    if domain == "FnB":
        color = "red"
        icon = "🍴"
        title = "FnB 도메인 인사이트"
        checks = [
            "인물/얼굴 비율 높음",
            "첫 3초 인물",
            "모션그래픽 보조 활용",
        ]
        bars = [
            ("인물 비율", 0.82),
            ("첫 3초 인물", 0.76),
            ("모션그래픽 보조", 0.61),
            ("경험/리액션 요소", 0.58),
            ("BGM/사운드 활용", 0.47),
        ]
        keywords = ["레시피", "먹방", "맛집", "ASMR", "비주얼", "반응", "꿀팁", "브이로그"]
        panel_class = "red"
        icon_class = ""
    else:
        color = "blue"
        icon = "🖥️"
        title = "IT 도메인 인사이트"
        checks = [
            "모션그래픽 핵심 활용",
            "첫 3초 텍스트",
            "기술설명형 포맷",
        ]
        bars = [
            ("모션그래픽 핵심", 0.84),
            ("첫 3초 텍스트", 0.78),
            ("기술설명형 포맷", 0.72),
            ("정보 밀도(텍스트)", 0.61),
            ("구조화/도식 활용", 0.50),
        ]
        keywords = ["기술설명", "비교", "기능", "튜토리얼", "꿀팁", "리뷰", "생산성", "업데이트"]
        panel_class = "blue"
        icon_class = "blue"

    bars_html = "".join([mini_bar(label, val, color=color) for label, val in bars])
    checks_html = check_items(checks, color=color)
    keywords_html = keyword_chips(keywords, color=color)

    return (
        f'<div class="domain-panel {panel_class}">'
        f'<div class="domain-title">'
        f'<span class="domain-icon {icon_class}">{icon}</span>'
        f'<span>{escape(title)}</span>'
        f'</div>'
        f'<div class="domain-grid">'
        f'<div><ul class="check-list">{checks_html}</ul></div>'
        f'<div>'
        f'<div class="mini-section-title">특징 강도 (상위 지표)</div>'
        f'{bars_html}'
        f'</div>'
        f'<div>'
        f'<div class="mini-section-title">주요 키워드</div>'
        f'<div class="keyword-wrap">{keywords_html}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def safe_thumbnail_url(row):
    thumb = str(row.get("thumbnail", "")).strip()

    if thumb and thumb != "nan" and thumb.startswith("http"):
        return thumb

    video_id = str(row.get("video_id", "")).strip()
    if video_id and video_id != "nan":
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    return "https://placehold.co/640x360?text=No+Thumbnail"


def choose_representative_videos(df, domain, n=2):
    temp = df.copy()
    temp["success_label"] = temp["success_label"].astype(str).str.lower()

    temp = temp[
        (temp["domain"] == domain)
        & (temp["success_label"] == "success")
    ].copy()

    if len(temp) == 0:
        return temp

    temp["person_ratio"] = pd.to_numeric(temp.get("person_ratio"), errors="coerce").fillna(0)
    temp["face_ratio"] = pd.to_numeric(temp.get("face_ratio"), errors="coerce").fillna(0)
    temp["text_ratio"] = pd.to_numeric(temp.get("text_ratio"), errors="coerce").fillna(0)

    if domain == "FnB":
        temp["rep_score"] = (
            temp["person_ratio"] * 0.35
            + temp["face_ratio"] * 0.35
            + temp.get("first_3sec", "").astype(str).eq("인물").astype(int) * 0.15
            + temp.get("motion_graphic", "").astype(str).eq("보조적").astype(int) * 0.10
            + temp.get("editing_pace", "").astype(str).isin(["빠름", "매우 빠름"]).astype(int) * 0.05
        )
    else:
        temp["rep_score"] = (
            temp["text_ratio"] * 0.20
            + temp.get("motion_graphic", "").astype(str).eq("핵심요소").astype(int) * 0.30
            + temp.get("first_3sec", "").astype(str).eq("텍스트").astype(int) * 0.20
            + temp.get("video_format", "").astype(str).isin(["기술설명", "제품리뷰", "튜토리얼"]).astype(int) * 0.20
            + temp.get("editing_pace", "").astype(str).isin(["빠름", "매우 빠름"]).astype(int) * 0.10
        )

    temp = temp.sort_values("rep_score", ascending=False)

    # 채널 다양성 확보
    selected_rows = []
    used_channels = set()

    for _, row in temp.iterrows():
        channel = get_channel_name(row)
        if channel not in used_channels:
            selected_rows.append(row)
            used_channels.add(channel)

        if len(selected_rows) >= n:
            break

    if len(selected_rows) < n:
        used_titles = set([get_video_title(r) for r in selected_rows])

        for _, row in temp.iterrows():
            title = get_video_title(row)
            if title not in used_titles:
                selected_rows.append(row)
                used_titles.add(title)

            if len(selected_rows) >= n:
                break

    return pd.DataFrame(selected_rows)


def render_case_card(row, domain):
    domain_badge_class = "case-domain-badge-red" if domain == "FnB" else "case-domain-badge-blue"
    feature_pill_class = "case-feature-pill-red" if domain == "FnB" else "case-feature-pill-blue"
    link_class = "case-link-red" if domain == "FnB" else "case-link-blue"

    title = escape(get_video_title(row))
    channel = escape(get_channel_name(row))
    url = escape(get_video_url(row))
    thumb = escape(safe_thumbnail_url(row))

    feature_items = []

    if pd.notna(row.get("first_3sec", None)):
        feature_items.append(f"first_3sec: {row.get('first_3sec')}")

    if pd.notna(row.get("motion_graphic", None)):
        feature_items.append(f"motion: {row.get('motion_graphic')}")

    if pd.notna(row.get("video_format", None)):
        feature_items.append(f"format: {row.get('video_format')}")

    if domain == "FnB" and pd.notna(row.get("person_ratio", None)):
        feature_items.append(f"person_ratio: {float(row.get('person_ratio')):.2f}")

    if domain == "IT" and pd.notna(row.get("text_ratio", None)):
        feature_items.append(f"text_ratio: {float(row.get('text_ratio')):.2f}")

    feature_items = feature_items[:4]

    features_html = "".join([
        f'<span class="{feature_pill_class}">{escape(str(item))}</span>'
        for item in feature_items
    ])

    st.markdown(
        f"""
        <div class="case-card">
            <div class="{domain_badge_class}">{domain}</div>
            <div class="case-thumb-wrap">
                <img src="{thumb}" class="case-thumb">
            </div>
            <div class="case-title">{title}</div>
            <div class="case-channel">{channel}</div>
            <div class="case-feature-wrap">
                {features_html}
            </div>
            <a href="{url}" target="_blank" class="case-link {link_class}">영상 보기 ↗</a>
        </div>
        """,
        unsafe_allow_html=True
    )


def guide_card_html():
    return (
        '<div class="guide-heading-red">FnB</div>'
        '<ul class="guide-ul">'
        '<li>인물 중심의 경험/리액션 숏츠로 공감과 몰입 유도</li>'
        '<li>첫 3초에 인물을 배치해 시선을 즉시 사로잡기</li>'
        '<li>모션그래픽은 보조적으로 사용하여 메시지 보완</li>'
        '<li>맛/비주얼/ASMR 요소로 감각 자극 강화</li>'
        '</ul>'
        '<div class="guide-heading-blue">IT</div>'
        '<ul class="guide-ul">'
        '<li>첫 3초에 텍스트 훅으로 핵심 메시지 전달</li>'
        '<li>모션그래픽을 핵심적으로 활용해 정보 이해도 향상</li>'
        '<li>기술설명형 포맷으로 구조화된 정보 제공</li>'
        '<li>비교/리스트 형식으로 명확한 가치 제안</li>'
        '</ul>'
    )

def render_sidebar_item(icon, label, active=False):
    active_class = "active" if active else ""

    html = (
        f'<div class="side-menu-item {active_class}">'
        f'<span class="side-menu-icon">{icon}</span>'
        f'<span>{label}</span>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# 대표 영상 관련 함수 추가
def format_duration(seconds):
    try:
        if pd.isna(seconds):
            return ""
        total = int(float(seconds))
        m = total // 60
        s = total % 60
        return f"{m}:{s:02d}"
    except Exception:
        return ""
    

def get_card_thumbnail(row):
    # 1순위: thumbnail 컬럼
    thumb = str(row.get("thumbnail", "")).strip()

    if thumb and thumb.lower() != "nan" and thumb.startswith("http"):
        return thumb

    # 2순위: video_id로 유튜브 썸네일 생성
    video_id = str(row.get("video_id", "")).strip()

    if video_id and video_id.lower() != "nan":
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    # 3순위: 빈 이미지
    return "https://placehold.co/640x360?text=No+Thumbnail"


def safe_str(val, default=""):
    if pd.isna(val):
        return default
    return str(val)

def pick_case_features(row):
    """
    카드에 보여줄 대표 특징 3~4개 선택
    """
    features = []

    first_3sec = safe_str(row.get("first_3sec"), "")
    motion = safe_str(row.get("motion_graphic"), "")
    video_format = safe_str(row.get("video_format"), "")
    person_ratio = row.get("person_ratio", None)
    face_ratio = row.get("face_ratio", None)
    text_ratio = row.get("text_ratio", None)

    if first_3sec:
        features.append(f"first_3sec: {first_3sec}")

    if motion:
        features.append(f"motion: {motion}")

    if video_format:
        features.append(f"format: {video_format}")

    # 도메인별로 마지막 수치형 1개 선택
    domain = safe_str(row.get("domain"), "")
    if domain == "FnB":
        if person_ratio is not None and not pd.isna(person_ratio):
            features.append(f"person_ratio: {float(person_ratio):.2f}")
        elif face_ratio is not None and not pd.isna(face_ratio):
            features.append(f"face_ratio: {float(face_ratio):.2f}")
    else:
        if text_ratio is not None and not pd.isna(text_ratio):
            features.append(f"text_ratio: {float(text_ratio):.2f}")
        elif person_ratio is not None and not pd.isna(person_ratio):
            features.append(f"person_ratio: {float(person_ratio):.2f}")

    return features[:4]

def render_case_card_horizontal(row):
    domain = str(row.get("domain", "FnB")).strip()
    domain_class = "fnb" if domain == "FnB" else "it"

    title_raw = row.get("title", "제목 없음")
    title = escape(truncate_text(title_raw, max_len=28))
    channel_raw = row.get("채널명", row.get("channel_title", "-"))
    channel = escape(truncate_text(channel_raw, max_len=18))
    url = escape(str(row.get("final_url", "#")))
    thumb = escape(get_card_thumbnail(row))

    duration = ""
    for duration_col in ["영상길이(초)", "duration", "duration_sec", "duration_seconds"]:
        if duration_col in row.index:
            duration = format_duration(row.get(duration_col))
            if duration:
                break

    first_3sec = row.get("first_3sec", "")
    motion = row.get("motion_graphic", "")
    video_format = row.get("video_format", "")
    person_ratio = row.get("person_ratio", None)
    text_ratio = row.get("text_ratio", None)

    chips = []

    if pd.notna(first_3sec) and str(first_3sec).strip():
        chips.append(f"first_3sec: {first_3sec}")

    if pd.notna(motion) and str(motion).strip():
        chips.append(f"motion: {motion}")

    if pd.notna(video_format) and str(video_format).strip():
        chips.append(f"format: {video_format}")

    if domain == "FnB":
        if pd.notna(person_ratio):
            chips.append(f"person_ratio: {float(person_ratio):.2f}")
    else:
        if pd.notna(text_ratio):
            chips.append(f"text_ratio: {float(text_ratio):.2f}")

    chips = chips[:3]

    chip_html = "".join(
        f'<span class="case-chip {domain_class}">{escape(str(chip))}</span>'
        for chip in chips
    )

    duration_html = (
        f'<div class="case-duration">{escape(duration)}</div>'
        if duration
        else ""
    )

    html = (
        f'<div class="case-card-horizontal">'
        f'  <div class="case-thumb-wrap">'
        f'    <div class="case-domain-badge {domain_class}">{escape(domain)}</div>'
        f'    <img src="{thumb}" class="case-thumb crop-shorts">'
        f'    {duration_html}'
        f'  </div>'
        f'  <div class="case-info-wrap">'
        f'    <div>'
        f'      <div class="case-title">{title}</div>'
        f'      <div class="case-channel">{channel}</div>'
        f'      <div class="case-chip-wrap">{chip_html}</div>'
        f'    </div>'
        f'    <div class="case-link-row">'
        f'      <a href="{url}" target="_blank">영상 보기 ↗</a>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# 2. CSS
# =========================================================
st.markdown("""
<style>
/* ================================
   Base
================================ */
.stApp {
    background: #f8f9fb;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Streamlit 기본 여백 줄이기 */
div[data-testid="stVerticalBlock"] {
    gap: 0.85rem;
}

/* ================================
   Top App Header
================================ */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #edf0f4;
    border-radius: 18px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
}

.app-header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.app-logo {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg, #ff1f1f, #e60000);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 19px;
    font-weight: 900;
}

.app-title {
    font-size: 28px;
    font-weight: 900;
    color: #111827;
    letter-spacing: -0.7px;
}

.app-subtitle {
    color: #6b7280;
    font-size: 14px;
    margin-top: 3px;
}

.app-header-right {
    color: #6b7280;
    font-size: 13px;
}

/* ================================
   KPI Cards
================================ */
.kpi-card-v2 {
    background: #ffffff;
    border: 1px solid #e9edf3;
    border-radius: 18px;
    padding: 18px 20px;
    min-height: 118px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    position: relative;
    overflow: hidden;
}

.kpi-card-v2 .label {
    color: #6b7280;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}

.kpi-card-v2 .value {
    font-size: 30px;
    font-weight: 900;
    color: #111827;
    line-height: 1.1;
}

.kpi-card-v2 .value.red {
    color: #ef233c;
}

.kpi-card-v2 .value.blue {
    color: #2563eb;
}

.kpi-card-v2 .desc {
    margin-top: 9px;
    color: #6b7280;
    font-size: 13px;
    line-height: 1.45;
}

.kpi-icon-circle {
    position: absolute;
    right: 18px;
    top: 22px;
    width: 58px;
    height: 58px;
    border-radius: 999px;
    background: #fee2e2;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}

.kpi-icon-circle.blue {
    background: #dbeafe;
}

.kpi-icon-circle.yellow {
    background: #fef3c7;
}

/* ================================
   Filter Bar
================================ */
.filter-bar {
    background: #ffffff;
    border: 1px solid #e9edf3;
    border-radius: 16px;
    padding: 12px 16px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    margin: 4px 0 12px 0;
}

.filter-label {
    color: #374151;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 4px;
}

/* ================================
   Domain Insight Panel
================================ */
.domain-panel {
    background: #ffffff;
    border: 1px solid #e9edf3;
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    min-height: 250px;
}

.domain-panel.red {
    border: 1px solid #ffc9c9;
    box-shadow: 0 5px 16px rgba(239, 35, 60, 0.06);
}

.domain-panel.blue {
    border: 1px solid #bfdbfe;
    box-shadow: 0 5px 16px rgba(37, 99, 235, 0.06);
}

.domain-title {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 19px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 14px;
}

.domain-icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: #fee2e2;
    color: #ef233c;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.domain-icon.blue {
    background: #dbeafe;
    color: #2563eb;
}

.domain-grid {
    display: grid;
    grid-template-columns: 1.05fr 1fr 0.95fr;
    gap: 16px;
    align-items: start;
}

.check-list {
    list-style: none;
    padding-left: 0;
    margin: 0;
}

.check-list li {
    font-size: 14px;
    color: #374151;
    margin: 0 0 12px 0;
    display: flex;
    gap: 8px;
    align-items: center;
}

.check-dot {
    width: 18px;
    height: 18px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 11px;
    font-weight: 900;
    background: #ef233c;
    flex-shrink: 0;
}

.check-dot.blue {
    background: #2563eb;
}

.mini-section-title {
    color: #6b7280;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 9px;
}

.mini-bar-row {
    display: grid;
    grid-template-columns: 86px 1fr 34px;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
}

.mini-bar-label {
    color: #374151;
    font-size: 12px;
    white-space: nowrap;
}

.mini-bar-bg {
    height: 9px;
    border-radius: 999px;
    background: #f1f5f9;
    overflow: hidden;
}

.mini-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff6b6b, #ef233c);
}

.mini-bar-fill.blue {
    background: linear-gradient(90deg, #60a5fa, #2563eb);
}

.mini-bar-val {
    color: #4b5563;
    font-size: 12px;
    text-align: right;
}

.keyword-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.keyword-chip {
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid #fecaca;
    color: #ef233c;
    background: #fff7f7;
}

.keyword-chip.blue {
    border: 1px solid #bfdbfe;
    color: #2563eb;
    background: #eff6ff;
}

/* ================================
   Compact Video Cards
================================ */
.case-section-title {
    font-size: 20px;
    font-weight: 900;
    color: #111827;
    margin-top: 10px;
    margin-bottom: 4px;
}

.case-subtitle {
    color: #6b7280;
    font-size: 12px;
    margin-bottom: 8px;
}

.video-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.compact-video-card {
    background: #ffffff;
    border: 1px solid #e9edf3;
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    display: grid;
    grid-template-columns: 120px 1fr;
    gap: 10px;
    min-height: 135px;
}

.compact-thumb {
    width: 120px;
    height: 116px;
    border-radius: 12px;
    object-fit: cover;
    background: linear-gradient(135deg, #111827, #374151);
}

.thumb-fallback {
    width: 120px;
    height: 116px;
    border-radius: 12px;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 13px;
}

.video-info {
    min-width: 0;
}

.video-domain {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 900;
    margin-bottom: 5px;
    background: #fee2e2;
    color: #ef233c;
}

.video-domain.blue {
    background: #dbeafe;
    color: #2563eb;
}

.video-title {
    font-size: 14px;
    font-weight: 900;
    color: #111827;
    line-height: 1.3;
    margin-bottom: 3px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.video-channel {
    color: #6b7280;
    font-size: 11px;
    margin-bottom: 6px;
}

.video-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 7px;
    background: #fff7f7;
    border: 1px solid #fecaca;
    color: #ef233c;
    font-size: 10.5px;
    font-weight: 700;
    margin: 0 3px 4px 0;
}

.video-chip.blue {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #2563eb;
}

.video-link {
    display: inline-block;
    color: #2563eb;
    font-size: 11px;
    font-weight: 800;
    text-decoration: none;
    margin-top: 2px;
}

/* ================================
   Bottom Cards
================================ */
.bottom-card {
    background: #ffffff;
    border: 1px solid #e9edf3;
    border-radius: 18px;
    padding: 16px 16px 10px 16px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    min-height: 330px;
}

.bottom-title {
    font-size: 16px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 8px;
}

.guide-heading-red {
    color: #ef233c;
    font-size: 15px;
    font-weight: 900;
    margin-bottom: 5px;
}

.guide-heading-blue {
    color: #2563eb;
    font-size: 15px;
    font-weight: 900;
    margin-top: 10px;
    margin-bottom: 5px;
}

.guide-ul {
    margin: 0;
    padding-left: 18px;
    color: #374151;
    font-size: 12.5px;
    line-height: 1.65;
}

/* ================================
   Sidebar
================================ */
section[data-testid="stSidebar"] {
    background: #f1f3f6;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #111827;
}

/* 버튼 */
.stButton > button {
    background: #ef233c;
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 900;
    height: 42px;
}

.stButton > button:hover {
    background: #d90429;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2-1. CSS Patch
# =========================================================

st.markdown("""
<style>
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #edf0f4;
    border-radius: 18px;
    padding: 18px 22px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
    min-height: 86px;
}

.app-title {
    font-size: 26px;
    font-weight: 900;
    color: #111827;
    letter-spacing: -0.7px;
    line-height: 1.25;
    margin-bottom: 4px;
}

.app-subtitle {
    color: #6b7280;
    font-size: 14px;
    line-height: 1.5;
}

/* 사이드바 메뉴 */
.side-menu-card {
    background: transparent;
    padding: 4px 0;
}

.side-menu-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 12px;
    border-radius: 12px;
    color: #4b5563;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 6px;
}

.side-menu-item.active {
    background: #fee2e2;
    color: #ef233c;
    border-right: 4px solid #ef233c;
}

.side-menu-icon {
    width: 22px;
    text-align: center;
    font-size: 16px;
}

/* 차트 카드: Streamlit 위젯 감싸는 용도 */
.chart-card {
    background: #ffffff;
    border: 1px solid #e9edf3;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    min-height: 360px;
}

.chart-title {
    font-size: 16px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 추가
# =========================================================

st.markdown("""
<style>
/* ---------- 공통 ---------- */
.chart-title {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 10px;
}

.small-muted {
    font-size: 12px;
    color: #64748b;
}

/* ---------- SHAP segmented ---------- */
div[data-testid="stSegmentedControl"] {
    margin-top: -2px;
}
div[data-testid="stSegmentedControl"] button {
    min-height: 32px !important;
    padding: 0 12px !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
}

/* ---------- 인사이트 카드 ---------- */
.insight-head {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 16px;
}
.insight-icon-box {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.insight-section-title {
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 12px;
    color: #374151;
}
.insight-bullet {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 20px;
    min-height: 54px;
}
.insight-check {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    color: white;
    font-size: 15px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.insight-bullet-text {
    font-size: 17px;
    font-weight: 700;
    line-height: 1.45;
    color: #1f2937;
    word-break: keep-all;
}

.strength-item {
    margin-bottom: 10px;
}
.strength-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 5px;
    color: #334155;
}
.strength-bar {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: #eef2f7;
    overflow: hidden;
}
.strength-fill-red {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff7b7b, #ef233c);
}
.strength-fill-blue {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #8db6ff, #2563eb);
}

.keyword-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.keyword-chip-red, .keyword-chip-blue {
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.keyword-chip-red {
    color: #ef233c;
    background: #fff1f2;
    border: 1px solid #fecdd3;
}
.keyword-chip-blue {
    color: #2563eb;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
}

/* ---------- 대표 영상 카드 ---------- */
.case-card {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 12px;
    background: white;
    height: 100%;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}
.case-thumb-wrap {
    width: 100%;
    aspect-ratio: 16 / 9;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 12px;
    background: #f1f5f9;
}
.case-thumb {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.case-domain-badge-red, .case-domain-badge-blue {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 8px;
}
.case-domain-badge-red {
    color: #ef233c;
    background: #fff1f2;
}
.case-domain-badge-blue {
    color: #2563eb;
    background: #eff6ff;
}
.case-title {
    font-size: 15px;
    font-weight: 800;
    line-height: 1.4;
    color: #111827;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 42px;
    margin-bottom: 6px;
}
.case-channel {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 10px;
}
.case-feature-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}
.case-feature-pill-red, .case-feature-pill-blue {
    border-radius: 999px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 700;
}
.case-feature-pill-red {
    color: #ef233c;
    background: #fff1f2;
    border: 1px solid #fecdd3;
}
.case-feature-pill-blue {
    color: #2563eb;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
}
.case-link {
    font-size: 13px;
    font-weight: 800;
    text-decoration: none;
}
.case-link-red {
    color: #ef233c;
}
.case-link-blue {
    color: #2563eb;
}

/* ---------- 카드 안 여백 ---------- */
.block-container {
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 대표 영상 사레 카드 CSS 패치
# =========================================================

st.markdown("""
<style>
/* ===== 대표 영상 사례 카드: 숏츠형 가로 카드 ===== */
.case-card-horizontal {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 10px;
    display: flex;
    gap: 12px;
    min-height: 205px;
    height: 205px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    overflow: hidden;
}

/* 숏츠 썸네일 영역: 세로형으로 잡아서 좌우 블러를 최대한 잘라냄 */
.case-thumb-wrap {
    position: relative;
    flex: 0 0 38%;
    max-width: 38%;
    height: 185px;
    border-radius: 12px;
    overflow: hidden;
    background: #f3f4f6;
}

/* 16:9 썸네일 안의 중앙 세로 숏츠 영역을 crop */
.case-thumb {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center center;
    display: block;
    border-radius: 12px;
    background: #f3f4f6;
}

.case-domain-badge {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 2;
    padding: 4px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    line-height: 1;
    background: #ffffff;
    border: 1px solid #e5e7eb;
}

.case-domain-badge.fnb {
    color: #ef233c;
}

.case-domain-badge.it {
    color: #2563eb;
}

.case-duration {
    position: absolute;
    right: 8px;
    bottom: 8px;
    z-index: 2;
    padding: 2px 7px;
    border-radius: 8px;
    background: rgba(17, 24, 39, 0.85);
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
}

.case-info-wrap {
    flex: 1;
    min-width: 0;
    height: 185px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.case-title {
    font-size: 13px;
    font-weight: 800;
    line-height: 1.35;
    color: #111827;
    margin-bottom: 4px;
    max-height: 36px;
    overflow: hidden;
}

.case-channel {
    font-size: 11.5px;
    color: #6b7280;
    margin-bottom: 7px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.case-chip-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    max-height: 84px;
    overflow: hidden;
    margin-bottom: 8px;
}

.case-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 10.5px;
    font-weight: 700;
    line-height: 1.1;
    border: 1px solid;
}

.case-chip.fnb {
    color: #ef233c;
    border-color: #f8b4bd;
    background: #fff5f6;
}

.case-chip.it {
    color: #2563eb;
    border-color: #bcd3ff;
    background: #f5f9ff;
}

.case-link-row {
    margin-top: auto;
    text-align: right;
}

.case-link-row a,
.case-link-row a.fnb,
.case-link-row a.it {
    font-size: 12px;
    font-weight: 800;
    text-decoration: none;
    color: #2563eb !important;
}

/* 모바일/좁은 화면 */
@media (max-width: 1200px) {
    .case-card-horizontal {
        flex-direction: column;
    }
    .case-thumb-wrap {
        max-width: 100%;
        flex-basis: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# 대표 영상 사례 추가2

st.markdown("""
<style>
/* =========================================================
   대표 영상 썸네일 최종 보정
   - 회색 여백 제거
   - 숏츠 중앙 영역만 보이도록 crop
   - 기존 case-thumb CSS 강제 덮어쓰기
========================================================= */

.case-card-horizontal {
    min-height: 150px !important;
    height: 150px !important;
    overflow: hidden !important;
}

/* 썸네일 박스를 고정 크기 세로형으로 설정 */
.case-thumb-wrap {
    position: relative !important;
    flex: 0 0 118px !important;
    width: 118px !important;
    max-width: 118px !important;
    height: 130px !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #f3f4f6 !important;
}

/* 이미지가 박스를 꽉 채우도록 강제 */
.case-thumb {
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    max-height: none !important;
    object-fit: cover !important;
    object-position: center center !important;
    aspect-ratio: auto !important;
    border-radius: 12px !important;
    display: block !important;
}

/* 숏츠 썸네일 좌우 블러 영역이 아직 보이면 중앙부를 살짝 확대 */
.case-thumb.crop-shorts {
    transform: scale(1.35) !important;
    transform-origin: center center !important;
}

/* 오른쪽 정보 영역 높이 맞춤 */
.case-info-wrap {
    height: 130px !important;
    min-height: 130px !important;
    display: flex !important;
    flex-direction: column !important;
}

/* 영상 보기 링크 우측 하단 고정 */
.case-link-row {
    margin-top: auto !important;
    text-align: right !important;
}

.case-link-row a,
.case-link-row a.fnb,
.case-link-row a.it {
    color: #2563eb !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    text-decoration: none !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. 데이터 로드
# =========================================================
df, data_path = load_main_data()
shap_original_df, shap_original_path = load_shap_original()
shap_onehot_df, shap_onehot_path = load_shap_onehot()

if df is None:
    st.error("메인 결과 CSV를 찾지 못했습니다. 경로를 확인해주세요.")
    st.code("\n".join([str(p) for p in RESULT_CANDIDATES]))
    st.stop()

# 필수 컬럼 확인
required_cols = ["domain", "success_label"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"필수 컬럼이 없습니다: {col}")
        st.stop()


# =========================================================
# 4. 사이드바
# =========================================================
with st.sidebar:
    st.markdown("### ▶ 기업 유튜브 성공 전략")
    st.caption("도메인별 콘텐츠 분석 대시보드")

    render_sidebar_item("🏠", "랜딩")
    render_sidebar_item("📊", "메타데이터 분석")
    render_sidebar_item("🖼️", "썸네일 분석")
    render_sidebar_item("💬", "댓글 분석")
    render_sidebar_item("🎬", "숏츠 영상 분석", active=True)
    render_sidebar_item("📘", "가이드라인")

    st.markdown("---")

    st.markdown("**이 페이지 구성**")
    st.caption("1. 개요")
    st.caption("2. 도메인 인사이트")
    st.caption("3. 대표 영상 사례")
    st.caption("4. SHAP 중요도")
    st.caption("5. 특징 비교")
    st.caption("6. 실행 가이드")

domain_filter = "전체"
success_filter = "전체"

# =========================================================
# 5. 필터 적용
# =========================================================
filtered_df = df.copy()

if domain_filter != "전체":
    filtered_df = filtered_df[filtered_df["domain"] == domain_filter]

if success_filter != "전체":
    filtered_df = filtered_df[filtered_df["success_label"] == success_filter]

# =========================================================
# 상단 Header + KPI
# =========================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-logo">▶</div>
            <div>
                <div class="app-title">숏츠 영상 분석</div>
                <div class="app-subtitle">도메인별 숏츠 영상의 성공 패턴을 분석하고 인사이트를 제공합니다.</div>
            </div>
        </div>
        <div class="app-header-right">📅 데이터 기준일 2024-05-20 &nbsp;&nbsp; ⓘ</div>
    </div>
    """,
    unsafe_allow_html=True
)

total_all = len(df)
fnb_n = (df["domain"] == "FnB").sum()
it_n = (df["domain"] == "IT").sum()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        kpi_message_card(
            "분석 영상 수",
            f"{total_all}",
            f"FnB {fnb_n}건  |  IT {it_n}건",
            icon="🎬",
            tone="red"
        ),
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        kpi_message_card(
            "FnB 성공 패턴",
            "인물·얼굴 중심",
            "인물 비중과 경험 중심 연출이 성과에 긍정적",
            icon="👤",
            tone="red"
        ),
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        kpi_message_card(
            "IT 성공 패턴",
            "정보 시각화형",
            "모션그래픽과 텍스트 훅이 성과에 긍정적",
            icon="🖥️",
            tone="blue"
        ),
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        kpi_message_card(
            "주요 시사점",
            "초반 3초가 중요",
            "도메인별 차별화된 오프닝 전략이 필요",
            icon="💡",
            tone="yellow"
        ),
        unsafe_allow_html=True
    )

# =========================================================
# 본문 가로 필터 바
# =========================================================

st.markdown('<div class="filter-bar">', unsafe_allow_html=True)

f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1.15])

with f1:
    st.markdown('<div class="filter-label">도메인</div>', unsafe_allow_html=True)
    domain_filter = st.selectbox(
        "도메인",
        ["전체", "FnB", "IT"],
        index=0,
        label_visibility="collapsed",
        key="domain_filter_main"
    )

with f2:
    st.markdown('<div class="filter-label">성공여부</div>', unsafe_allow_html=True)
    success_filter = st.selectbox(
        "성공여부",
        ["전체", "success", "fail"],
        index=0,
        label_visibility="collapsed",
        key="success_filter_main"
    )

with f3:
    st.markdown('<div class="filter-label">채널 규모</div>', unsafe_allow_html=True)
    st.selectbox(
        "채널 규모",
        ["전체", "대형", "중형", "소형"],
        label_visibility="collapsed",
        key="channel_size_dummy"
    )

with f4:
    st.markdown('<div class="filter-label">기간</div>', unsafe_allow_html=True)
    st.selectbox(
        "기간",
        ["전체", "최근 90일", "최근 6개월", "최근 1년"],
        label_visibility="collapsed",
        key="period_dummy"
    )

with f5:
    st.markdown('<div class="filter-label">&nbsp;</div>', unsafe_allow_html=True)
    if st.button("✨ AI 가이드라인 생성", use_container_width=True):
        st.session_state["show_ai_guide"] = True

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get("show_ai_guide", False):
    selected_domain_for_guide = domain_filter_main if "domain_filter_main" in locals() else domain_filter

    guide_html = build_rule_based_ai_guideline(
        df=df,
        shap_original_df=shap_original_df,
        selected_domain=selected_domain_for_guide
    )

    st.markdown(
        dedent(f"""
        <div class="note-box">
            <div style="font-size:18px; font-weight:900; color:#111827; margin-bottom:10px;">
                ✨ 분석 결과 기반 AI 가이드라인
            </div>
            <div style="font-size:13px; color:#6b7280; margin-bottom:12px;">
                현재 대시보드의 성공 숏츠 분석 결과, 도메인별 성공 패턴, SHAP 상위 변수를 기반으로 자동 생성한 가이드라인입니다.
            </div>
            {guide_html}
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

# =========================================================
# 도메인 인사이트 패널
# =========================================================

p1, p2 = st.columns(2)

with p1:
    render_domain_insight_card(df, "FnB")

with p2:
    render_domain_insight_card(df, "IT")

# =========================================================
# Compact 대표 영상 사례
# =========================================================

st.markdown(
    '<div class="case-section-title">대표 영상 사례</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="case-subtitle">성공 패턴을 잘 보여주는 대표 숏츠 영상을 자동으로 추출했습니다.</div>',
    unsafe_allow_html=True
)

# FnB 2개 + IT 2개 대표 영상 추출
fnb_rep_df = choose_representative_videos(df, domain="FnB", n=2)
it_rep_df = choose_representative_videos(df, domain="IT", n=2)

# 하나의 대표 사례 데이터프레임으로 합치기
case_df = pd.concat([fnb_rep_df, it_rep_df], axis=0).reset_index(drop=True)

# 4개 카드 한 줄 배치
case_cols = st.columns(4, gap="small")

for i, (_, row) in enumerate(case_df.head(4).iterrows()):
    with case_cols[i]:
        render_case_card_horizontal(row)

# 대표 영상 카드와 하단 차트 영역 사이 여백
st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)

# =========================================================
# 하단 3열: SHAP / 비교 / 실행 가이드
# =========================================================

b1, b2, b3 = st.columns([1.05, 1.25, 1.05])

with b1:
    with st.container(border=True):

        # 선택값 기본 설정
        if "shap_domain_filter_value" not in st.session_state:
            st.session_state["shap_domain_filter_value"] = "FnB"

        title_col, filter_col = st.columns([1.85, 1.15])

        with title_col:
            st.markdown(
                '<div class="chart-title">SHAP 주요 변수 중요도</div>',
                unsafe_allow_html=True
            )

        with filter_col:
            btn_fnb, btn_it = st.columns(2)

            with btn_fnb:
                if st.button("FnB", key="shap_btn_fnb", use_container_width=True):
                    st.session_state["shap_domain_filter_value"] = "FnB"

            with btn_it:
                if st.button("IT", key="shap_btn_it", use_container_width=True):
                    st.session_state["shap_domain_filter_value"] = "IT"

        shap_domain_filter = st.session_state["shap_domain_filter_value"]

        # SHAP 토글 버튼 색상 설정
        if shap_domain_filter == "FnB":
            fnb_bg = "#ef233c"
            fnb_color = "white"
            fnb_border = "#ef233c"

            it_bg = "white"
            it_color = "#111827"
            it_border = "#d1d5db"
        else:
            fnb_bg = "white"
            fnb_color = "#111827"
            fnb_border = "#d1d5db"

            it_bg = "#2563eb"
            it_color = "white"
            it_border = "#2563eb"

        # 기존 전역 .stButton 스타일보다 우선 적용되도록 key 기반으로 덮어쓰기
        st.markdown(
            f"""
            <style>
            .st-key-shap_btn_fnb button {{
                background-color: {fnb_bg} !important;
                color: {fnb_color} !important;
                border: 1px solid {fnb_border} !important;
                border-radius: 999px !important;
                font-weight: 700 !important;
                min-height: 32px !important;
                height: 32px !important;
                padding: 0 12px !important;
                box-shadow: none !important;
            }}

            .st-key-shap_btn_it button {{
                background-color: {it_bg} !important;
                color: {it_color} !important;
                border: 1px solid {it_border} !important;
                border-radius: 999px !important;
                font-weight: 700 !important;
                min-height: 32px !important;
                height: 32px !important;
                padding: 0 12px !important;
                box-shadow: none !important;
            }}

            .st-key-shap_btn_fnb button:hover {{
                background-color: {fnb_bg} !important;
                color: {fnb_color} !important;
                border-color: #ef233c !important;
            }}

            .st-key-shap_btn_it button:hover {{
                background-color: {it_bg} !important;
                color: {it_color} !important;
                border-color: #2563eb !important;
            }}

            .st-key-shap_btn_fnb button:focus,
            .st-key-shap_btn_it button:focus {{
                box-shadow: none !important;
                outline: none !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        if shap_original_df is not None:
            shap_temp = shap_original_df.copy()
            shap_temp = shap_temp[shap_temp["domain"] == shap_domain_filter].copy()

            color = "#ef233c" if shap_domain_filter == "FnB" else "#2563eb"

            shap_temp = shap_temp.sort_values("mean_abs_shap", ascending=False).head(7)

            fig = px.bar(
                shap_temp.sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="original_feature",
                orientation="h",
                text="mean_abs_shap",
                color_discrete_sequence=[color],
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="inside",
                textfont_color="white"
            )

            fig.update_layout(
                height=280,
                template="plotly_white",
                margin=dict(l=0, r=0, t=5, b=0),
                xaxis_title="평균 |SHAP value|",
                yaxis_title="",
                showlegend=False,
                font=dict(size=11),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("SHAP 결과 파일이 없습니다.")


with b2:
    with st.container(border=True):
        st.markdown('<div class="chart-title">도메인별 성공 특징 비교</div>', unsafe_allow_html=True)

        compare_rows = []

        for d in ["FnB", "IT"]:
            tmp = df[(df["domain"] == d) & (df["success_label"] == "success")].copy()

            compare_rows.append({
                "metric": "첫 3초 인물",
                "domain": d,
                "value": tmp["first_3sec"].astype(str).eq("인물").mean() if "first_3sec" in tmp.columns else np.nan,
            })
            compare_rows.append({
                "metric": "첫 3초 텍스트",
                "domain": d,
                "value": tmp["first_3sec"].astype(str).eq("텍스트").mean() if "first_3sec" in tmp.columns else np.nan,
            })
            compare_rows.append({
                "metric": "모션그래픽 핵심활용",
                "domain": d,
                "value": tmp["motion_graphic"].astype(str).eq("핵심요소").mean() if "motion_graphic" in tmp.columns else np.nan,
            })
            compare_rows.append({
                "metric": "기술설명형 포맷",
                "domain": d,
                "value": tmp["video_format"].astype(str).eq("기술설명").mean() if "video_format" in tmp.columns else np.nan,
            })
            compare_rows.append({
                "metric": "인물 비율 0.7+",
                "domain": d,
                "value": (pd.to_numeric(tmp["person_ratio"], errors="coerce") >= 0.7).mean() if "person_ratio" in tmp.columns else np.nan,
            })

        compare_df = pd.DataFrame(compare_rows)

        fig = px.bar(
            compare_df,
            x="metric",
            y="value",
            color="domain",
            barmode="group",
            text_auto=".0%",
            color_discrete_map={
                "FnB": "#ef233c",
                "IT": "#2563eb",
            },
        )
        fig.update_layout(
            height=310,
            template="plotly_white",
            margin=dict(l=0, r=0, t=5, b=0),
            xaxis_title="",
            yaxis_title="",
            legend_title="",
            font=dict(size=11),
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)


with b3:
    with st.container(border=True):
        st.markdown('<div class="chart-title">🎯 실행 가이드</div>', unsafe_allow_html=True)
        st.markdown(guide_card_html(), unsafe_allow_html=True)