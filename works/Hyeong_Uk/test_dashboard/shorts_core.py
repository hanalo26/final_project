from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Shorts analysis shared core
# 위치 권장: works/Hyeong_Uk/test_dashboard/shorts_core.py
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RESULT_CANDIDATES = [
    DATA_DIR / "result_sample_shorts_all_for_video_agent_fixed.csv",
    DATA_DIR / "result_sample_shorts_all_for_video_agent.csv",
    BASE_DIR / "result_sample_shorts_all_for_video_agent_fixed.csv",
    BASE_DIR / "result_sample_shorts_all_for_video_agent.csv",
    Path("works/Hyeong_Uk/test_dashboard/data/result_sample_shorts_all_for_video_agent_fixed.csv"),
    Path("works/Hyeong_Uk/test_dashboard/data/result_sample_shorts_all_for_video_agent.csv"),
    Path("works/Hyeong_Uk/shorts_video_analysis/results/result_sample_shorts_all_for_video_agent_fixed.csv"),
    Path("works/Hyeong_Uk/shorts_video_analysis/results/result_sample_shorts_all_for_video_agent.csv"),
    Path("works/Hyeong_Uk/shorts_video_analysis/data/result_sample_shorts_all_for_video_agent_fixed.csv"),
    Path("works/Hyeong_Uk/shorts_video_analysis/data/result_sample_shorts_all_for_video_agent.csv"),
]

SHAP_ORIGINAL_CANDIDATES = [
    DATA_DIR / "shorts_shap_importance_original_feature.csv",
    BASE_DIR / "shorts_shap_importance_original_feature.csv",
    Path("works/Hyeong_Uk/shorts_video_analysis/eda_outputs/shorts_shap_importance_original_feature.csv"),
    Path("works/Hyeong_Uk/test_dashboard/data/shorts_shap_importance_original_feature.csv"),
]

SHAP_ONEHOT_CANDIDATES = [
    DATA_DIR / "shorts_shap_importance_onehot.csv",
    BASE_DIR / "shorts_shap_importance_onehot.csv",
    Path("works/Hyeong_Uk/shorts_video_analysis/eda_outputs/shorts_shap_importance_onehot.csv"),
    Path("works/Hyeong_Uk/test_dashboard/data/shorts_shap_importance_onehot.csv"),
]


DOMAIN_STYLE = {
    "FnB": {
        "main": "#ef233c",
        "sub": "#fb7185",
        "light": "#fff1f2",
        "border": "#fecdd3",
        "accent": "#f97316",
        "gradient": "linear-gradient(135deg, #ef233c 0%, #fb7185 100%)",
        "emoji": "🍴",
    },
    "IT": {
        "main": "#2563eb",
        "sub": "#60a5fa",
        "light": "#eff6ff",
        "border": "#bfdbfe",
        "accent": "#06b6d4",
        "gradient": "linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)",
        "emoji": "💻",
    },
}


FNB_DEFAULT = {
    "person_ratio": 0.786,
    "face_ratio": 0.649,
    "text_ratio": 0.31,
    "avg_brightness": 142.0,
    "first_3sec": {"인물": 62.0, "텍스트": 18.0, "제품": 12.0, "기타": 8.0},
    "motion_graphic": {"보조적": 58.0, "핵심요소": 22.0, "거의없음": 20.0},
    "video_format": {"웹예능": 34.0, "제품리뷰": 28.0, "광고/CF": 16.0, "기타": 22.0},
    "summary": "사람이 제품을 경험하는 장면과 얼굴 반응이 중요합니다.",
}

IT_DEFAULT = {
    "person_ratio": 0.46,
    "face_ratio": 0.32,
    "text_ratio": 0.52,
    "avg_brightness": 138.0,
    "first_3sec": {"텍스트": 54.0, "제품": 18.0, "인물": 15.0, "기타": 13.0},
    "motion_graphic": {"핵심요소": 52.0, "보조적": 31.0, "거의없음": 17.0},
    "video_format": {"기술설명": 36.0, "튜토리얼": 24.0, "제품리뷰": 18.0, "기타": 22.0},
    "summary": "짧은 시간 안에 핵심 메시지와 기능을 이해시키는 구성이 중요합니다.",
}


FNB_IMPORTANCE_DEFAULT = [
    {"label": "사람 등장 비율", "score": 0.95, "direction": "up", "desc": "제품 경험을 쉽게 이해"},
    {"label": "얼굴·표정 비율", "score": 0.88, "direction": "up", "desc": "반응과 감정 전달"},
    {"label": "첫 3초 인물 등장", "score": 0.74, "direction": "up", "desc": "초반 시선 확보"},
    {"label": "그래픽 보조 활용", "score": 0.58, "direction": "up", "desc": "메시지 보조 효과"},
    {"label": "제품 경험형 포맷", "score": 0.51, "direction": "up", "desc": "경험 중심 구성"},
]

IT_IMPORTANCE_DEFAULT = [
    {"label": "화면 속 텍스트 비율", "score": 0.91, "direction": "up", "desc": "핵심 메시지 빠른 전달"},
    {"label": "첫 3초 핵심 문구", "score": 0.86, "direction": "up", "desc": "초반 이해도 상승"},
    {"label": "그래픽 핵심 활용", "score": 0.79, "direction": "up", "desc": "복잡한 정보 시각화"},
    {"label": "기술 설명형 포맷", "score": 0.62, "direction": "up", "desc": "문제와 해결책 전달"},
    {"label": "빠른 편집 흐름", "score": 0.54, "direction": "up", "desc": "정보 전달 속도 상승"},
]


def find_existing_file(candidates: list[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_shorts_data() -> tuple[pd.DataFrame, Optional[Path]]:
    file_path = find_existing_file(RESULT_CANDIDATES)
    if file_path is None:
        return pd.DataFrame(), None

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

    if "domain" in df.columns:
        df["domain"] = (
            df["domain"].astype(str)
            .str.replace("F&B", "FnB", regex=False)
            .replace({"fnb": "FnB", "FNB": "FnB", "FnB": "FnB", "it": "IT", "IT": "IT"})
        )

    if "success_label" in df.columns:
        df["success_label"] = (
            df["success_label"].astype(str).str.lower()
            .replace({"성공": "success", "실패": "fail", "success": "success", "fail": "fail"})
        )
    elif "grade" in df.columns:
        df["success_label"] = (
            df["grade"].astype(str).replace({"성공": "success", "실패": "fail"})
        )

    for col in [
        "person_ratio", "face_ratio", "text_ratio",
        "avg_brightness", "avg_blue", "avg_green", "avg_red",
        "duration", "영상길이(초)"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, file_path


@st.cache_data(show_spinner=False)
def load_shap_data() -> tuple[pd.DataFrame, Optional[Path]]:
    """숏츠 중요 요소 파일 로드.

    기본은 원핫 인코딩된 중요도 파일을 우선 사용합니다.
    원핫 파일이 없을 때만 원본 피처 기준 중요도 파일을 사용합니다.
    """
    path = find_existing_file(SHAP_ONEHOT_CANDIDATES)
    if path is None:
        path = find_existing_file(SHAP_ORIGINAL_CANDIDATES)
    if path is None:
        return pd.DataFrame(), None

    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig")

    if "domain" in df.columns:
        df["domain"] = (
            df["domain"].astype(str)
            .str.replace("F&B", "FnB", regex=False)
            .replace({"fnb": "FnB", "FNB": "FnB", "FnB": "FnB", "it": "IT", "IT": "IT"})
        )

    for col in ["mean_abs_shap", "importance", "score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, path


def get_counts(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0, "success": 0, "fail": 0, "fnb": 0, "it": 0}

    return {
        "total": len(df),
        "success": int((df.get("success_label", pd.Series(dtype=str)) == "success").sum()),
        "fail": int((df.get("success_label", pd.Series(dtype=str)) == "fail").sum()),
        "fnb": int((df.get("domain", pd.Series(dtype=str)) == "FnB").sum()),
        "it": int((df.get("domain", pd.Series(dtype=str)) == "IT").sum()),
    }


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.mean()) if not s.empty else float("nan")


def _rate(df: pd.DataFrame, col: str, value: str) -> float:
    if col not in df.columns or df.empty:
        return float("nan")
    return float(df[col].astype(str).eq(value).mean() * 100)


def _dist(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns or df.empty:
        return {}
    return (df[col].dropna().astype(str).value_counts(normalize=True) * 100).round(1).to_dict()


def compute_domain_pattern(df: pd.DataFrame, domain: str) -> dict:
    if df.empty or "domain" not in df.columns or "success_label" not in df.columns:
        return FNB_DEFAULT.copy() if domain == "FnB" else IT_DEFAULT.copy()

    success_df = df[(df["domain"] == domain) & (df["success_label"] == "success")].copy()
    fail_df = df[(df["domain"] == domain) & (df["success_label"] == "fail")].copy()

    if success_df.empty:
        return FNB_DEFAULT.copy() if domain == "FnB" else IT_DEFAULT.copy()

    pattern = {
        "person_ratio": _safe_mean(success_df, "person_ratio"),
        "face_ratio": _safe_mean(success_df, "face_ratio"),
        "text_ratio": _safe_mean(success_df, "text_ratio"),
        "avg_brightness": _safe_mean(success_df, "avg_brightness"),
        "first_3sec": _dist(success_df, "first_3sec"),
        "motion_graphic": _dist(success_df, "motion_graphic"),
        "video_format": _dist(success_df, "video_format"),
        "_n_success": len(success_df),
        "_n_fail": len(fail_df),
        "_from_csv": True,
    }

    if domain == "FnB":
        pattern["summary"] = "성공 숏츠는 인물·얼굴·제품 경험 장면이 더 두드러지는 경향이 있습니다."
    else:
        pattern["summary"] = "성공 숏츠는 텍스트와 모션그래픽으로 핵심 메시지를 빠르게 전달하는 경향이 있습니다."

    return pattern


def get_patterns(df: pd.DataFrame) -> tuple[dict, dict]:
    return compute_domain_pattern(df, "FnB"), compute_domain_pattern(df, "IT")


def build_metric_comparison(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """성공/실패 평균 비교용 데이터."""
    if df.empty or "domain" not in df.columns or "success_label" not in df.columns:
        base = FNB_DEFAULT if domain == "FnB" else IT_DEFAULT
        rows = [
            {"지표": "인물 비율", "성공": base["person_ratio"], "실패": max(base["person_ratio"] - 0.18, 0)},
            {"지표": "얼굴 비율", "성공": base["face_ratio"], "실패": max(base["face_ratio"] - 0.20, 0)},
            {"지표": "텍스트 비율", "성공": base["text_ratio"], "실패": max(base["text_ratio"] - 0.12, 0)},
        ]
        return pd.DataFrame(rows)

    domain_df = df[df["domain"] == domain].copy()
    success_df = domain_df[domain_df["success_label"] == "success"].copy()
    fail_df = domain_df[domain_df["success_label"] == "fail"].copy()

    metric_map = {
        "person_ratio": "인물 비율",
        "face_ratio": "얼굴 비율",
        "text_ratio": "텍스트 비율",
    }

    rows = []
    for col, label in metric_map.items():
        if col in domain_df.columns:
            rows.append({
                "지표": label,
                "성공": _safe_mean(success_df, col),
                "실패": _safe_mean(fail_df, col),
            })

    return pd.DataFrame(rows)


def get_distribution_df(pattern: dict, key: str, label: str) -> pd.DataFrame:
    dist = pattern.get(key, {}) or {}
    if not dist:
        return pd.DataFrame(columns=[label, "비율"])
    return pd.DataFrame([{label: k, "비율": v} for k, v in dist.items()])


FEATURE_NAME_MAP = {
    "person_ratio": "사람 등장 비율",
    "face_ratio": "얼굴·표정 비율",
    "text_ratio": "화면 속 텍스트 비율",
    "avg_brightness": "영상 밝기",
    "avg_red": "붉은색 계열",
    "avg_green": "초록색 계열",
    "avg_blue": "파란색 계열",
    "first_3sec": "첫 3초 구성",
    "first_3sec_인물": "첫 3초 인물 등장",
    "first_3sec_텍스트": "첫 3초 핵심 문구",
    "first_3sec_제품": "첫 3초 제품 노출",
    "first_3sec_기타": "첫 3초 기타 장면",
    "motion_graphic": "그래픽 활용",
    "motion_graphic_핵심요소": "그래픽 핵심 활용",
    "motion_graphic_보조적": "그래픽 보조 활용",
    "motion_graphic_거의없음": "그래픽 거의 없음",
    "video_format": "영상 포맷",
    "video_format_웹예능": "웹예능형 포맷",
    "video_format_제품리뷰": "제품 경험형 포맷",
    "video_format_기술설명": "기술 설명형 포맷",
    "video_format_튜토리얼": "튜토리얼형 포맷",
    "video_format_광고/CF": "광고형 포맷",
    "video_format_인터뷰": "인터뷰형 포맷",
    "editing_pace": "편집 속도",
    "editing_pace_빠름": "빠른 편집 흐름",
    "editing_pace_매우 빠름": "매우 빠른 편집 흐름",
    "production_quality": "영상 완성도",
    "lighting_style": "조명 스타일",
    "color_mood": "색감 분위기",
}


def pretty_feature_name(raw_name: str) -> str:
    raw_name = str(raw_name)
    if raw_name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[raw_name]

    clean = raw_name
    for prefix in ["cls_", "cat__", "num__", "original_", "feature_", "onehot__"]:
        clean = clean.replace(prefix, "")

    clean = clean.replace("=", "_").replace(":", "_")

    if clean in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[clean]

    for key, value in FEATURE_NAME_MAP.items():
        if clean == key or clean.endswith("_" + key) or key in clean:
            return value

    # 자주 나오는 범주형 원핫 컬럼을 사람이 읽기 쉽게 처리
    if "first_3sec" in clean:
        value = clean.split("first_3sec")[-1].strip("_ ")
        return f"첫 3초 {value}" if value else "첫 3초 구성"
    if "motion_graphic" in clean:
        value = clean.split("motion_graphic")[-1].strip("_ ")
        return f"그래픽 {value}" if value else "그래픽 활용"
    if "video_format" in clean:
        value = clean.split("video_format")[-1].strip("_ ")
        return f"{value} 포맷" if value else "영상 포맷"
    if "editing_pace" in clean:
        value = clean.split("editing_pace")[-1].strip("_ ")
        return f"{value} 편집 흐름" if value else "편집 속도"

    clean = clean.replace("_", " ")
    return clean


def describe_feature(label: str, domain: str) -> str:
    if "사람" in label or "인물" in label:
        return "장면 이해도 상승"
    if "얼굴" in label or "표정" in label:
        return "감정과 반응 전달"
    if "텍스트" in label or "문구" in label:
        return "핵심 메시지 빠른 전달"
    if "첫 3초" in label:
        return "초반 이탈 방지"
    if "그래픽" in label:
        return "정보 이해도 상승"
    if "포맷" in label or "웹예능" in label or "제품" in label or "기술" in label or "튜토리얼" in label:
        return "콘텐츠 흐름 명확화"
    if "밝기" in label:
        return "화면 가독성 개선"
    if "색" in label or "색감" in label:
        return "시각적 주목도 영향"
    if "편집" in label:
        return "전개 속도 조절"
    if "완성도" in label or "조명" in label:
        return "브랜드 신뢰감 영향"

    if domain == "FnB":
        return "제품 경험 전달"
    return "정보 전달 보조"


def infer_feature_direction(label: str, domain: str) -> str:
    """해석 요약 카드용 상승/하강 방향 추정."""
    down_keywords = ["너무 긴", "과도", "길이", "거의 없음", "기타", "어두", "복잡"]
    if any(keyword in label for keyword in down_keywords):
        return "down"

    # IT에서는 인물 자체보다 텍스트/그래픽이 핵심이라 과한 인물 중심은 주의로 볼 수 있음
    if domain == "IT" and ("사람 등장" in label or "얼굴" in label) and not ("첫 3초" in label):
        return "down"

    # FnB에서는 텍스트 과다는 음식/경험 장면을 가릴 수 있음
    if domain == "FnB" and "텍스트 비율" in label:
        return "down"

    return "up"


def get_importance_data(domain: str) -> list[dict]:
    shap_df, _ = load_shap_data()
    if not shap_df.empty and "domain" in shap_df.columns:
        temp = shap_df[shap_df["domain"] == domain].copy()
        if not temp.empty:
            label_col = None
            for col in ["original_feature", "feature", "label"]:
                if col in temp.columns:
                    label_col = col
                    break

            score_col = None
            for col in ["mean_abs_shap", "importance", "score"]:
                if col in temp.columns:
                    score_col = col
                    break

            if label_col and score_col:
                temp = temp.sort_values(score_col, ascending=False).head(8)
                items = []
                for _, row in temp.iterrows():
                    label = pretty_feature_name(str(row[label_col]))
                    items.append(
                        {
                            "label": label,
                            "score": float(row[score_col]),
                            "direction": infer_feature_direction(label, domain),
                            "desc": describe_feature(label, domain),
                        }
                    )
                return items

    return FNB_IMPORTANCE_DEFAULT if domain == "FnB" else IT_IMPORTANCE_DEFAULT


def get_video_title(row: pd.Series) -> str:
    for col in ["title", "video_title", "제목"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return "제목 없음"


def get_channel_name(row: pd.Series) -> str:
    for col in ["채널명", "channel_title", "channel_name"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return "-"


def get_video_url(row: pd.Series) -> str:
    for col in ["final_url", "url", "video_url"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    video_id = str(row.get("video_id", "")).strip()
    if video_id and video_id.lower() != "nan":
        return f"https://www.youtube.com/shorts/{video_id}"
    return "#"


def get_thumbnail_url(row: pd.Series) -> str:
    thumb = str(row.get("thumbnail", "")).strip()
    if thumb.startswith("http"):
        return thumb
    video_id = str(row.get("video_id", "")).strip()
    if video_id and video_id.lower() != "nan":
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    return ""


def choose_representative_videos(df: pd.DataFrame, domain: str, n: int = 3) -> pd.DataFrame:
    if df.empty or "domain" not in df.columns or "success_label" not in df.columns:
        return pd.DataFrame()

    temp = df[(df["domain"] == domain) & (df["success_label"] == "success")].copy()
    if temp.empty:
        return pd.DataFrame()

    for col in ["person_ratio", "face_ratio", "text_ratio"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").fillna(0)

    if domain == "FnB":
        temp["rep_score"] = (
            temp.get("person_ratio", 0) * 0.35
            + temp.get("face_ratio", 0) * 0.35
            + temp.get("first_3sec", "").astype(str).eq("인물").astype(int) * 0.15
            + temp.get("motion_graphic", "").astype(str).eq("보조적").astype(int) * 0.10
            + temp.get("video_format", "").astype(str).isin(["웹예능", "제품리뷰"]).astype(int) * 0.05
        )
    else:
        temp["rep_score"] = (
            temp.get("text_ratio", 0) * 0.25
            + temp.get("motion_graphic", "").astype(str).eq("핵심요소").astype(int) * 0.25
            + temp.get("first_3sec", "").astype(str).eq("텍스트").astype(int) * 0.20
            + temp.get("video_format", "").astype(str).isin(["기술설명", "튜토리얼", "제품리뷰"]).astype(int) * 0.20
            + temp.get("editing_pace", "").astype(str).isin(["빠름", "매우 빠름"]).astype(int) * 0.10
        )

    selected = temp.sort_values("rep_score", ascending=False).head(n)
    return selected


def get_domain_guides(domain: str) -> list[dict]:
    if domain == "FnB":
        return [
            {"title": "첫 3초", "desc": "사람이 제품을 먹거나 사용하는 장면으로 시작해 시선을 잡습니다."},
            {"title": "인물 활용", "desc": "얼굴, 표정, 반응이 보이는 장면을 적극적으로 배치합니다."},
            {"title": "그래픽 활용", "desc": "모션그래픽은 제품명, 반응 포인트, 짧은 문구를 보조하는 수준으로 사용합니다."},
        ]

    return [
        {"title": "첫 3초", "desc": "문제 상황, 기능명, 핵심 혜택을 짧은 텍스트로 먼저 보여줍니다."},
        {"title": "정보 전달", "desc": "복잡한 기능은 자막, 아이콘, 화면 전환으로 쉽게 설명합니다."},
        {"title": "영상 포맷", "desc": "기술설명형, 기능 시연형, 짧은 튜토리얼형 구성을 우선 고려합니다."},
    ]
