from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

THUMBNAIL_DATA_CANDIDATES = [
    DATA_DIR / "all_thumbnail.csv",
    BASE_DIR / "all_thumbnail.csv",
    Path("works/Hyeong_Uk/test_dashboard/data/all_thumbnail.csv"),
    Path("works/Hyeong_Uk/test_dashboard/all_thumbnail.csv"),
    Path("works/Hyeong_Uk/thumbnail_analysis/all_thumbnail.csv"),
    Path("works/Hyeong_Uk/thumbnail_strategy/all_thumbnail.csv"),
    Path("all_thumbnail.csv"),
]


FNB_DEFAULT = {
    "has_person": 91.8,
    "has_text": 97.9,
    "brand": 84.9,
    "brightness": 146.8,
    "saturation": 89.7,
    "contrast": 70.0,
    "visual_hook": 2.61,
    "design_quality": 4.14,
    "text_len": 46.4,
    "category": {
        "예능/콘텐츠형": 40.4,
        "정보 전달형": 31.5,
        "인터뷰/인물형": 10.3,
        "브랜드 이미지형": 8.2,
        "제품 홍보형": 5.5,
        "리뷰/비교형": 2.7,
    },
    "color_tone": {"neutral": 39.0, "warm": 31.5, "cool": 29.5},
    "text_size": {"large": 73.3, "medium": 24.0},
    "person_cat": {"2명+": 60.3, "1명": 31.5, "0명": 8.2},
}

IT_DEFAULT = {
    "has_person": 80.7,
    "has_text": 98.1,
    "brand": 79.1,
    "brightness": 141.5,
    "saturation": 83.6,
    "contrast": 68.6,
    "visual_hook": 2.30,
    "design_quality": 3.95,
    "text_len": 42.6,
    "category": {
        "정보 전달형": 43.7,
        "예능/콘텐츠형": 29.0,
        "인터뷰/인물형": 13.1,
        "브랜드 이미지형": 9.7,
        "리뷰/비교형": 1.9,
        "제품 홍보형": 1.3,
    },
    "color_tone": {"neutral": 45.0, "cool": 42.4, "warm": 12.6},
    "text_size": {"large": 69.4, "medium": 26.8},
    "person_cat": {"2명+": 50.7, "1명": 30.0, "0명": 19.3},
}


SHAP_IT = [
    {"label": "텍스트 크기", "shap": 0.0739, "direction": "up", "desc": "텍스트가 클수록 성공률 상승"},
    {"label": "썸네일 밝기", "shap": 0.0527, "direction": "up", "desc": "밝은 썸네일이 성과에 긍정적"},
    {"label": "파란색 강도", "shap": 0.0522, "direction": "up", "desc": "IT 도메인의 신뢰감·전문성 전달"},
    {"label": "텍스트 길이", "shap": 0.0329, "direction": "down", "desc": "텍스트가 너무 길면 성과 하락"},
    {"label": "등장 인물 수", "shap": 0.0259, "direction": "up", "desc": "인물 등장이 클릭 유도에 기여"},
    {"label": "초록색 강도", "shap": 0.0191, "direction": "up", "desc": "포인트 색상으로 활용 가능"},
    {"label": "시각적 후킹", "shap": 0.0157, "direction": "up", "desc": "강한 후킹 요소가 클릭 유도"},
    {"label": "영어 텍스트 사용", "shap": 0.0129, "direction": "up", "desc": "전문성 인식 제고"},
]

SHAP_FNB = [
    {"label": "붉은색 강도", "shap": 1.4154, "direction": "up", "desc": "식욕·감성 자극"},
    {"label": "초록색 강도", "shap": 1.3168, "direction": "up", "desc": "신선함·자연스러움 전달"},
    {"label": "파란색 강도", "shap": 1.1142, "direction": "up", "desc": "색상 대비 강화"},
    {"label": "썸네일 밝기", "shap": 0.9633, "direction": "up", "desc": "밝고 선명한 음식 사진"},
    {"label": "텍스트 길이", "shap": 0.6908, "direction": "down", "desc": "텍스트가 길면 음식 이미지가 가려짐"},
    {"label": "색 채도", "shap": 0.6069, "direction": "up", "desc": "선명한 음식 색감"},
    {"label": "등장 인물 수", "shap": 0.5125, "direction": "up", "desc": "친근감·신뢰감 강화"},
    {"label": "명암 대비", "shap": 0.5049, "direction": "up", "desc": "음식 질감 강조"},
]


def find_existing_file(candidates: list[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_thumbnail_data() -> tuple[pd.DataFrame, Optional[Path]]:
    file_path = find_existing_file(THUMBNAIL_DATA_CANDIDATES)
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

    if "grade" in df.columns:
        df["grade"] = df["grade"].astype(str)
    elif "success_label" in df.columns:
        df["grade"] = (
            df["success_label"].astype(str).str.lower()
            .replace({"success": "성공", "fail": "실패", "성공": "성공", "실패": "실패"})
        )

    numeric_cols = [
        "has_person", "has_text", "brand_name_visible",
        "brightness_mean", "saturation_mean", "contrast_std",
        "visual_hook_level", "design_quality_level", "text_len",
        "avg_red", "avg_green", "avg_blue", "person_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, file_path


def _dist(series: pd.Series) -> dict:
    if series.dropna().empty:
        return {}
    return (series.astype(str).value_counts(normalize=True) * 100).round(1).to_dict()


def compute_benchmark_from_df(df: pd.DataFrame, domain: str) -> dict:
    if df.empty or "domain" not in df.columns or "grade" not in df.columns:
        return {}

    success_df = df[(df["domain"] == domain) & (df["grade"] == "성공")].copy()
    fail_df = df[(df["domain"] == domain) & (df["grade"] == "실패")].copy()
    if success_df.empty:
        return {}

    def mean_percent(col: str) -> float:
        if col not in success_df.columns:
            return 0.0
        return round(success_df[col].astype(float).mean() * 100, 1)

    def mean_value(col: str) -> float:
        if col not in success_df.columns:
            return 0.0
        return round(success_df[col].mean(), 2)

    return {
        "has_person": mean_percent("has_person"),
        "has_text": mean_percent("has_text"),
        "brand": mean_percent("brand_name_visible"),
        "brightness": mean_value("brightness_mean"),
        "saturation": mean_value("saturation_mean"),
        "contrast": mean_value("contrast_std"),
        "visual_hook": mean_value("visual_hook_level"),
        "design_quality": mean_value("design_quality_level"),
        "text_len": mean_value("text_len"),
        "category": _dist(success_df["thumbnail_category"]) if "thumbnail_category" in success_df.columns else {},
        "color_tone": _dist(success_df["color_tone"]) if "color_tone" in success_df.columns else {},
        "text_size": _dist(success_df["text_size_level"]) if "text_size_level" in success_df.columns else {},
        "person_cat": _dist(success_df["person_cat"]) if "person_cat" in success_df.columns else {},
        "_n_success": len(success_df),
        "_n_fail": len(fail_df),
        "_n_total": len(df[df["domain"] == domain]),
        "_from_csv": True,
    }


def load_benchmarks() -> tuple[dict, dict, pd.DataFrame, Optional[Path]]:
    df, path = load_thumbnail_data()

    fnb = FNB_DEFAULT.copy()
    it = IT_DEFAULT.copy()

    if not df.empty:
        csv_fnb = compute_benchmark_from_df(df, "FnB")
        csv_it = compute_benchmark_from_df(df, "IT")
        if csv_fnb:
            fnb.update(csv_fnb)
        if csv_it:
            it.update(csv_it)

    return fnb, it, df, path


def get_shap_data(domain: str) -> list[dict]:
    return SHAP_FNB if domain == "FnB" else SHAP_IT


def get_domain_strategy(domain: str, bench: dict) -> list[dict]:
    if domain == "FnB":
        return [
            {"title": "색감 우선", "desc": "붉고 선명한 색감을 활용해 음식의 식감과 주목도를 강화합니다."},
            {"title": "인물 중심 구도", "desc": f"성공 썸네일의 인물 등장률은 약 {bench.get('has_person', 0)}%입니다. 제품 경험 장면을 적극 활용하세요."},
            {"title": "간결한 텍스트", "desc": "텍스트는 핵심 키워드 중심으로 줄이고 음식 이미지가 가려지지 않게 구성합니다."},
            {"title": "브랜드 노출", "desc": f"브랜드 로고 또는 브랜드명을 약 {bench.get('brand', 0)}% 수준으로 명확히 노출하는 전략이 유리합니다."},
        ]

    return [
        {"title": "텍스트 크기 최우선", "desc": "핵심 키워드를 크게 배치해 한눈에 주제를 이해할 수 있게 만듭니다."},
        {"title": "밝고 단순한 배경", "desc": f"성공 썸네일의 평균 밝기는 약 {bench.get('brightness', 0)}입니다. 어두운 배경보다 선명한 구성이 유리합니다."},
        {"title": "전문성 있는 색감", "desc": "Cool/Neutral 톤과 파란색 계열을 활용해 신뢰감과 기술적 이미지를 강화합니다."},
        {"title": "정보 전달형 구성", "desc": "서비스 구조, 기능 흐름, 비교 포인트가 바로 보이도록 구성합니다."},
    ]


def get_overall_counts(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0, "success": 0, "fail": 0, "fnb": 0, "it": 0}

    return {
        "total": len(df),
        "success": int((df.get("grade", pd.Series(dtype=str)) == "성공").sum()),
        "fail": int((df.get("grade", pd.Series(dtype=str)) == "실패").sum()),
        "fnb": int((df.get("domain", pd.Series(dtype=str)) == "FnB").sum()),
        "it": int((df.get("domain", pd.Series(dtype=str)) == "IT").sum()),
    }
