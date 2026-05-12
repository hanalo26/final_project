from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================
# 경로 / 파일명
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REQUIRED_FILES = {
    "train": "train_data(longform_comment).csv",
    "test": "test_data(longform_comment).csv",
    "weights": "train_weights(longform_comment).csv",
    "model": "longform_comment_model.joblib",
    "encoding": "longform_comment_encoding_map.joblib",
}


# =========================
# 표시명 / 색상
# =========================
COL_RED = "#ef233c"
COL_BLUE = "#2563eb"
COL_GREEN = "#16a34a"
COL_ORANGE = "#f97316"
COL_PURPLE = "#7c3aed"
COL_GRAY = "#9ca3af"

FEATURE_LABELS = {
    "domain": "도메인",
    "영상길이(초)": "영상 길이",
    "has_paid_product_placement": "유료 광고 포함",
    "channel_tier": "채널 규모",
    "upload_month": "업로드 월",
    "upload_dayofweek": "업로드 요일",
    "upload_hour": "업로드 시간",
    "upload_quarter": "업로드 분기",
    "description_length": "설명 길이",
    "tags_count": "태그 수",
    "cls_content_type": "콘텐츠 유형",
    "cls_marketing_purpose": "마케팅 목적",
    "cls_cta_type": "CTA 유형",
    "cls_is_series": "시리즈 여부",
    "cls_is_collaboration": "콜라보 여부",
    "target_feature": "댓글 긍정 비율",
}


def _file_path(filename: str) -> Path:
    """data 폴더 우선, 없으면 현재 모듈 폴더에서 탐색."""
    p1 = DATA_DIR / filename
    if p1.exists():
        return p1
    return BASE_DIR / filename


@st.cache_data(show_spinner=False)
def _load_comment_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(_file_path(REQUIRED_FILES["train"]))
    test = pd.read_csv(_file_path(REQUIRED_FILES["test"]))
    weights = pd.read_csv(_file_path(REQUIRED_FILES["weights"]))
    return train, test, weights


@st.cache_resource(show_spinner=False)
def _load_comment_assets() -> tuple[Any, dict]:
    model = joblib.load(_file_path(REQUIRED_FILES["model"]))
    enc_map = joblib.load(_file_path(REQUIRED_FILES["encoding"]))
    return model, enc_map


def _check_files() -> list[str]:
    missing = []
    for filename in REQUIRED_FILES.values():
        if not _file_path(filename).exists():
            missing.append(filename)
    return missing


def _inject_comment_style():
    st.markdown(
        """
        <style>
        .comment-hero {
            background: linear-gradient(135deg, #ffffff 0%, #fff7f8 100%);
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 4px 16px rgba(15,23,42,0.04);
            margin-bottom: 16px;
        }
        .comment-hero-title {
            font-size: 22px;
            font-weight: 950;
            color: #111827;
            margin-bottom: 6px;
            letter-spacing: -0.4px;
        }
        .comment-hero-desc {
            font-size: 13px;
            color: #6b7280;
            line-height: 1.7;
            word-break: keep-all;
        }
        .comment-metric {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15,23,42,0.04);
            border-left: 4px solid var(--accent);
            min-height: 118px;
        }
        .comment-metric-label {
            font-size: 12px;
            color: #6b7280;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .comment-metric-value {
            font-size: 26px;
            font-weight: 950;
            color: #111827;
            line-height: 1.15;
        }
        .comment-metric-sub {
            font-size: 12px;
            color: #6b7280;
            margin-top: 6px;
            line-height: 1.5;
        }
        .comment-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15,23,42,0.04);
            margin-bottom: 14px;
        }
        .comment-card-title {
            font-size: 16px;
            font-weight: 950;
            color: #111827;
            margin-bottom: 5px;
        }
        .comment-card-desc {
            font-size: 12.5px;
            color: #6b7280;
            line-height: 1.6;
            margin-bottom: 10px;
            word-break: keep-all;
        }
        .comment-insight {
            background: #fff1f2;
            border-left: 4px solid #ef233c;
            border-radius: 0 12px 12px 0;
            padding: 12px 14px;
            font-size: 13px;
            color: #374151;
            line-height: 1.7;
            margin-top: 10px;
            word-break: keep-all;
        }
        .comment-insight-blue {
            background: #eff6ff;
            border-left-color: #2563eb;
            color: #1e3a8a;
        }
        .comment-insight-green {
            background: #f0fdf4;
            border-left-color: #16a34a;
            color: #14532d;
        }
        .comment-chip {
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 11.5px;
            font-weight: 850;
            margin: 2px 4px 2px 0;
            border: 1px solid #e5e7eb;
            background: #f9fafb;
            color: #374151;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{digits}f}%"


def _decode_value(col: str, value: Any, enc_map: dict) -> str:
    if col not in enc_map:
        if col in ["cls_is_series", "cls_is_collaboration", "has_paid_product_placement"]:
            return "예" if int(round(float(value))) == 1 else "아니오"
        if col == "domain":
            return str(value)
        return str(value)

    try:
        key = int(round(float(value)))
    except Exception:
        key = value

    return str(enc_map[col].get(key, value))


def _decode_df(df: pd.DataFrame, enc_map: dict) -> pd.DataFrame:
    out = df.copy()
    for col in enc_map.keys():
        if col in out.columns:
            out[col + "_label"] = out[col].apply(lambda x, c=col: _decode_value(c, x, enc_map))
    return out


def _domain_key(label: str) -> str | None:
    if "FnB" in label:
        return "FnB"
    if "IT" in label:
        return "IT"
    return None


def _plotly_layout(height: int = 320) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans KR", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
    )


def _metric_card(label: str, value: str, sub: str = "", color: str = COL_RED):
    st.markdown(
        f"""
        <div class="comment-metric" style="--accent:{color};">
            <div class="comment-metric-label">{label}</div>
            <div class="comment-metric-value">{value}</div>
            <div class="comment-metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_importance_df(model, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    else:
        vals = np.zeros(len(feature_cols), dtype=float)

    vals = vals[: len(feature_cols)]
    df = pd.DataFrame({"feature": feature_cols[: len(vals)], "importance": vals})
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    return df.sort_values("importance", ascending=False)


def _avg_by_category(df: pd.DataFrame, col: str, enc_map: dict, min_count: int = 3) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["label", "mean", "count"])

    g = (
        df.groupby(col, as_index=False)
        .agg(mean=("target_feature", "mean"), count=("target_feature", "size"))
        .sort_values("mean", ascending=False)
    )
    g["label"] = g[col].apply(lambda x: _decode_value(col, x, enc_map))
    return g[g["count"] >= min_count].copy()


def _make_sample_from_inputs(model, enc_map: dict, train: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, dict]:
    feature_cols = [c for c in train.columns if c != "target_feature"]
    base = {}

    # 기본값: 수치형 중앙값, 범주형 최빈값
    for col in feature_cols:
        s = train[col]
        if pd.api.types.is_numeric_dtype(s):
            base[col] = float(s.median())
        else:
            mode = s.mode()
            base[col] = mode.iloc[0] if len(mode) else s.iloc[0]

    inv = {
        col: {v: k for k, v in mapping.items()}
        for col, mapping in enc_map.items()
    }

    # 입력 UI
    domain_code = inv.get("domain", {}).get(domain, 0)
    content_label = st.selectbox(
        "콘텐츠 유형",
        list(inv.get("cls_content_type", {"브이로그": 3}).keys()),
        key=f"comment_sim_content_{domain}",
    )
    cta_label = st.selectbox(
        "CTA 유형",
        list(inv.get("cls_cta_type", {"기타": 2}).keys()),
        key=f"comment_sim_cta_{domain}",
    )
    purpose_label = st.selectbox(
        "마케팅 목적",
        list(inv.get("cls_marketing_purpose", {"브랜드캠페인": 3}).keys()),
        key=f"comment_sim_purpose_{domain}",
    )
    day_label = st.selectbox(
        "업로드 요일",
        list(inv.get("upload_dayofweek", {"금요일": 0}).keys()),
        key=f"comment_sim_day_{domain}",
    )
    upload_hour = st.slider("업로드 시간", 0, 23, 18, key=f"comment_sim_hour_{domain}")
    video_len = st.slider("영상 길이(초)", 30, 3600, 600, step=30, key=f"comment_sim_len_{domain}")
    desc_len = st.slider("설명 길이", 0, 3000, 500, step=50, key=f"comment_sim_desc_{domain}")
    tags = st.slider("태그 수", 0, 50, 5, key=f"comment_sim_tags_{domain}")
    is_series = st.toggle("시리즈 영상", value=False, key=f"comment_sim_series_{domain}")
    is_collab = st.toggle("콜라보 영상", value=False, key=f"comment_sim_collab_{domain}")

    overrides = {
        "domain": domain_code,
        "cls_content_type": inv.get("cls_content_type", {}).get(content_label, base.get("cls_content_type", 0)),
        "cls_cta_type": inv.get("cls_cta_type", {}).get(cta_label, base.get("cls_cta_type", 0)),
        "cls_marketing_purpose": inv.get("cls_marketing_purpose", {}).get(purpose_label, base.get("cls_marketing_purpose", 0)),
        "upload_dayofweek": inv.get("upload_dayofweek", {}).get(day_label, base.get("upload_dayofweek", 0)),
        "upload_hour": upload_hour,
        "영상길이(초)": video_len,
        "description_length": desc_len,
        "tags_count": tags,
        "cls_is_series": int(is_series),
        "cls_is_collaboration": int(is_collab),
    }

    for col, val in overrides.items():
        if col in base:
            base[col] = val

    sample = pd.DataFrame([base])[feature_cols]

    # train dtype과 최대한 맞춤
    for col in sample.columns:
        try:
            sample[col] = sample[col].astype(train[col].dtype)
        except Exception:
            pass

    user_labels = {
        "콘텐츠 유형": content_label,
        "CTA 유형": cta_label,
        "마케팅 목적": purpose_label,
        "업로드 요일": day_label,
        "업로드 시간": f"{upload_hour}시",
        "영상 길이": f"{video_len:,}초",
        "설명 길이": f"{desc_len:,}자",
        "태그 수": f"{tags}개",
    }

    return sample, user_labels


def page_longform_comment_analysis():
    """롱폼 분석 결과 탭 안에 들어가는 댓글 반응 분석 섹션."""
    _inject_comment_style()

    st.markdown(
        """
        <div class="comment-hero">
            <div class="comment-hero-title">💬 롱폼 댓글 반응 분석</div>
            <div class="comment-hero-desc">
                롱폼 영상의 메타데이터와 콘텐츠 분류 정보를 바탕으로 <b>댓글 긍정 비율</b>을 분석합니다.
                도메인별 댓글 반응 차이와 긍정 반응을 높이는 요소를 확인하고, 간단한 예측 시뮬레이터로 운영 전략을 점검할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = _check_files()
    if missing:
        st.error("롱폼 댓글 분석에 필요한 파일을 찾지 못했습니다.")
        st.caption("아래 파일들을 test_dashboard/data 폴더 또는 test_dashboard 폴더에 넣어주세요.")
        st.code("\n".join(missing))
        return

    try:
        train, test, weights = _load_comment_csvs()
        model, enc_map = _load_comment_assets()
    except Exception as e:
        st.error("롱폼 댓글 분석 파일 로드 중 오류가 발생했습니다.")
        st.exception(e)
        return

    all_df = pd.concat(
        [train.assign(split="train"), test.assign(split="test")],
        ignore_index=True,
    )
    all_dec = _decode_df(all_df, enc_map)

    domain_choice = st.segmented_control(
        "도메인 선택",
        ["전체", "FnB", "IT"],
        default="전체",
        key="longform_comment_domain_select",
    )
    dom_key = _domain_key(domain_choice)

    if dom_key:
        view_df = all_dec[all_dec["domain_label"] == dom_key].copy()
    else:
        view_df = all_dec.copy()

    if view_df.empty:
        st.warning("선택한 도메인에 해당하는 댓글 분석 데이터가 없습니다.")
        return

    feature_cols = [c for c in train.columns if c != "target_feature"]
    imp_df = _feature_importance_df(model, feature_cols)
    top_feature = imp_df.iloc[0]["label"] if not imp_df.empty else "-"

    avg_pos = view_df["target_feature"].mean()
    high_ratio = (view_df["target_feature"] >= 0.7).mean()
    low_ratio = (view_df["target_feature"] < 0.4).mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _metric_card("분석 영상 수", f"{len(view_df):,}개", "train/test 데이터 통합 기준", COL_RED)
    with m2:
        _metric_card("평균 댓글 긍정 비율", _fmt_pct(avg_pos), "target_feature 평균", COL_GREEN)
    with m3:
        _metric_card("긍정 반응 높은 영상", _fmt_pct(high_ratio), "댓글 긍정 비율 70% 이상", COL_BLUE)
    with m4:
        _metric_card("가장 중요한 변수", top_feature, "모델 feature importance 기준", COL_ORANGE)

    st.markdown("---")

    # 상단 차트
    col1, col2 = st.columns([1.2, 1.2], gap="large")

    with col1:
        st.markdown(
            """
            <div class="comment-card">
                <div class="comment-card-title">도메인별 댓글 긍정 비율</div>
                <div class="comment-card-desc">FnB와 IT의 평균 댓글 긍정 비율을 비교합니다.</div>
            """,
            unsafe_allow_html=True,
        )

        dom_avg = (
            all_dec.groupby("domain_label", as_index=False)
            .agg(mean=("target_feature", "mean"), count=("target_feature", "size"))
            .sort_values("mean", ascending=False)
        )
        colors = [COL_RED if d == "FnB" else COL_BLUE for d in dom_avg["domain_label"]]
        fig = go.Figure(go.Bar(
            x=dom_avg["domain_label"],
            y=dom_avg["mean"] * 100,
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in dom_avg["mean"]],
            textposition="outside",
            customdata=dom_avg["count"],
            hovertemplate="도메인: %{x}<br>긍정 비율: %{y:.1f}%<br>영상 수: %{customdata}개<extra></extra>",
        ))
        fig.update_layout(**_plotly_layout(300), yaxis_title="댓글 긍정 비율", yaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="comment-card">
                <div class="comment-card-title">중요 변수 TOP 8</div>
                <div class="comment-card-desc">댓글 긍정 비율 예측에 크게 작용한 변수를 확인합니다.</div>
            """,
            unsafe_allow_html=True,
        )

        top = imp_df.head(8).sort_values("importance", ascending=True)
        fig = go.Figure(go.Bar(
            x=top["importance"],
            y=top["label"],
            orientation="h",
            marker_color=COL_RED if dom_key != "IT" else COL_BLUE,
            text=[f"{v:.3f}" for v in top["importance"]],
            textposition="outside",
        ))
        fig.update_layout(**_plotly_layout(300), xaxis_title="중요도", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 카테고리별 분석
    st.markdown("### 댓글 반응에 영향을 주는 콘텐츠 조건")
    c1, c2, c3 = st.columns(3, gap="large")

    analysis_items = [
        (c1, "cls_content_type", "콘텐츠 유형별 댓글 긍정 비율"),
        (c2, "cls_cta_type", "CTA 유형별 댓글 긍정 비율"),
        (c3, "upload_dayofweek", "업로드 요일별 댓글 긍정 비율"),
    ]

    for col_box, feat, title in analysis_items:
        with col_box:
            g = _avg_by_category(view_df, feat, enc_map, min_count=2).head(8)
            st.markdown(
                f"""
                <div class="comment-card">
                    <div class="comment-card-title">{title}</div>
                    <div class="comment-card-desc">영상 수가 2개 이상인 항목만 표시합니다.</div>
                """,
                unsafe_allow_html=True,
            )

            if g.empty:
                st.info("표시할 데이터가 부족합니다.")
            else:
                fig = go.Figure(go.Bar(
                    x=g["mean"] * 100,
                    y=g["label"],
                    orientation="h",
                    marker_color=COL_GREEN,
                    text=[f"{v*100:.1f}%" for v in g["mean"]],
                    textposition="outside",
                    customdata=g["count"],
                    hovertemplate="%{y}<br>긍정 비율: %{x:.1f}%<br>영상 수: %{customdata}개<extra></extra>",
                ))
                fig.update_layout(
                    **_plotly_layout(310),
                    xaxis_title="댓글 긍정 비율",
                    xaxis_ticksuffix="%",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # 인사이트
    best_domain = dom_avg.iloc[0]["domain_label"] if not dom_avg.empty else "-"
    best_content = _avg_by_category(view_df, "cls_content_type", enc_map, min_count=2).head(1)
    best_cta = _avg_by_category(view_df, "cls_cta_type", enc_map, min_count=2).head(1)

    best_content_txt = best_content.iloc[0]["label"] if not best_content.empty else "-"
    best_cta_txt = best_cta.iloc[0]["label"] if not best_cta.empty else "-"

    st.markdown(
        f"""
        <div class="comment-insight">
            <b>해석 요약</b><br>
            선택 범위 기준 평균 댓글 긍정 비율은 <b>{_fmt_pct(avg_pos)}</b>입니다.
            현재 데이터에서는 <b>{best_content_txt}</b> 유형과 <b>{best_cta_txt}</b> CTA가 댓글 긍정 반응이 높은 조건으로 나타났습니다.
            중요도 점수는 모델이 댓글 긍정 비율을 예측할 때 참고한 상대적 기준이므로, 같은 그래프 안에서 순위 중심으로 해석하는 것이 좋습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # content_type × marketing_purpose 결합 인사이트
    st.markdown("---")
    st.markdown("### 콘텐츠 유형 × 마케팅 목적 결합 분석")
    st.caption("댓글 감정은 콘텐츠 유형 하나만이 아니라, 어떤 마케팅 목적과 결합되는지에 따라 달라질 수 있습니다.")

    combo_df = view_df.copy()
    if "cls_content_type_label" not in combo_df.columns and "cls_content_type" in combo_df.columns:
        combo_df["cls_content_type_label"] = combo_df["cls_content_type"].apply(
            lambda x: _decode_value("cls_content_type", x, enc_map)
        )
    if "cls_marketing_purpose_label" not in combo_df.columns and "cls_marketing_purpose" in combo_df.columns:
        combo_df["cls_marketing_purpose_label"] = combo_df["cls_marketing_purpose"].apply(
            lambda x: _decode_value("cls_marketing_purpose", x, enc_map)
        )

    if {"cls_content_type_label", "cls_marketing_purpose_label", "target_feature"}.issubset(combo_df.columns):
        combo_summary = (
            combo_df
            .groupby(["cls_content_type_label", "cls_marketing_purpose_label"], as_index=False)
            .agg(
                mean=("target_feature", "mean"),
                count=("target_feature", "size"),
            )
        )

        min_combo_count = st.slider(
            "결합 분석 최소 영상 수",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="longform_comment_combo_min_count",
            help="영상 수가 너무 적은 조합은 우연일 수 있어 제외할 수 있습니다.",
        )

        combo_valid = combo_summary[combo_summary["count"] >= min_combo_count].copy()

        if combo_valid.empty:
            st.info("선택한 조건에서 표시할 결합 분석 데이터가 부족합니다.")
        else:
            top_combos = combo_valid.sort_values("mean", ascending=False).head(5)
            low_combos = combo_valid.sort_values("mean", ascending=True).head(5)

            # 1) 히트맵을 상단 전체 폭으로 배치
            st.markdown(
                """
                <div class="comment-card">
                    <div class="comment-card-title">결합 조건별 댓글 긍정 비율</div>
                    <div class="comment-card-desc">
                        콘텐츠 유형과 마케팅 목적을 함께 봤을 때 댓글 긍정 비율이 어떻게 달라지는지 확인합니다.
                        각 칸의 숫자는 댓글 긍정 비율과 해당 조합의 영상 수입니다.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            top_content_labels = (
                combo_valid
                .groupby("cls_content_type_label")["count"]
                .sum()
                .sort_values(ascending=False)
                .head(8)
                .index
                .tolist()
            )
            top_purpose_labels = (
                combo_valid
                .groupby("cls_marketing_purpose_label")["count"]
                .sum()
                .sort_values(ascending=False)
                .head(8)
                .index
                .tolist()
            )

            heat_src = combo_valid[
                combo_valid["cls_content_type_label"].isin(top_content_labels)
                & combo_valid["cls_marketing_purpose_label"].isin(top_purpose_labels)
            ].copy()

            pivot_mean = heat_src.pivot_table(
                index="cls_content_type_label",
                columns="cls_marketing_purpose_label",
                values="mean",
                aggfunc="mean",
            ).reindex(index=top_content_labels, columns=top_purpose_labels)

            pivot_count = heat_src.pivot_table(
                index="cls_content_type_label",
                columns="cls_marketing_purpose_label",
                values="count",
                aggfunc="sum",
            ).reindex(index=top_content_labels, columns=top_purpose_labels)

            z = pivot_mean.fillna(0).values * 100
            text_matrix = []
            for r_idx in range(pivot_mean.shape[0]):
                row_text = []
                for c_idx in range(pivot_mean.shape[1]):
                    mean_v = pivot_mean.iloc[r_idx, c_idx]
                    cnt_v = pivot_count.iloc[r_idx, c_idx]
                    if pd.isna(mean_v):
                        row_text.append("")
                    else:
                        row_text.append(f"{mean_v*100:.0f}%<br>n={int(cnt_v)}")
                text_matrix.append(row_text)

            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=pivot_mean.columns.tolist(),
                    y=pivot_mean.index.tolist(),
                    colorscale=[
                        [0.0, "#fee2e2"],
                        [0.5, "#fef3c7"],
                        [1.0, "#dcfce7"],
                    ],
                    zmin=0,
                    zmax=100,
                    text=text_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    colorbar=dict(title="긍정 비율"),
                    hovertemplate=(
                        "콘텐츠 유형: %{y}<br>"
                        "마케팅 목적: %{x}<br>"
                        "댓글 긍정 비율: %{z:.1f}%<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                **{
                    **_plotly_layout(430),
                    "margin": dict(l=10, r=10, t=20, b=80),
                },
                xaxis_title="마케팅 목적",
                yaxis_title="콘텐츠 유형",
                xaxis=dict(tickangle=-20),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 2) 조합 카드를 하단 2열로 배치
            combo_col1, combo_col2 = st.columns(2, gap="large")

            with combo_col1:
                st.markdown(
                    """
                    <div class="comment-card">
                        <div class="comment-card-title">댓글 반응이 좋은 조합</div>
                        <div class="comment-card-desc">
                            댓글 긍정 비율이 높게 나타난 콘텐츠 유형과 마케팅 목적 조합입니다.
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                for i, row in enumerate(top_combos.itertuples(index=False), 1):
                    st.markdown(
                        f"""
                        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:13px;
                                    padding:10px 12px;margin-bottom:8px;">
                            <div style="font-size:12px;font-weight:900;color:#166534;margin-bottom:4px;">
                                {i}. {row.cls_content_type_label} × {row.cls_marketing_purpose_label}
                            </div>
                            <div style="font-size:12px;color:#14532d;line-height:1.5;">
                                댓글 긍정 비율 <b>{row.mean*100:.1f}%</b> · 영상 {int(row.count)}개
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            with combo_col2:
                st.markdown(
                    """
                    <div class="comment-card">
                        <div class="comment-card-title">주의가 필요한 조합</div>
                        <div class="comment-card-desc">
                            댓글 긍정 비율이 낮게 나타난 조합입니다. 콘텐츠 목적과 형식의 조정이 필요할 수 있습니다.
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                for i, row in enumerate(low_combos.itertuples(index=False), 1):
                    st.markdown(
                        f"""
                        <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:13px;
                                    padding:10px 12px;margin-bottom:8px;">
                            <div style="font-size:12px;font-weight:900;color:#9a3412;margin-bottom:4px;">
                                {i}. {row.cls_content_type_label} × {row.cls_marketing_purpose_label}
                            </div>
                            <div style="font-size:12px;color:#7c2d12;line-height:1.5;">
                                댓글 긍정 비율 <b>{row.mean*100:.1f}%</b> · 영상 {int(row.count)}개
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="comment-insight comment-insight-blue">
                    <b>핵심 인사이트</b><br>
                    IT와 FnB 모두에서 <b>콘텐츠 유형 × 마케팅 목적</b>의 효과크기가 크게 나타났습니다.
                    즉, 롱폼 영상의 댓글 감정은 콘텐츠 유형 하나만으로 결정되기보다,
                    해당 콘텐츠가 <b>어떤 마케팅 목적과 결합되는지</b>에 따라 크게 달라질 수 있습니다.
                    따라서 기업 유튜브 롱폼 전략은 콘텐츠 유형을 단독으로 선택하기보다,
                    <b>목적별로 적합한 콘텐츠 유형을 매칭</b>하는 방식으로 설계하는 것이 좋습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("content_type × marketing_purpose 결합 분석에 필요한 컬럼이 없습니다.")
