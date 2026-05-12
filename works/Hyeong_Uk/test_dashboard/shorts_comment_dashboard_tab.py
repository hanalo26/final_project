from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

COMMENT_FILES = {
    ("FnB", "성공"): "sentiment_filtered_cleaned_fnb_shorts_success_comment.csv",
    ("FnB", "실패"): "sentiment_filtered_cleaned_fnb_shorts_fail_comment.csv",
    ("IT", "성공"): "sentiment_filtered_cleaned_it_shorts_success_comment.csv",
    ("IT", "실패"): "sentiment_filtered_cleaned_it_shorts_fail_comment.csv",
}

SENTIMENT_ORDER = ["긍정", "중립", "부정"]
SENTIMENT_COLORS = {
    "긍정": "#16a34a",
    "중립": "#9ca3af",
    "부정": "#ef4444",
}
DOMAIN_COLORS = {
    "FnB": "#ef233c",
    "IT": "#2563eb",
}


def _file_path(filename: str) -> Path:
    p1 = DATA_DIR / filename
    if p1.exists():
        return p1
    return BASE_DIR / filename


@st.cache_data(show_spinner=False)
def _load_shorts_comment_data() -> pd.DataFrame:
    frames = []

    for (domain, grade), filename in COMMENT_FILES.items():
        path = _file_path(filename)
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["domain"] = domain
        df["grade"] = grade
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "comment_id",
                "video_id",
                "sentiment",
                "is_korean",
                "is_event",
                "reason",
                "domain",
                "grade",
            ]
        )

    out = pd.concat(frames, ignore_index=True)

    for col in ["is_korean", "is_event"]:
        if col in out.columns:
            out[col] = out[col].astype(bool)

    return out


def _clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "is_korean" in out.columns:
        out = out[out["is_korean"] == True]

    if "is_event" in out.columns:
        out = out[out["is_event"] == False]

    return out.reset_index(drop=True)


def _fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{digits}f}%"


def _inject_compact_css():
    st.markdown(
        """
        <style>
        .shorts-comment-compact-card {
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:18px;
            padding:18px 20px;
            box-shadow:0 4px 16px rgba(15,23,42,.045);
            margin-bottom:14px;
        }

        .shorts-comment-mini-note {
            background:#fffbeb;
            border:1px solid #fde68a;
            border-left:4px solid #f59e0b;
            border-radius:13px;
            padding:10px 12px;
            color:#78350f;
            font-size:12.5px;
            line-height:1.65;
            word-break:keep-all;
            margin-bottom:12px;
        }

        .shorts-comment-mini-title {
            font-size:16px;
            font-weight:950;
            color:#111827;
            margin-bottom:5px;
        }

        .shorts-comment-mini-desc {
            font-size:12.5px;
            color:#6b7280;
            line-height:1.6;
            word-break:keep-all;
            margin-bottom:10px;
        }

        .shorts-comment-kpi-mini {
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-left:4px solid var(--accent);
            border-radius:15px;
            padding:13px 14px;
            box-shadow:0 4px 14px rgba(15,23,42,.04);
            min-height:96px;
            margin-bottom:12px;
        }

        .shorts-comment-kpi-mini-label {
            font-size:11.5px;
            color:#6b7280;
            font-weight:850;
            margin-bottom:7px;
        }

        .shorts-comment-kpi-mini-value {
            font-size:23px;
            font-weight:950;
            color:#111827;
            line-height:1.12;
        }

        .shorts-comment-kpi-mini-sub {
            font-size:11.5px;
            color:#6b7280;
            margin-top:6px;
            line-height:1.45;
        }

        .shorts-comment-summary-box {
            background:var(--bg);
            border:1px solid var(--border);
            border-left:4px solid var(--accent);
            border-radius:14px;
            padding:12px 13px;
            font-size:13px;
            line-height:1.65;
            color:var(--text);
            word-break:keep-all;
            min-height:136px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mini_kpi(label: str, value: str, sub: str, color: str):
    st.markdown(
        f"""
        <div class="shorts-comment-kpi-mini" style="--accent:{color};">
            <div class="shorts-comment-kpi-mini-label">{label}</div>
            <div class="shorts-comment-kpi-mini-value">{value}</div>
            <div class="shorts-comment-kpi-mini-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _success_sentiment_rates(df: pd.DataFrame) -> pd.DataFrame:
    success = df[df["grade"] == "성공"].copy()

    if success.empty:
        return pd.DataFrame(columns=["domain", "sentiment", "count", "ratio"])

    g = (
        success.groupby(["domain", "sentiment"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    total = g.groupby("domain")["count"].transform("sum")
    g["ratio"] = g["count"] / total

    return g


def _plot_success_sentiment_compact(rates: pd.DataFrame):
    domains = ["FnB", "IT"]
    fig = go.Figure()

    for sentiment in SENTIMENT_ORDER:
        vals = []
        texts = []
        counts = []

        for domain in domains:
            row = rates[(rates["domain"] == domain) & (rates["sentiment"] == sentiment)]

            if row.empty:
                vals.append(0)
                texts.append("0.0%")
                counts.append(0)
            else:
                ratio = float(row.iloc[0]["ratio"])
                cnt = int(row.iloc[0]["count"])
                vals.append(ratio * 100)
                texts.append(f"{ratio*100:.1f}%")
                counts.append(cnt)

        fig.add_trace(
            go.Bar(
                x=domains,
                y=vals,
                name=sentiment,
                marker_color=SENTIMENT_COLORS[sentiment],
                text=texts,
                textposition="inside",
                customdata=counts,
                hovertemplate=(
                    "도메인: %{x}<br>"
                    "감성: " + sentiment + "<br>"
                    "비율: %{y:.1f}%<br>"
                    "댓글 수: %{customdata:,}개<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="댓글 비율",
        xaxis_title="",
        yaxis_ticksuffix="%",
        legend_title_text="",
        font=dict(size=13, color="#374151"),
    )
    fig.update_yaxes(gridcolor="#e5e7eb", range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)



def _positive_factor_data(domain: str) -> pd.DataFrame:
    """분석 정리문에서 대시보드용으로 선별한 대표 긍정 반응 요소."""
    rows = {
        "FnB": [
            {"요소": "보통 편집 × 비비드", "긍정비율": 96.77, "설명": "일반적인 숏츠 속도와 선명한 색감 조합"},
            {"요소": "빠른 편집 × 따뜻함", "긍정비율": 95.55, "설명": "음식·공간 분위기와 역동적 편집 조합"},
            {"요소": "따뜻함 × 실내", "긍정비율": 95.53, "설명": "붉고 노란 계열 색감과 실내/매장 배경"},
            {"요소": "시설소개", "긍정비율": 92.80, "설명": "공간과 분위기를 직접 보여주는 포맷"},
            {"요소": "광고/CF × 인물", "긍정비율": 86.90, "설명": "제품 메시지와 인물 반응을 함께 제시"},
        ],
        "IT": [
            {"요소": "웹드라마", "긍정비율": 94.90, "설명": "기술을 상황·스토리로 전달하는 포맷"},
            {"요소": "웹드라마 × 텍스트", "긍정비율": 94.74, "설명": "드라마형 도입부에서 자막·타이틀 활용"},
            {"요소": "광고/CF × 인물", "긍정비율": 94.51, "설명": "서비스 메시지를 인물 중심으로 제시"},
            {"요소": "첫 3초 인물", "긍정비율": 89.20, "설명": "초반에 사람이 등장해 이해와 몰입을 유도"},
            {"요소": "광고/CF", "긍정비율": 78.60, "설명": "짧고 명확한 브랜드 메시지 전달"},
        ],
    }
    return pd.DataFrame(rows.get(domain, []))


def _plot_positive_factor_bars(domain: str, color: str):
    factor_df = _positive_factor_data(domain)

    if factor_df.empty:
        st.info("표시할 대표 제작 요소가 없습니다.")
        return

    chart_df = factor_df.sort_values("긍정비율", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=chart_df["긍정비율"],
            y=chart_df["요소"],
            orientation="h",
            marker_color=color,
            text=[f"{v:.1f}%" for v in chart_df["긍정비율"]],
            textposition="outside",
            customdata=chart_df["설명"],
            hovertemplate=(
                "요소: %{y}<br>"
                "긍정 댓글 비율: %{x:.1f}%<br>"
                "%{customdata}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=285,
        margin=dict(l=10, r=30, t=8, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="긍정 댓글 비율",
        yaxis_title="",
        xaxis_ticksuffix="%",
        font=dict(size=12, color="#374151"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", range=[0, 105])

    st.plotly_chart(fig, use_container_width=True)


def _render_factor_table(domain: str):
    factor_df = _positive_factor_data(domain)

    if factor_df.empty:
        return

    for row in factor_df.head(3).itertuples(index=False):
        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                        padding:9px 11px;margin-bottom:7px;">
                <div style="font-size:12.5px;font-weight:950;color:#111827;margin-bottom:3px;">
                    {row.요소} <span style="color:#16a34a;">{row.긍정비율:.1f}%</span>
                </div>
                <div style="font-size:11.8px;color:#6b7280;line-height:1.45;word-break:keep-all;">
                    {row.설명}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_shorts_comment_analysis():
    """숏츠 분석 결과 탭 중간에 들어가는 간략 댓글 반응 보조 분석."""
    _inject_compact_css()

    raw_df = _load_shorts_comment_data()

    if raw_df.empty:
        st.warning("숏츠 댓글 분석 CSV 파일을 찾지 못했습니다.")
        return

    clean_df = _clean_comments(raw_df)
    success_df = clean_df[clean_df["grade"] == "성공"].copy()
    fail_df = clean_df[clean_df["grade"] == "실패"].copy()

    fnb_success = success_df[success_df["domain"] == "FnB"]
    it_success = success_df[success_df["domain"] == "IT"]

    fnb_pos = (fnb_success["sentiment"] == "긍정").mean() if not fnb_success.empty else float("nan")
    it_pos = (it_success["sentiment"] == "긍정").mean() if not it_success.empty else float("nan")
    success_share = len(success_df) / len(clean_df) if len(clean_df) else float("nan")

    st.markdown('<div class="section-title">댓글 반응 보조 분석</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">성공 숏츠 댓글의 감성 분포를 보조 근거로 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _mini_kpi("분석 댓글 수", f"{len(clean_df):,}개", "비한국어·이벤트 댓글 제외", "#111827")
    with k2:
        _mini_kpi("성공 댓글 비중", _fmt_pct(success_share), f"실패 댓글 {len(fail_df):,}개", "#f97316")
    with k3:
        _mini_kpi("FnB 성공 긍정", _fmt_pct(fnb_pos), f"성공 댓글 {len(fnb_success):,}개 기준", DOMAIN_COLORS["FnB"])
    with k4:
        _mini_kpi("IT 성공 긍정", _fmt_pct(it_pos), f"성공 댓글 {len(it_success):,}개 기준", DOMAIN_COLORS["IT"])

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown(
            """
            <div class="shorts-comment-compact-card">
                <div class="shorts-comment-mini-title">도메인별 성공 숏츠 댓글 감성 비율</div>
                <div class="shorts-comment-mini-desc">
                    성공 숏츠 댓글만 기준으로 긍정·중립·부정 비중을 비교합니다.
                </div>
            """,
            unsafe_allow_html=True,
        )
        _plot_success_sentiment_compact(_success_sentiment_rates(clean_df))
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        fnb_negneu = 1 - fnb_pos if not pd.isna(fnb_pos) else float("nan")
        it_negneu = 1 - it_pos if not pd.isna(it_pos) else float("nan")
        pos_gap = fnb_pos - it_pos if not pd.isna(fnb_pos) and not pd.isna(it_pos) else float("nan")

        st.markdown(
            f"""
            <div class="shorts-comment-summary-box"
                 style="--bg:#fff1f2;--border:#fecdd3;--accent:#ef233c;--text:#881337;">
                <b>FnB 댓글 반응</b><br>
                성공 숏츠 댓글 중 <b>{_fmt_pct(fnb_pos)}</b>가 긍정 반응입니다.
                전체 감성 분포만 보면 FnB는 댓글 반응이 긍정 중심으로 강하게 형성되는 편입니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="shorts-comment-summary-box"
                 style="--bg:#eff6ff;--border:#bfdbfe;--accent:#2563eb;--text:#1e3a8a;margin-top:10px;">
                <b>IT 댓글 반응</b><br>
                IT 성공 숏츠의 긍정 비율은 <b>{_fmt_pct(it_pos)}</b>이고,
                부정·중립 비중은 <b>{_fmt_pct(it_negneu)}</b>입니다.
                긍정이 가장 많지만, FnB보다 의견 차이나 해석 반응이 함께 나타나는 구조입니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="shorts-comment-summary-box"
                 style="--bg:#f0fdf4;--border:#bbf7d0;--accent:#16a34a;--text:#14532d;margin-top:10px;">
                <b>운영 활용</b><br>
                FnB와 IT의 긍정 비율 차이는 <b>{'-' if pd.isna(pos_gap) else f'{pos_gap*100:+.1f}%p'}</b>입니다.
                세부 제작 방향은 아래 <b>긍정 반응 제작 요소</b> 그래프와 함께 해석합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-caption" style="margin-top:8px;margin-bottom:8px;">아래 그래프는 댓글 분석 정리문에서 긍정 비율이 높게 나타난 제작 요소만 선별한 것입니다. 오른쪽 해석 카드의 제작 방향은 이 그래프를 근거로 봅니다.</div>',
        unsafe_allow_html=True,
    )

    factor_left, factor_right = st.columns(2, gap="large")

    with factor_left:
        st.markdown(
            """
            <div class="shorts-comment-compact-card">
                <div class="shorts-comment-mini-title">FnB 긍정 반응 제작 요소</div>
                <div class="shorts-comment-mini-desc">
                    음식·공간·제품 경험이 잘 느껴지는 시각 조합이 긍정 댓글과 연결됩니다.
                </div>
            """,
            unsafe_allow_html=True,
        )
        _plot_positive_factor_bars("FnB", DOMAIN_COLORS["FnB"])
        st.markdown("</div>", unsafe_allow_html=True)

    with factor_right:
        st.markdown(
            """
            <div class="shorts-comment-compact-card">
                <div class="shorts-comment-mini-title">IT 긍정 반응 제작 요소</div>
                <div class="shorts-comment-mini-desc">
                    기술 설명을 그대로 나열하기보다 인물·상황·스토리로 이해시키는 구성이 유리합니다.
                </div>
            """,
            unsafe_allow_html=True,
        )
        _plot_positive_factor_bars("IT", DOMAIN_COLORS["IT"])
        st.markdown("</div>", unsafe_allow_html=True)

    guide1, guide2, guide3 = st.columns(3, gap="large")

    with guide1:
        st.markdown(
            """
            <div class="shorts-comment-summary-box"
                 style="--bg:#fff1f2;--border:#fecdd3;--accent:#ef233c;--text:#881337;">
                <b>FnB 제작 방향</b><br>
                따뜻한 색감, 비비드한 색감, 실내·매장 분위기, 시설소개·광고/CF 포맷을 우선 고려하세요.
                제품 설명보다 맛, 공간, 인물 반응이 먼저 느껴지도록 구성하는 것이 좋습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with guide2:
        st.markdown(
            """
            <div class="shorts-comment-summary-box"
                 style="--bg:#eff6ff;--border:#bfdbfe;--accent:#2563eb;--text:#1e3a8a;">
                <b>IT 제작 방향</b><br>
                웹드라마·광고/CF 포맷, 프로페셔널한 제작 퀄리티, 첫 3초 인물 등장을 활용하세요.
                기술 내용을 상황과 인물을 통해 쉽게 이해시키는 구성이 유리합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with guide3:
        st.markdown(
            """
            <div class="shorts-comment-summary-box"
                 style="--bg:#f0fdf4;--border:#bbf7d0;--accent:#16a34a;--text:#14532d;">
                <b>공통 가이드</b><br>
                댓글 반응은 단일 요소보다 포맷, 첫 3초, 색감, 배경, 편집 속도의 조합에 따라 달라집니다.
                따라서 포맷과 시각 구성을 함께 설계하는 방식이 필요합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
