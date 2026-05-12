import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from thumbnail_core import (
    get_domain_strategy,
    get_overall_counts,
    get_shap_data,
    load_benchmarks,
)


# ============================================================
# Thumbnail analysis dashboard tab v2
# 위치 권장: works/Hyeong_Uk/test_dashboard/thumbnail_dashboard_tab.py
# ============================================================

DOMAIN_STYLE = {
    "FnB": {
        "main": "#ef233c",
        "sub": "#fb7185",
        "light": "#fff1f2",
        "border": "#fecdd3",
        "accent": "#f97316",
        "gradient": "linear-gradient(135deg, #ef233c 0%, #fb7185 100%)",
        "name": "FnB",
        "emoji": "🍴",
    },
    "IT": {
        "main": "#2563eb",
        "sub": "#60a5fa",
        "light": "#eff6ff",
        "border": "#bfdbfe",
        "accent": "#06b6d4",
        "gradient": "linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)",
        "name": "IT",
        "emoji": "💻",
    },
}


def _inject_css():
    st.markdown(
        """
        <style>
        .thumb-hero {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
            margin-bottom: 18px;
        }

        .thumb-hero-title {
            color: #111827;
            font-size: 27px;
            font-weight: 800;
            letter-spacing: -0.8px;
            margin-bottom: 8px;
        }

        .thumb-hero-desc {
            color: #4b5563;
            font-size: 15px;
            line-height: 1.75;
            word-break: keep-all;
        }

        .thumb-kpi-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 19px;
            box-shadow: 0 5px 18px rgba(15,23,42,0.055);
            min-height: 120px;
        }

        .thumb-kpi-label {
            color: #6b7280;
            font-size: 13px;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .thumb-kpi-value {
            color: #111827;
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.6px;
            line-height: 1.15;
        }

        .thumb-kpi-desc {
            color: #6b7280;
            font-size: 12.5px;
            margin-top: 8px;
            line-height: 1.45;
            font-weight: 650;
        }

        .domain-band {
            border-radius: 20px;
            padding: 20px 22px;
            margin: 20px 0 16px 0;
            color: white;
            box-shadow: 0 8px 24px rgba(15,23,42,0.12);
        }

        .domain-band-title {
            font-size: 23px;
            font-weight: 800;
            letter-spacing: -0.6px;
            margin-bottom: 5px;
        }

        .domain-band-desc {
            font-size: 14px;
            line-height: 1.6;
            opacity: 0.94;
            word-break: keep-all;
        }

        .section-title {
            color: #111827;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 26px 0 12px 0;
        }

        .section-caption {
            color: #6b7280;
            font-size: 13.5px;
            line-height: 1.65;
            margin-top: -4px;
            margin-bottom: 12px;
            word-break: keep-all;
        }

        .plot-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 18px 20px 20px 20px;
            box-shadow: 0 5px 18px rgba(15,23,42,0.055);
            margin-bottom: 16px;
        }

        .plot-title {
            color: #111827;
            font-size: 17px;
            font-weight: 800;
            margin-bottom: 4px;
            letter-spacing: -0.2px;
        }

        .plot-desc {
            color: #6b7280;
            font-size: 12.5px;
            line-height: 1.55;
            margin-bottom: 12px;
        }

        .strategy-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 20px 22px;
            box-shadow: 0 5px 18px rgba(15,23,42,0.055);
            margin-bottom: 16px;
        }

        .strategy-head {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #111827;
            font-size: 17px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .strategy-list {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            align-items: stretch;
            width: 100%;
            min-width: 0;
            contain: layout paint;
        }

        .strategy-item {
            background: #fbfbfc;
            border: 1px solid #eef2f7;
            border-radius: 17px;
            padding: 16px 17px;
            height: 126px;
            box-sizing: border-box;
            min-width: 0;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .strategy-title {
            color: #111827;
            font-size: 13.2px;
            font-weight: 800;
            line-height: 1.35;
            letter-spacing: -0.25px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 8px;
            min-height: 18px;
            word-break: keep-all;
        }

        .strategy-desc {
            color: #4b5563;
            font-size: 12.6px;
            line-height: 1.55;
            word-break: keep-all;
            overflow-wrap: normal;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .insight-item {
            border-radius: 15px;
            padding: 13px 15px;
            margin-bottom: 10px;
            border: 1px solid;
        }

        .insight-title {
            font-size: 14px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .insight-desc {
            color: #4b5563;
            font-size: 13px;
            line-height: 1.6;
            word-break: keep-all;
        }

        .guide-box {
            background:
                radial-gradient(circle at 92% 18%, rgba(239,35,60,0.13) 0%, rgba(239,35,60,0.00) 31%),
                linear-gradient(135deg, #ffffff 0%, #fff7f8 100%);
            border: 1px solid #fecdd3;
            border-radius: 22px;
            padding: 22px 24px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.07);
            margin-top: 12px;
            margin-bottom: 8px;
        }

        .guide-title-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }

        .guide-icon {
            width: 46px;
            height: 46px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
            color: #ffffff;
            flex-shrink: 0;
        }

        .guide-title {
            color: #111827;
            font-size: 19px;
            font-weight: 800;
            margin-bottom: 3px;
        }

        .guide-subtitle {
            color: #6b7280;
            font-size: 12.5px;
            line-height: 1.55;
        }

        .guide-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }

        .guide-step {
            background: rgba(255,255,255,0.78);
            border: 1px solid #fee2e2;
            border-radius: 17px;
            padding: 14px 15px;
        }

        .guide-step-title {
            color: #111827;
            font-size: 13.5px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .guide-step-desc {
            color: #4b5563;
            font-size: 12.5px;
            line-height: 1.6;
            word-break: keep-all;
        }

        /* Custom domain selector */
        .domain-selector-wrap {
            display: flex;
            gap: 10px;
            margin: 2px 0 14px 0;
        }

        .domain-selector-wrap a {
            text-decoration: none !important;
        }

        .domain-selector-btn {
            border-radius: 999px;
            padding: 10px 18px;
            font-size: 14px;
            font-weight: 800;
            border: 1px solid #e5e7eb;
            color: #4b5563;
            background: #ffffff;
            box-shadow: 0 3px 12px rgba(15,23,42,0.04);
            transition: 0.15s ease;
        }

        .domain-selector-btn:hover {
            transform: translateY(-1px);
            border-color: #fecdd3;
            background: #fff7f8;
        }

        .domain-selector-btn.active-fnb {
            background: linear-gradient(135deg, #ef233c 0%, #fb7185 100%);
            border-color: #ef233c;
            color: #ffffff;
            box-shadow: 0 8px 20px rgba(239,35,60,0.22);
        }

        .domain-selector-btn.active-it {
            background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
            border-color: #2563eb;
            color: #ffffff;
            box-shadow: 0 8px 20px rgba(37,99,235,0.20);
        }

        .domain-selector-caption {
            color: #6b7280;
            font-size: 12.5px;
            margin-bottom: 6px;
            font-weight: 750;
        }

        b, strong {
            font-weight: 800;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header():
    st.markdown(
        """
        <div class="thumb-hero">
            <div class="thumb-hero-title">썸네일 분석 결과</div>
            <div class="thumb-hero-desc">
                성공 영상의 썸네일에서 반복적으로 나타나는 <b>인물, 텍스트, 색감, 브랜드 노출</b> 요소를 도메인별로 비교합니다.
                <b>FnB와 IT의 성공 패턴</b>을 분리해 신규 기업 채널의 <b>썸네일 제작 기준</b>으로 활용할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi(label: str, value: str, desc: str, color: str = "#111827"):
    st.markdown(
        f"""
        <div class="thumb-kpi-card">
            <div class="thumb-kpi-label">{label}</div>
            <div class="thumb-kpi-value" style="color:{color};">{value}</div>
            <div class="thumb-kpi-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _percent(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "-"


def _num(value, suffix=""):
    try:
        return f"{float(value):.1f}{suffix}"
    except Exception:
        return "-"


def _bench_dataframe(bench: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"지표": "인물 등장률", "값": bench.get("has_person", 0), "구분": "구성 요소"},
            {"지표": "텍스트 삽입률", "값": bench.get("has_text", 0), "구분": "구성 요소"},
            {"지표": "브랜드 노출률", "값": bench.get("brand", 0), "구분": "구성 요소"},
            {"지표": "평균 밝기", "값": bench.get("brightness", 0), "구분": "이미지 톤"},
            {"지표": "평균 채도", "값": bench.get("saturation", 0), "구분": "이미지 톤"},
            {"지표": "평균 대비", "값": bench.get("contrast", 0), "구분": "이미지 톤"},
            {"지표": "시각적 후킹", "값": bench.get("visual_hook", 0) * 20, "구분": "품질 점수"},
            {"지표": "디자인 품질", "값": bench.get("design_quality", 0) * 20, "구분": "품질 점수"},
        ]
    )


def _radar_values(bench: dict) -> dict:
    """레이더 차트용 0~100 스케일 값."""
    person_cat = bench.get("person_cat", {}) or {}
    two_plus = person_cat.get("2명+", person_cat.get("2명 이상", person_cat.get("2인 이상", 0)))

    return {
        "인물 등장": bench.get("has_person", 0),
        "2인 이상": two_plus,
        "브랜드 노출": bench.get("brand", 0),
        "밝기": min(100, bench.get("brightness", 0) / 2.55),
        "채도": min(100, bench.get("saturation", 0)),
        "디자인": min(100, bench.get("design_quality", 0) * 20),
    }


def _render_radar_chart(domain: str, bench: dict):
    """선택한 도메인의 성공 썸네일 기준을 레이더 차트로 표시."""
    style = DOMAIN_STYLE[domain]
    values_dict = _radar_values(bench)

    categories = list(values_dict.keys())
    values = list(values_dict.values())

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    main_color = style["main"]
    fill_color = "rgba(239, 35, 60, 0.18)" if domain == "FnB" else "rgba(37, 99, 235, 0.16)"

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=domain,
            line=dict(color=main_color, width=4),
            fillcolor=fill_color,
            marker=dict(size=7, color=main_color),
        )
    )

    fig.update_layout(
        height=470,
        margin=dict(l=50, r=50, t=30, b=34),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#6b7280"),
                gridcolor="#e5e7eb",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#374151"),
                gridcolor="#e5e7eb",
            ),
            bgcolor="white",
        ),
        showlegend=False,
        paper_bgcolor="white",
        font=dict(color="#374151"),
    )

    st.plotly_chart(fig, use_container_width=True)


def _dist_dataframe(dist: dict, label: str) -> pd.DataFrame:
    if not dist:
        return pd.DataFrame(columns=[label, "비율"])
    return pd.DataFrame([{label: k, "비율": v} for k, v in dist.items()])


def _domain_band(domain: str):
    style = DOMAIN_STYLE[domain]
    if domain == "FnB":
        desc = "FnB는 음식의 감각적 매력과 제품 경험을 빠르게 전달하는 썸네일이 중요합니다."
    else:
        desc = "IT는 기능, 문제 해결, 정보 전달 포인트를 한눈에 이해시키는 썸네일이 중요합니다."

    st.markdown(
        f"""
        <div class="domain-band" style="background:{style['gradient']};">
            <div class="domain-band-title">{style['emoji']} {domain} 성공 썸네일 기준</div>
            <div class="domain-band-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_strategy(domain: str, bench: dict):
    """CSS class 의존을 줄이고, 핵심 카드 스타일은 인라인으로 고정 렌더링."""
    style = DOMAIN_STYLE[domain]
    strategies = get_domain_strategy(domain, bench)

    item_html = ""
    for item in strategies[:4]:
        item_html += (
            f'<div style="'
            f'background:#fbfbfc;'
            f'border:1px solid #eef2f7;'
            f'border-left:5px solid {style["main"]};'
            f'border-radius:17px;'
            f'padding:16px 17px;'
            f'height:126px;'
            f'box-sizing:border-box;'
            f'overflow:hidden;'
            f'display:flex;'
            f'flex-direction:column;'
            f'justify-content:flex-start;'
            f'">'
            f'<div title="{item["title"]}" style="'
            f'color:#111827;'
            f'font-size:13.2px;'
            f'font-weight:800;'
            f'line-height:1.35;'
            f'letter-spacing:-0.25px;'
            f'white-space:nowrap;'
            f'overflow:hidden;'
            f'text-overflow:ellipsis;'
            f'margin-bottom:8px;'
            f'min-height:18px;'
            f'word-break:keep-all;'
            f'">{item["title"]}</div>'
            f'<div style="'
            f'color:#4b5563;'
            f'font-size:12.6px;'
            f'font-weight:400;'
            f'line-height:1.55;'
            f'word-break:keep-all;'
            f'overflow-wrap:normal;'
            f'display:-webkit-box;'
            f'-webkit-line-clamp:3;'
            f'-webkit-box-orient:vertical;'
            f'overflow:hidden;'
            f'">{item["desc"]}</div>'
            f'</div>'
        )

    html = (
        f'<div style="'
        f'background:#ffffff;'
        f'border:1px solid #e5e7eb;'
        f'border-radius:20px;'
        f'padding:20px 22px;'
        f'box-shadow:0 5px 18px rgba(15,23,42,0.055);'
        f'margin-bottom:16px;'
        f'">'
        f'<div style="'
        f'display:flex;'
        f'align-items:center;'
        f'gap:10px;'
        f'color:#111827;'
        f'font-size:17px;'
        f'font-weight:800;'
        f'margin-bottom:12px;'
        f'">'
        f'<span>{style["emoji"]}</span>'
        f'<span>{domain} 썸네일 운영 인사이트</span>'
        f'</div>'
        f'<div style="'
        f'font-size:12.5px;'
        f'font-weight:400;'
        f'color:#6b7280;'
        f'line-height:1.6;'
        f'margin:-4px 0 13px 0;'
        f'word-break:keep-all;'
        f'">'
        f'성공 썸네일에서 반복적으로 나타난 요소를 제작 체크포인트 형태로 정리했습니다.'
        f'</div>'
        f'<div style="'
        f'display:grid;'
        f'grid-template-columns:repeat(4,minmax(0,1fr));'
        f'gap:12px;'
        f'align-items:stretch;'
        f'width:100%;'
        f'min-width:0;'
        f'">'
        f'{item_html}'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

def _plot_card_start(title: str, desc: str = ""):
    st.markdown(
        f"""
        <div class="plot-card">
            <div class="plot-title">{title}</div>
            <div class="plot-desc">{desc}</div>
        """,
        unsafe_allow_html=True,
    )


def _plot_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def render_thumbnail_dashboard():
    _inject_css()
    _render_header()

    fnb, it, df, _source_path = load_benchmarks()
    counts = get_overall_counts(df)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi("분석 썸네일 수", f"{counts['total']:,}개" if counts["total"] else "-", "전체 썸네일 표본", "#111827")
    with k2:
        _kpi("성공 썸네일 수", f"{counts['success']:,}개" if counts["total"] else "-", "성공 라벨 기준", "#ef233c")
    with k3:
        _kpi("분석 도메인", "FnB · IT", "도메인별 성공 기준 비교", "#2563eb")
    with k4:
        _kpi("핵심 요소", "인물 · 텍스트 · 색감", "썸네일 시각 구성 중심", "#f97316")

    st.markdown('<div class="section-title">도메인별 성공 썸네일 기준</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">도메인을 선택하면 해당 산업에서 성공 썸네일에 자주 나타난 <b>시각적 특징</b>을 확인할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    query_domain = str(st.query_params.get("thumb_domain", "FnB"))
    domain = "IT" if query_domain.upper() == "IT" else "FnB"

    fnb_active = "active-fnb" if domain == "FnB" else ""
    it_active = "active-it" if domain == "IT" else ""

    st.markdown(
        f"""
        <div class="domain-selector-caption">도메인 선택</div>
        <div class="domain-selector-wrap">
            <a href="?page=thumbnail&thumb_domain=FnB" target="_self">
                <div class="domain-selector-btn {fnb_active}">🍴 FnB</div>
            </a>
            <a href="?page=thumbnail&thumb_domain=IT" target="_self">
                <div class="domain-selector-btn {it_active}">💻 IT</div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    style = DOMAIN_STYLE[domain]
    bench = fnb if domain == "FnB" else it

    _domain_band(domain)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("인물 등장률", _percent(bench.get("has_person", 0)), "성공 썸네일 기준", style["main"])
    with c2:
        _kpi("텍스트 삽입률", _percent(bench.get("has_text", 0)), "핵심 문구 포함 여부", style["accent"])
    with c3:
        _kpi("브랜드 노출률", _percent(bench.get("brand", 0)), "로고/브랜드명 표시", "#7c3aed")
    with c4:
        _kpi("디자인 품질", _num(bench.get("design_quality", 0), "/5"), "AI/수작업 평가 기반", "#059669")

    st.markdown('<div class="section-title">주요 시각 지표와 운영 인사이트</div>', unsafe_allow_html=True)

    visual_col, radar_col = st.columns([1.28, 1.0], gap="large")

    with visual_col:
        with st.container(border=True):
            st.markdown('<div class="plot-title">주요 시각 지표</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="plot-desc"><b>구성 요소</b>는 비율(%), <b>품질 점수</b>는 100점 환산값으로 표시했습니다.</div>',
                unsafe_allow_html=True,
            )
            bench_df = _bench_dataframe(bench)
            color_map = {
                "구성 요소": style["main"],
                "이미지 톤": style["sub"],
                "품질 점수": style["accent"],
            }
            fig = px.bar(
                bench_df,
                x="지표",
                y="값",
                color="구분",
                text="값",
                color_discrete_map=color_map,
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(
                height=445,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="값",
                xaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=13, color="#374151"),
                legend_title_text="",
            )
            fig.update_yaxes(gridcolor="#e5e7eb")
            st.plotly_chart(fig, use_container_width=True)

    with radar_col:
        with st.container(border=True):
            st.markdown(f'<div class="plot-title">{domain} 썸네일 기준 레이더</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="plot-desc">선택한 도메인의 성공 썸네일 특징을 0~100 기준으로 요약했습니다.</div>',
                unsafe_allow_html=True,
            )
            _render_radar_chart(domain, bench)

    _render_strategy(domain, bench)

    st.markdown('<div class="section-title">구성 요소 분포</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">성공 썸네일의 카테고리, 색상 톤, 인물 구성을 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        with st.container(border=True):
            st.markdown('<div class="plot-title">콘텐츠형 썸네일 카테고리</div>', unsafe_allow_html=True)
            st.markdown('<div class="plot-desc">성공 썸네일의 콘텐츠 유형 분포입니다.</div>', unsafe_allow_html=True)
            cat_df = _dist_dataframe(bench.get("category", {}), "카테고리")
            if cat_df.empty:
                st.info("카테고리 분포 데이터가 없습니다.")
            else:
                fig = px.pie(
                    cat_df,
                    names="카테고리",
                    values="비율",
                    hole=0.52,
                    color_discrete_sequence=[style["main"], style["sub"], style["accent"], "#f59e0b", "#8b5cf6", "#94a3b8"],
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(
                    height=360,
                    margin=dict(l=5, r=5, t=5, b=5),
                    font=dict(size=12, color="#374151"),
                    showlegend=True,
                    legend=dict(font=dict(size=11)),
                )
                st.plotly_chart(fig, use_container_width=True)

    with d2:
        with st.container(border=True):
            st.markdown('<div class="plot-title">색상 톤 분포</div>', unsafe_allow_html=True)
            st.markdown('<div class="plot-desc">성공 썸네일에서 자주 나타난 색상 톤입니다.</div>', unsafe_allow_html=True)
            tone_df = _dist_dataframe(bench.get("color_tone", {}), "색상 톤")
            if tone_df.empty:
                st.info("색상 톤 분포 데이터가 없습니다.")
            else:
                tone_colors = {
                    "warm": "#f97316",
                    "cool": "#2563eb",
                    "neutral": "#64748b",
                }
                fig = px.bar(
                    tone_df,
                    x="색상 톤",
                    y="비율",
                    text="비율",
                    color="색상 톤",
                    color_discrete_map=tone_colors,
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(
                    height=360,
                    margin=dict(l=5, r=5, t=5, b=5),
                    yaxis_title="비율(%)",
                    xaxis_title="",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(size=13, color="#374151"),
                    showlegend=False,
                )
                fig.update_yaxes(gridcolor="#e5e7eb")
                st.plotly_chart(fig, use_container_width=True)

    with d3:
        with st.container(border=True):
            st.markdown('<div class="plot-title">인물 구성 분포</div>', unsafe_allow_html=True)
            st.markdown('<div class="plot-desc">썸네일에 등장하는 인물 수의 분포입니다.</div>', unsafe_allow_html=True)
            person_df = _dist_dataframe(bench.get("person_cat", {}), "인물 구성")
            if person_df.empty:
                st.info("인물 구성 분포 데이터가 없습니다.")
            else:
                fig = px.bar(
                    person_df,
                    x="인물 구성",
                    y="비율",
                    text="비율",
                    color="인물 구성",
                    color_discrete_sequence=[style["main"], style["sub"], "#94a3b8"],
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(
                    height=360,
                    margin=dict(l=5, r=5, t=5, b=5),
                    yaxis_title="비율(%)",
                    xaxis_title="",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(size=13, color="#374151"),
                    showlegend=False,
                )
                fig.update_yaxes(gridcolor="#e5e7eb")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">SHAP 기반 주요 변수</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">성공 여부에 영향을 준 <b>변수 중요도</b>를 요약했습니다. 방향은 해당 변수가 성공률에 미치는 일반적인 해석입니다.</div>',
        unsafe_allow_html=True,
    )

    shap_data = pd.DataFrame(get_shap_data(domain)).head(8)

    s1, s2 = st.columns([1.05, 0.95])

    with s1:
        with st.container(border=True):
            st.markdown('<div class="plot-title">성공 영향 변수 TOP 8</div>', unsafe_allow_html=True)
            st.markdown('<div class="plot-desc">막대가 길수록 성공 분류에 더 큰 영향을 준 변수입니다.</div>', unsafe_allow_html=True)
            shap_plot = shap_data.copy()
            shap_plot["bar_color"] = shap_plot["direction"].map({"down": "#7c3aed"}).fillna(style["main"])
            shap_plot = shap_plot.sort_values("shap", ascending=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=shap_plot["shap"],
                    y=shap_plot["label"],
                    orientation="h",
                    text=[f"{v:.3f}" for v in shap_plot["shap"]],
                    textposition="outside",
                    marker_color=shap_plot["bar_color"],
                    hovertemplate="%{y}<br>중요도: %{x:.4f}<extra></extra>",
                )
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="중요도",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=13, color="#374151"),
                showlegend=False,
            )
            fig.update_xaxes(gridcolor="#e5e7eb")
            st.plotly_chart(fig, use_container_width=True)

    with s2:
        with st.container(border=True):
            st.markdown('<div class="plot-title">해석 요약</div>', unsafe_allow_html=True)
            st.markdown('<div class="plot-desc">상위 변수별 실무적 의미를 정리했습니다.</div>', unsafe_allow_html=True)

            for _, row in shap_data.head(6).iterrows():
                direction_icon = "↑" if row["direction"] == "up" else "↓"
                direction_color = "#16a34a" if row["direction"] == "up" else "#7c3aed"
                bg = "#f0fdf4" if row["direction"] == "up" else "#f5f3ff"
                bd = "#bbf7d0" if row["direction"] == "up" else "#ddd6fe"
                st.markdown(
                    f"""
                    <div class="insight-item" style="background:{bg};border-color:{bd};">
                        <div class="insight-title">
                            <span style="color:{direction_color};font-weight:800;">{direction_icon}</span>
                            {row["label"]}
                        </div>
                        <div class="insight-desc">{row["desc"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"""
        <div class="guide-box"
             style="background:radial-gradient(circle at 92% 18%, {style['light']} 0%, rgba(255,255,255,0) 31%),
                               linear-gradient(135deg, #ffffff 0%, {style['light']} 100%);
                    border:1px solid {style['border']};">
            <div class="guide-title-row">
                <div class="guide-icon" style="background:{style['gradient']};">🎨</div>
                <div>
                    <div class="guide-title">썸네일 제작 가이드라인 연결</div>
                    <div class="guide-subtitle">
                        분석 결과를 <b>agent 진단</b>과 <b>썸네일 제작 탭</b>에서 바로 활용할 수 있도록 정리했습니다.
                    </div>
                </div>
            </div>
            <div class="guide-grid">
                <div class="guide-step" style="border-color:{style['border']};">
                    <div class="guide-step-title">1. 도메인 기준 적용</div>
                    <div class="guide-step-desc">
                        <b>{domain}</b> 성공 썸네일 기준을 제작 전 체크리스트로 활용합니다.
                    </div>
                </div>
                <div class="guide-step" style="border-color:{style['border']};">
                    <div class="guide-step-title">2. 우선 개선 요소 선택</div>
                    <div class="guide-step-desc">
                        <b>SHAP 상위 변수</b>를 기준으로 텍스트, 색감, 인물, 브랜드 노출을 먼저 점검합니다.
                    </div>
                </div>
                <div class="guide-step" style="border-color:{style['border']};">
                    <div class="guide-step-title">3. 브랜드 톤으로 조정</div>
                    <div class="guide-step-desc">
                        도메인 평균을 그대로 복사하기보다 <b>브랜드 톤</b>과 영상 주제에 맞게 조정합니다.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
