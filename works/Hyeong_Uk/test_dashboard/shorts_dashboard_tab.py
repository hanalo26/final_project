import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from shorts_comment_dashboard_tab import render_shorts_comment_analysis
except Exception as _shorts_comment_import_error:
    def render_shorts_comment_analysis():
        st.error("숏츠 댓글 분석 섹션을 불러오지 못했습니다.")
        st.exception(_shorts_comment_import_error)

from shorts_core import (
    DOMAIN_STYLE,
    build_metric_comparison,
    choose_representative_videos,
    get_counts,
    get_distribution_df,
    get_domain_guides,
    get_importance_data,
    get_patterns,
    get_channel_name,
    get_thumbnail_url,
    get_video_title,
    get_video_url,
    load_shorts_data,
)


# ============================================================
# Shorts dashboard tab v1
# 위치 권장: works/Hyeong_Uk/test_dashboard/shorts_dashboard_tab.py
# ============================================================


def _inject_css():
    st.markdown(
        """
        <style>
        .shorts-hero {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
            margin-bottom: 18px;
        }

        .shorts-hero-title {
            color: #111827;
            font-size: 27px;
            font-weight: 950;
            letter-spacing: -0.8px;
            margin-bottom: 8px;
        }

        .shorts-hero-desc {
            color: #4b5563;
            font-size: 15px;
            line-height: 1.75;
            word-break: keep-all;
        }

        .shorts-kpi-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 19px;
            box-shadow: 0 5px 18px rgba(15,23,42,0.055);
            min-height: 120px;
        }

        .shorts-kpi-label {
            color: #6b7280;
            font-size: 13px;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .shorts-kpi-value {
            color: #111827;
            font-size: 30px;
            font-weight: 950;
            letter-spacing: -0.6px;
            line-height: 1.15;
        }

        .shorts-kpi-desc {
            color: #6b7280;
            font-size: 12.5px;
            margin-top: 8px;
            line-height: 1.45;
            font-weight: 650;
        }

        .domain-selector-caption {
            color: #6b7280;
            font-size: 12.5px;
            margin-bottom: 6px;
            font-weight: 750;
        }

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
            font-weight: 950;
            border: 1px solid #e5e7eb;
            color: #4b5563;
            background: #ffffff;
            box-shadow: 0 3px 12px rgba(15,23,42,0.04);
            transition: 0.15s ease;
        }

        .domain-selector-btn:hover {
            transform: translateY(-1px);
            background: #f9fafb;
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

        .domain-band {
            border-radius: 20px;
            padding: 20px 22px;
            margin: 20px 0 16px 0;
            color: white;
            box-shadow: 0 8px 24px rgba(15,23,42,0.12);
        }

        .domain-band-title {
            font-size: 23px;
            font-weight: 950;
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
            font-weight: 950;
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

        .plot-title {
            color: #111827;
            font-size: 17px;
            font-weight: 950;
            margin-bottom: 4px;
            letter-spacing: -0.2px;
        }

        .plot-desc {
            color: #6b7280;
            font-size: 12.5px;
            line-height: 1.55;
            margin-bottom: 12px;
        }

        .insight-card {
            position: relative;
            overflow: hidden;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 19px 20px;
            box-shadow: 0 5px 18px rgba(15,23,42,0.055);
            margin-bottom: 16px;
        }

        .insight-card::after {
            content: "";
            position: absolute;
            right: -28px;
            top: -28px;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: var(--insight-glow, rgba(239,35,60,0.10));
        }

        .insight-head {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #111827;
            font-size: 18px;
            font-weight: 950;
            margin-bottom: 14px;
            position: relative;
            z-index: 1;
        }

        .insight-main {
            border-radius: 16px;
            padding: 15px 16px;
            margin-bottom: 12px;
            position: relative;
            z-index: 1;
        }

        .insight-main-title {
            color: #111827;
            font-size: 14px;
            font-weight: 950;
            margin-bottom: 5px;
        }

        .insight-main-desc {
            color: #374151;
            font-size: 13px;
            line-height: 1.65;
            word-break: keep-all;
        }

        .insight-item {
            background: #fbfbfc;
            border: 1px solid #eef2f7;
            border-radius: 15px;
            padding: 13px 14px;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        .insight-title {
            color: #111827;
            font-size: 14px;
            font-weight: 950;
            margin-bottom: 5px;
        }

        .insight-desc {
            color: #4b5563;
            font-size: 13px;
            line-height: 1.6;
            word-break: keep-all;
        }

        .case-card-horizontal {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 10px;
            display: flex;
            gap: 12px;
            align-items: stretch;
            min-height: 198px;
            height: 198px;
            overflow: hidden;
            box-shadow: 0 5px 18px rgba(15,23,42,0.055);
            margin-bottom: 14px;
        }

        .case-thumb-wrap {
            position: relative;
            flex: 0 0 128px;
            width: 128px;
            max-width: 128px;
            height: 178px;
            min-height: 178px;
            align-self: stretch;
            border-radius: 12px;
            overflow: hidden;
            background: #f3f4f6;
        }

        .case-thumb-wrap img.case-thumb,
        .case-thumb {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            max-width: none;
            max-height: none;
            object-fit: cover;
            object-position: center center;
            border-radius: 12px;
            display: block;
        }

        .case-domain-badge {
            position: absolute;
            top: 8px;
            left: 8px;
            z-index: 2;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 900;
            line-height: 1;
            background: #ffffff;
            border: 1px solid #e5e7eb;
        }

        .case-domain-badge.fnb { color: #ef233c; }
        .case-domain-badge.it { color: #2563eb; }

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
            font-weight: 800;
        }

        .case-body {
            flex: 1;
            min-width: 0;
            height: 178px;
            min-height: 178px;
            display: flex;
            flex-direction: column;
            padding: 4px 2px 2px 2px;
            overflow: hidden;
        }

        .case-title {
            color: #111827;
            font-size: 13.5px;
            font-weight: 950;
            line-height: 1.38;
            height: 40px;
            max-height: 40px;
            overflow: hidden;
            margin-bottom: 7px;
            word-break: keep-all;
        }

        .case-channel {
            color: #6b7280;
            font-size: 12px;
            margin-bottom: 9px;
            font-weight: 750;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .case-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-height: 56px;
            overflow: hidden;
            margin-bottom: 8px;
        }

        .case-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 5px 8px;
            font-size: 11px;
            font-weight: 850;
            line-height: 1.1;
            border: 1px solid;
        }

        .case-link-row {
            margin-top: auto;
            text-align: right;
            padding-top: 8px;
        }

        .case-link {
            color: #2563eb !important;
            font-size: 12.5px;
            font-weight: 950;
            text-decoration: none !important;
        }

        .guide-box {
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
            font-weight: 950;
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
            font-weight: 950;
            margin-bottom: 6px;
        }

        .guide-step-desc {
            color: #4b5563;
            font-size: 12.5px;
            line-height: 1.6;
            word-break: keep-all;
        }

        b, strong {
            font-weight: 950;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header():
    st.markdown(
        """
        <div class="shorts-hero">
            <div class="shorts-hero-title">숏츠 분석 결과</div>
            <div class="shorts-hero-desc">
                성공한 기업 숏츠에서 반복적으로 나타나는 <b>첫 3초 구성, 인물·얼굴 비율, 텍스트 비율, 모션그래픽, 영상 포맷</b>을 도메인별로 비교합니다.
                분석 결과는 숏츠 agent와 최종 가이드라인의 기준으로 활용됩니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi(label: str, value: str, desc: str, color: str = "#111827"):
    st.markdown(
        f"""
        <div class="shorts-kpi-card">
            <div class="shorts-kpi-label">{label}</div>
            <div class="shorts-kpi-value" style="color:{color};">{value}</div>
            <div class="shorts-kpi-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pct_from_ratio(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def _pct(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "-"


def _short_text(text, max_len=36):
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def _domain_selector() -> str:
    query_domain = str(st.query_params.get("shorts_domain", "FnB"))
    domain = "IT" if query_domain.upper() == "IT" else "FnB"

    fnb_active = "active-fnb" if domain == "FnB" else ""
    it_active = "active-it" if domain == "IT" else ""

    st.markdown(
        f"""
        <div class="domain-selector-caption">도메인 선택</div>
        <div class="domain-selector-wrap">
            <a href="?page=shorts&shorts_domain=FnB" target="_self">
                <div class="domain-selector-btn {fnb_active}">🍴 FnB</div>
            </a>
            <a href="?page=shorts&shorts_domain=IT" target="_self">
                <div class="domain-selector-btn {it_active}">💻 IT</div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return domain


def _domain_band(domain: str, pattern: dict):
    style = DOMAIN_STYLE[domain]
    st.markdown(
        f"""
        <div class="domain-band" style="background:{style['gradient']};">
            <div class="domain-band-title">{style['emoji']} {domain} 성공 숏츠 기준</div>
            <div class="domain-band-desc">{pattern.get("summary", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _distribution_chart(pattern: dict, key: str, label: str, color: str, title: str, desc: str):
    with st.container(border=True):
        st.markdown(f'<div class="plot-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plot-desc">{desc}</div>', unsafe_allow_html=True)

        dist_df = get_distribution_df(pattern, key, label)
        if dist_df.empty:
            st.info("표시할 데이터가 없습니다.")
            return

        fig = px.bar(
            dist_df,
            x=label,
            y="비율",
            text="비율",
            color=label,
            color_discrete_sequence=[color, "#f59e0b", "#8b5cf6", "#94a3b8", "#10b981"],
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            height=340,
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


def _render_insights(domain: str, style: dict):
    guides = get_domain_guides(domain)

    if domain == "FnB":
        main_title = "사람의 반응이 곧 후킹 요소"
        main_desc = "FnB 숏츠는 제품을 설명하기보다 <b>사람이 먹고, 쓰고, 반응하는 장면</b>을 빠르게 보여줄 때 이해와 몰입이 쉬워집니다."
    else:
        main_title = "핵심 메시지를 먼저 보여주기"
        main_desc = "IT 숏츠는 분위기보다 <b>문제, 기능, 혜택</b>을 초반에 바로 보여줘야 짧은 시간 안에 가치를 전달할 수 있습니다."

    st.markdown(
        f"""
        <div class="insight-card" style="--insight-glow:{style['light']};">
            <div class="insight-head">
                <span>{style['emoji']}</span>
                <span>{domain} 숏츠 운영 인사이트</span>
            </div>
            <div class="insight-main" style="background:{style['light']};border:1px solid {style['border']};">
                <div class="insight-main-title">{main_title}</div>
                <div class="insight-main-desc">{main_desc}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    for item in guides:
        st.markdown(
            f"""
            <div class="insight-item" style="border-left:4px solid {style['main']};">
                <div class="insight-title">{item["title"]}</div>
                <div class="insight-desc">{item["desc"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _format_duration(seconds):
    try:
        if pd.isna(seconds):
            return ""
        total = int(float(seconds))
        return f"{total // 60}:{total % 60:02d}"
    except Exception:
        return ""


def _render_case_card(row, style: dict):
    domain = str(row.get("domain", "FnB")).strip()
    domain_class = "fnb" if domain == "FnB" else "it"

    title = _short_text(get_video_title(row), 34)
    channel = _short_text(get_channel_name(row), 20)
    url = get_video_url(row)
    thumb = get_thumbnail_url(row)

    first_3sec = row.get("first_3sec", "-")
    motion = row.get("motion_graphic", "-")
    video_format = row.get("video_format", "-")

    duration = ""
    for duration_col in ["영상길이(초)", "duration", "duration_sec", "duration_seconds"]:
        if duration_col in row.index:
            duration = _format_duration(row.get(duration_col))
            if duration:
                break

    thumb_html = f'<img src="{thumb}" class="case-thumb">' if thumb else '<div class="case-thumb"></div>'
    duration_html = f'<div class="case-duration">{duration}</div>' if duration else ""

    st.markdown(
        f"""
        <div class="case-card-horizontal">
            <div class="case-thumb-wrap">
                <div class="case-domain-badge {domain_class}">{domain}</div>
                {thumb_html}
                {duration_html}
            </div>
            <div class="case-body">
                <div>
                    <div class="case-title">{title}</div>
                    <div class="case-channel">{channel}</div>
                    <div class="case-chip-wrap">
                        <span class="case-chip" style="background:{style['light']};color:{style['main']};border-color:{style['border']};">첫 3초: {first_3sec}</span>
                        <span class="case-chip" style="background:#f5f3ff;color:#7c3aed;border-color:#ddd6fe;">그래픽: {motion}</span>
                        <span class="case-chip" style="background:#f9fafb;color:#374151;border-color:#e5e7eb;">포맷: {video_format}</span>
                    </div>
                </div>
                <div class="case-link-row">
                    <a href="{url}" target="_blank" class="case-link">영상 보기 ↗</a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shorts_dashboard():
    _inject_css()
    _render_header()

    df, _source_path = load_shorts_data()
    fnb_pattern, it_pattern = get_patterns(df)
    counts = get_counts(df)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi("분석 숏츠 수", f"{counts['total']:,}개" if counts["total"] else "-", "전체 분석 표본", "#111827")
    with k2:
        _kpi("성공 숏츠 수", f"{counts['success']:,}개" if counts["total"] else "-", "참여율 기준 성공 라벨", "#ef233c")
    with k3:
        _kpi("핵심 분석 축", "첫 3초 · 인물 · 텍스트", "짧은 영상 구성 요소 중심", "#2563eb")
    with k4:
        _kpi("활용 목적", "운영 가이드라인", "숏츠 제작 방향 제안", "#f97316")

    st.markdown('<div class="section-title">도메인별 성공 숏츠 기준</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">도메인을 선택하면 해당 산업에서 성공 숏츠에 자주 나타난 영상 구성 특징을 확인할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    domain = _domain_selector()
    style = DOMAIN_STYLE[domain]
    pattern = fnb_pattern if domain == "FnB" else it_pattern

    _domain_band(domain, pattern)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("인물 비율", _pct_from_ratio(pattern.get("person_ratio", 0)), "영상 안에 사람이 등장한 정도", style["main"])
    with c2:
        _kpi("얼굴 비율", _pct_from_ratio(pattern.get("face_ratio", 0)), "표정과 반응이 보이는 정도", "#f97316")
    with c3:
        _kpi("텍스트 비율", _pct_from_ratio(pattern.get("text_ratio", 0)), "자막·문구가 차지하는 정도", "#2563eb")
    with c4:
        _kpi("평균 밝기", f"{pattern.get('avg_brightness', 0):.1f}", "영상 전체 밝기 평균", "#059669")

    st.markdown('<div class="section-title">핵심 영상 구성 지표</div>', unsafe_allow_html=True)

    left, right = st.columns([1.12, 0.88])

    with left:
        with st.container(border=True):
            st.markdown('<div class="plot-title">성공/실패 평균 비교</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="plot-desc"><b>인물·얼굴·텍스트 비율</b>이 성공 영상과 실패 영상에서 어떻게 다른지 비교합니다.</div>',
                unsafe_allow_html=True,
            )
            comp_df = build_metric_comparison(df, domain)
            comp_long = comp_df.melt(id_vars="지표", var_name="구분", value_name="값")
            fig = px.bar(
                comp_long,
                x="지표",
                y="값",
                color="구분",
                barmode="group",
                text="값",
                color_discrete_map={"성공": style["main"], "실패": "#94a3b8"},
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(
                height=410,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="평균 비율",
                xaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=13, color="#374151"),
                legend_title_text="",
            )
            fig.update_yaxes(gridcolor="#e5e7eb")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        _render_insights(domain, style)

    st.markdown('<div class="section-title">구성 요소 분포</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">성공 숏츠에서 초반 장면, 모션그래픽, 영상 포맷이 어떻게 나타났는지 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        _distribution_chart(pattern, "first_3sec", "첫 3초", style["main"], "첫 3초 구성", "영상 시작 부분에서 무엇이 먼저 등장하는지 보여줍니다.")
    with d2:
        _distribution_chart(pattern, "motion_graphic", "모션그래픽", "#7c3aed", "모션그래픽 활용", "자막, 아이콘, 화면 전환 등 그래픽 활용 정도입니다.")
    with d3:
        _distribution_chart(pattern, "video_format", "영상 포맷", style["accent"], "영상 포맷", "성공 숏츠에서 자주 사용된 콘텐츠 형식입니다.")

    st.markdown('<div class="section-title">대표 성공 숏츠 사례</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">선택한 도메인의 성공 패턴을 잘 보여주는 대표 영상을 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    reps = choose_representative_videos(df, domain, n=3)
    if reps.empty:
        st.info("대표 영상 사례를 표시하려면 숏츠 분석 결과 CSV에 video_id, title, success_label, domain 컬럼이 필요합니다.")
    else:
        case_cols = st.columns(3, gap="medium")
        for idx, (_, row) in enumerate(reps.iterrows()):
            with case_cols[idx % 3]:
                _render_case_card(row, style)

    st.markdown('<div class="section-title">성공에 영향을 준 요소</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">원핫 처리된 중요도 파일을 우선 사용하고, 읽는 사람이 이해하기 쉬운 이름과 해석으로 정리했습니다.</div>',
        unsafe_allow_html=True,
    )

    importance = pd.DataFrame(get_importance_data(domain)).head(8)

    s1, s2 = st.columns([1.05, 0.95])
    with s1:
        with st.container(border=True):
            st.markdown('<div class="plot-title">중요 요소 TOP 8</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="plot-desc">막대가 길수록 성공 여부를 설명하는 데 더 중요했던 요소입니다. 숫자는 <b>중요도 점수</b>이며, 서로 비교할 때 의미가 있습니다.</div>',
                unsafe_allow_html=True,
            )
            fig = px.bar(
                importance.sort_values("score", ascending=True),
                x="score",
                y="label",
                orientation="h",
                text="score",
                color_discrete_sequence=[style["main"]],
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
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
            st.markdown('<div class="plot-desc">상위 요소별 실무적 의미를 정리했습니다.</div>', unsafe_allow_html=True)

            for _, row in importance.head(6).iterrows():
                direction = str(row.get("direction", "up"))
                if direction == "down":
                    arrow = "↓"
                    arrow_color = "#7c3aed"
                    bg = "#f5f3ff"
                    bd = "#ddd6fe"
                else:
                    arrow = "↑"
                    arrow_color = "#16a34a"
                    bg = "#f0fdf4"
                    bd = "#bbf7d0"

                st.markdown(
                    f"""
                    <div class="insight-item" style="background:{bg};border-color:{bd};">
                        <div class="insight-title">
                            <span style="color:{arrow_color};font-weight:950;">{arrow}</span>
                            {row["label"]}
                        </div>
                        <div class="insight-desc">{row["desc"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    render_shorts_comment_analysis()

    guides = get_domain_guides(domain)

    st.markdown(
        f"""
        <div class="guide-box" style="background:radial-gradient(circle at 92% 18%, {style['light']} 0%, rgba(255,255,255,0) 31%), linear-gradient(135deg, #ffffff 0%, {style['light']} 100%); border:1px solid {style['border']};">
            <div class="guide-title-row">
                <div class="guide-icon" style="background:{style['gradient']};">🎬</div>
                <div>
                    <div class="guide-title">숏츠 운영 가이드라인 연결</div>
                    <div class="guide-subtitle">
                        분석 결과를 <b>숏츠 agent 진단</b>과 <b>최종 가이드라인</b>에서 바로 활용할 수 있도록 정리했습니다.
                    </div>
                </div>
            </div>
            <div class="guide-grid">
                <div class="guide-step" style="border-color:{style['border']};">
                    <div class="guide-step-title">1. {guides[0]["title"]}</div>
                    <div class="guide-step-desc">{guides[0]["desc"]}</div>
                </div>
                <div class="guide-step" style="border-color:{style['border']};">
                    <div class="guide-step-title">2. {guides[1]["title"]}</div>
                    <div class="guide-step-desc">{guides[1]["desc"]}</div>
                </div>
                <div class="guide-step" style="border-color:{style['border']};">
                    <div class="guide-step-title">3. {guides[2]["title"]}</div>
                    <div class="guide-step-desc">{guides[2]["desc"]}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
