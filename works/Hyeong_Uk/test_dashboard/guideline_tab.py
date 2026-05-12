from __future__ import annotations

from datetime import datetime

import streamlit as st


# ============================================================
# guideline_tab.py
# 최종 가이드라인 페이지
#
# 원칙:
# - agent 결과 / saved 폴더 / Gemini 생성 agent 사용하지 않음
# - 분석 결과에서 도출한 운영 가이드라인을 도메인별로 정리
# - 도메인 선택: FnB / IT
# - 체크리스트 제거
# - 가이드라인 저장/다운로드에 포커스
# - 카드 깨짐 방지를 위해 내부 div 폭 고정/overflow 방지
# ============================================================


DOMAIN_DATA = {
    "FnB": {
        "emoji": "🍴",
        "color": "#ef233c",
        "soft": "#fff1f2",
        "soft2": "#fff7f8",
        "border": "#fecdd3",
        "title": "FnB 기업 유튜브 운영 가이드라인",
        "subtitle": "감각적 경험, 인물 반응, 제품 노출을 중심으로 식욕과 방문·구매 욕구를 자극하는 전략입니다.",
        "summary": [
            {
                "title": "롱폼",
                "icon": "▶",
                "source": "롱폼 분석 결과 · 롱폼 댓글 분석",
                "items": [
                    "콘텐츠 유형 하나보다 콘텐츠 유형과 마케팅 목적의 결합을 기준으로 기획합니다.",
                    "제품 경험, 공간 분위기, 인물 반응, 브랜드 친근감이 드러나는 구성이 적합합니다.",
                    "댓글 반응은 다음 기획을 보완하는 근거로 활용합니다.",
                ],
            },
            {
                "title": "숏츠",
                "icon": "⚡",
                "source": "숏츠 분석 결과 · 숏츠 댓글 분석",
                "items": [
                    "첫 3초 안에 음식, 제품, 사람 반응, 상황 중 하나를 명확히 보여줍니다.",
                    "인물 반응과 제품 경험 장면을 활용해 감각적 반응을 유도합니다.",
                    "모션그래픽은 제품명, 상황 설명, 반응 포인트를 보조하는 정도로 활용합니다.",
                ],
            },
            {
                "title": "썸네일",
                "icon": "▧",
                "source": "썸네일 분석 결과",
                "items": [
                    "선명한 색감, 인물, 음식·제품 노출, 짧은 핵심 문구를 우선합니다.",
                    "긴 설명보다 맛, 상황, 궁금증을 압축한 문구가 적합합니다.",
                    "따뜻한 색감과 높은 시각적 대비로 클릭 전환을 높입니다.",
                ],
            },
        ],
        "strategy": [
            {
                "title": "경험형 콘텐츠를 우선 설계",
                "desc": "제품을 설명하기보다 먹는 장면, 반응, 공간 분위기, 상황극을 통해 시청자가 경험을 상상하게 만듭니다.",
                "basis": "롱폼 콘텐츠 유형 × 마케팅 목적 분석, 댓글 반응 분석",
            },
            {
                "title": "첫 화면에서 제품·인물 반응을 보여주기",
                "desc": "제품 단독 컷보다 인물이 제품을 먹거나 사용하는 장면, 놀람·만족 같은 얼굴 반응을 앞에 배치합니다.",
                "basis": "숏츠 영상 구성 분석, 숏츠 댓글 감성 분석",
            },
            {
                "title": "썸네일은 감각 요소를 압축",
                "desc": "음식·제품이 크게 보이고, 인물 표정이나 짧은 후킹 문구가 함께 보이도록 구성합니다.",
                "basis": "롱폼 썸네일 시각 요소 분석",
            },
            {
                "title": "댓글 반응을 다음 소재로 확장",
                "desc": "댓글에서 많이 언급된 맛, 공간, 이벤트, 가격, 재미 요소를 다음 콘텐츠 아이디어로 사용합니다.",
                "basis": "롱폼·숏츠 댓글 분석",
            },
        ],
    },
    "IT": {
        "emoji": "💻",
        "color": "#2563eb",
        "soft": "#eff6ff",
        "soft2": "#f8fbff",
        "border": "#bfdbfe",
        "title": "IT 기업 유튜브 운영 가이드라인",
        "subtitle": "명확한 정보 구조, 기술 설명, 전문적 시각 구성을 중심으로 신뢰와 이해도를 높이는 전략입니다.",
        "summary": [
            {
                "title": "롱폼",
                "icon": "▶",
                "source": "롱폼 분석 결과 · 롱폼 댓글 분석",
                "items": [
                    "정보 전달, 기술 설명, 서비스 활용, 문제 해결형 콘텐츠가 적합합니다.",
                    "복잡한 기능은 문제 상황 → 해결 방식 → 기대 효과 순서로 구조화합니다.",
                    "제목·설명·CTA는 서비스 이해와 사용 유도 목적에 맞춰 일관되게 작성합니다.",
                ],
            },
            {
                "title": "숏츠",
                "icon": "⚡",
                "source": "숏츠 분석 결과 · 숏츠 댓글 분석",
                "items": [
                    "첫 3초 안에 핵심 문구나 문제 상황을 텍스트로 제시합니다.",
                    "모션그래픽과 화면 전환은 기술 정보를 빠르게 이해시키는 핵심 도구로 활용합니다.",
                    "텍스트, 화면 예시, 기능 흐름이 명확히 보이는 구성이 적합합니다.",
                ],
            },
            {
                "title": "썸네일",
                "icon": "▧",
                "source": "썸네일 분석 결과",
                "items": [
                    "큰 텍스트, 밝고 단순한 배경, 전문적인 색감, 명확한 정보 구조가 중요합니다.",
                    "기능명, 숫자, 비교 포인트, 혜택을 짧게 압축해 보여줍니다.",
                    "파란색·중립 톤 등 신뢰감 있는 색감을 활용하고 배경 복잡도는 낮춥니다.",
                ],
            },
        ],
        "strategy": [
            {
                "title": "정보 구조를 먼저 설계",
                "desc": "기능을 나열하기보다 사용자가 겪는 문제, 해결 과정, 결과를 단계별로 보여줍니다.",
                "basis": "롱폼 콘텐츠 유형 × 마케팅 목적 분석",
            },
            {
                "title": "숏츠는 텍스트와 그래픽 중심",
                "desc": "첫 3초 안에 핵심 문구를 크게 보여주고, 모션그래픽으로 기능 흐름을 시각화합니다.",
                "basis": "숏츠 영상 구성 분석",
            },
            {
                "title": "썸네일은 명확성과 신뢰감 우선",
                "desc": "큰 텍스트, 밝은 배경, 전문적인 색감, 단순한 정보 구조로 클릭 이유를 바로 전달합니다.",
                "basis": "롱폼 썸네일 시각 요소 분석",
            },
            {
                "title": "댓글로 이해도 점검",
                "desc": "댓글에서 어렵다, 헷갈린다, 더 알고 싶다는 반응이 나오면 다음 콘텐츠에서 설명 순서를 보완합니다.",
                "basis": "롱폼·숏츠 댓글 분석",
            },
        ],
    },
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .gd-title {
            font-size: 34px;
            font-weight: 950;
            color: #111827;
            letter-spacing: -1.1px;
            margin-bottom: 4px;
        }

        .gd-desc {
            color: #6b7280;
            font-size: 14px;
            line-height: 1.65;
            margin-bottom: 16px;
            word-break: keep-all;
        }

        .gd-domain-hero {
            box-sizing: border-box;
            width: 100%;
            border-radius: 22px;
            padding: 22px 24px;
            margin-bottom: 14px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
        }

        .gd-domain-title {
            font-size: 24px;
            font-weight: 950;
            letter-spacing: -0.55px;
            margin-bottom: 6px;
            word-break: keep-all;
        }

        .gd-domain-subtitle {
            color: #4b5563;
            font-size: 13.5px;
            line-height: 1.65;
            word-break: keep-all;
        }

        .gd-section-title {
            font-size: 21px;
            font-weight: 950;
            color: #111827;
            letter-spacing: -0.45px;
            margin-bottom: 6px;
        }

        .gd-section-caption {
            color: #6b7280;
            font-size: 12.8px;
            line-height: 1.6;
            word-break: keep-all;
            margin-bottom: 12px;
        }

        .gd-summary-head {
            display: flex;
            align-items: center;
            gap: 9px;
            margin-bottom: 7px;
            min-width: 0;
        }

        .gd-summary-icon {
            width: 32px;
            height: 32px;
            border-radius: 999px;
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            font-weight: 950;
            flex-shrink: 0;
        }

        .gd-card-title {
            font-size: 15px;
            font-weight: 950;
            margin: 0;
            letter-spacing: -0.2px;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }

        .gd-source-chip {
            display: inline-block;
            max-width: 100%;
            box-sizing: border-box;
            font-size: 11px;
            font-weight: 800;
            color: #6b7280;
            background: #f9fafb;
            border: 1px solid #eef2f7;
            border-radius: 999px;
            padding: 3px 8px;
            margin: 0 0 8px 0;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        .gd-card-body {
            color: #374151;
            font-size: 13px;
            line-height: 1.58;
            word-break: keep-all;
            overflow-wrap: anywhere;
            margin-bottom: 4px;
        }

        .gd-basis-native {
            color: #8b95a1;
            font-size: 11.5px;
            line-height: 1.45;
            word-break: keep-all;
            overflow-wrap: anywhere;
            margin-top: 6px;
            padding: 0;
            background: transparent;
            border: 0;
        }

        .gd-save-box {
            box-sizing: border-box;
            width: 100%;
            border-radius: 22px;
            padding: 22px 24px;
            margin-top: 4px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        }

        .gd-save-title {
            font-size: 22px;
            font-weight: 950;
            color: #111827;
            letter-spacing: -0.5px;
            margin-bottom: 6px;
        }

        .gd-save-desc {
            color: #4b5563;
            font-size: 13.5px;
            line-height: 1.65;
            word-break: keep-all;
            margin-bottom: 10px;
        }

        .gd-preview-title {
            color: #111827;
            font-size: 16px;
            font-weight: 950;
            margin: 8px 0 8px 0;
        }

        div[data-testid="stDownloadButton"] > button {
            font-weight: 850 !important;
            border-radius: 14px !important;
        }

        /* Streamlit bordered container 안쪽 overflow 보정 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            overflow: hidden;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] * {
            max-width: 100%;
            box-sizing: border-box;
        }

        @media (max-width: 900px) {
            .gd-title { font-size: 28px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _as_bullets(items: list[str]) -> str:
    return "\n".join([f"- {item}" for item in items])


def _make_markdown(domain: str) -> str:
    data = DOMAIN_DATA[domain]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        f"# {data['emoji']} {data['title']}",
        "",
        f"> 생성 시각: {now}",
        "",
        data["subtitle"],
        "",
        "## 1. 분석 기반 핵심 요약",
    ]

    for block in data["summary"]:
        md.append(f"### {block['title']}")
        md.append(f"_근거: {block['source']}_")
        md.append(_as_bullets(block["items"]))
        md.append("")

    md.append("## 2. 최종 운영 전략")
    for idx, item in enumerate(data["strategy"], start=1):
        md.append(f"### {idx}. {item['title']}")
        md.append(item["desc"])
        md.append(f"- 근거: {item['basis']}")
        md.append("")

    md.append("---")
    md.append("본 가이드라인은 롱폼, 숏츠, 롱폼 썸네일, 댓글 분석 결과를 종합해 작성한 분석 기반 운영 지침입니다.")
    return "\n".join(md)


def _render_header() -> None:
    st.markdown('<div class="gd-title">최종 가이드라인</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="gd-desc">
            롱폼, 숏츠, 롱폼 썸네일, 댓글 분석 결과에서 도출한 핵심 운영 기준을
            도메인별 실행 가이드라인으로 정리합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_domain_selector() -> str:
    return st.radio(
        "도메인 선택",
        ["FnB", "IT"],
        horizontal=True,
        key="guideline_domain_radio",
    )


def _render_domain_hero(domain: str) -> None:
    data = DOMAIN_DATA[domain]
    st.markdown(
        f"""
        <div class="gd-domain-hero" style="border:1px solid {data['border']}; background:linear-gradient(135deg,#ffffff 0%,{data['soft']} 100%);">
            <div class="gd-domain-title" style="color:{data['color']};">{data['emoji']} {data['title']}</div>
            <div class="gd-domain-subtitle">{data['subtitle']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary(domain: str) -> None:
    data = DOMAIN_DATA[domain]

    with st.container(border=True):
        st.markdown('<div class="gd-section-title">분석 기반 핵심 요약</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="gd-section-caption">각 분석 탭에서 가이드라인으로 전환할 수 있는 내용을 요약했습니다.</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(3, gap="medium")
        for col, block in zip(cols, data["summary"]):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="gd-summary-head">
                            <div class="gd-summary-icon" style="background:{data['color']};">{block['icon']}</div>
                            <div class="gd-card-title" style="color:{data['color']};">{block['title']}</div>
                        </div>
                        <div class="gd-source-chip">{block['source']}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(_as_bullets(block["items"]))


def _render_strategy(domain: str) -> None:
    data = DOMAIN_DATA[domain]

    with st.container(border=True):
        st.markdown('<div class="gd-section-title">최종 운영 전략</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="gd-section-caption">분석 결과를 실제 콘텐츠 운영 의사결정에 활용할 수 있도록 실행 문장으로 정리했습니다.</div>',
            unsafe_allow_html=True,
        )

        for row in [data["strategy"][:2], data["strategy"][2:]]:
            cols = st.columns(2, gap="medium")
            for col, item in zip(cols, row):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="gd-card-title" style="color:{data["color"]};">{item["title"]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f'<div class="gd-card-body">{item["desc"]}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="gd-basis-native">근거 · {item["basis"]}</div>',
                            unsafe_allow_html=True,
                        )


def _render_save(domain: str) -> None:
    data = DOMAIN_DATA[domain]
    report = _make_markdown(domain)
    filename = f"{domain}_기업_유튜브_운영_가이드라인_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    st.markdown(
        f"""
        <div class="gd-save-box" style="border:1px solid {data['border']}; background:linear-gradient(135deg,#ffffff 0%,{data['soft']} 100%);">
            <div class="gd-save-title">가이드라인 저장</div>
            <div class="gd-save-desc">
                현재 선택한 <b>{domain}</b> 도메인의 최종 운영 가이드라인을 Markdown 파일로 다운로드합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gd-preview-title">Markdown 가이드라인</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(report)

    st.download_button(
        "최종 가이드라인 저장",
        data=report,
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
    )


def render_guideline_page() -> None:
    _inject_css()
    _render_header()
    domain = _render_domain_selector()
    _render_domain_hero(domain)
    _render_summary(domain)
    _render_strategy(domain)
    _render_save(domain)


def render_guideline_tab() -> None:
    render_guideline_page()


def render() -> None:
    render_guideline_page()
