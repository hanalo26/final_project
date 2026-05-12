import streamlit as st


def _module_card(icon: str, title: str, desc: str):
    st.markdown(
        f"""
        <div class="module-card">
            <div class="module-icon">{icon}</div>
            <div class="module-title">{title}</div>
            <div class="module-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing_page():
    st.markdown(
        """
        <style>
        .landing-wrap { color: #111827; }

        .hero {
            background:
                radial-gradient(circle at 86% 30%, rgba(239,35,60,0.16) 0%, rgba(239,35,60,0.00) 34%),
                linear-gradient(135deg, #ffffff 0%, #fff8f9 100%);
            border: 1px solid #e9edf3;
            border-radius: 24px;
            padding: 34px 38px;
            min-height: 245px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
            display: grid;
            grid-template-columns: 1.35fr 0.75fr;
            gap: 22px;
            align-items: center;
            margin-bottom: 16px;
        }

        .hero-kicker {
            color: #ef233c;
            font-size: 13px;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .hero-title {
            color: #111827;
            font-size: 36px;
            line-height: 1.28;
            font-weight: 950;
            letter-spacing: -1.2px;
            margin-bottom: 14px;
            word-break: keep-all;
        }

        .hero-desc {
            color: #4b5563;
            font-size: 15px;
            line-height: 1.8;
            max-width: 830px;
            word-break: keep-all;
        }

        .hero-actions {
            display: flex;
            gap: 12px;
            margin-top: 22px;
            flex-wrap: wrap;
        }

        .hero-actions a { text-decoration: none !important; }

        .hero-primary {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #ef233c;
            color: #ffffff !important;
            border-radius: 14px;
            padding: 12px 18px;
            font-size: 13px;
            font-weight: 900;
            box-shadow: 0 8px 20px rgba(239,35,60,0.22);
            transition: 0.15s ease;
        }

        .hero-primary:hover {
            transform: translateY(-1px);
            background: #dc1f36;
        }

        .hero-secondary {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #ffffff;
            color: #374151 !important;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px 18px;
            font-size: 13px;
            font-weight: 900;
            transition: 0.15s ease;
        }

        .hero-secondary:hover {
            transform: translateY(-1px);
            border-color: #fecdd3;
            background: #fff7f8;
            color: #ef233c !important;
        }

        .hero-visual {
            height: 188px;
            position: relative;
            border-radius: 24px;
            background:
                linear-gradient(90deg, rgba(239,35,60,0.10) 0 14%, transparent 14% 24%, rgba(239,35,60,0.14) 24% 38%, transparent 38% 48%, rgba(239,35,60,0.18) 48% 64%, transparent 64% 100%),
                linear-gradient(135deg, #fff1f2 0%, #ffffff 100%);
            border: 1px solid #fee2e2;
            overflow: hidden;
        }

        .hero-visual::before {
            content: "▶";
            position: absolute;
            left: 50%;
            top: 48%;
            transform: translate(-50%, -50%);
            width: 118px;
            height: 82px;
            background: linear-gradient(135deg, #ff3535, #e60012);
            color: #ffffff;
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 42px;
            box-shadow: 0 22px 34px rgba(239,35,60,0.25);
        }

        .hero-visual::after {
            content: "";
            position: absolute;
            right: 28px;
            bottom: 30px;
            width: 78px;
            height: 78px;
            border-radius: 50%;
            background:
                conic-gradient(#ef233c 0deg 105deg, #fecdd3 105deg 220deg, #f3f4f6 220deg 360deg);
            box-shadow: 0 10px 24px rgba(15,23,42,0.08);
        }

        .section-title {
            font-size: 20px;
            font-weight: 950;
            color: #111827;
            letter-spacing: -0.4px;
            margin: 22px 0 8px 0;
        }

        .section-subtitle {
            font-size: 13px;
            color: #6b7280;
            line-height: 1.7;
            margin-top: -2px;
            margin-bottom: 12px;
            word-break: keep-all;
        }

        .module-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 20px 18px;
            min-height: 154px;
            box-shadow: 0 6px 18px rgba(15,23,42,0.045);
        }

        .module-icon {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: #fff1f2;
            border: 1px solid #fecdd3;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 21px;
            margin-bottom: 14px;
        }

        .module-title {
            font-size: 16px;
            font-weight: 950;
            color: #111827;
            margin-bottom: 8px;
        }

        .module-desc {
            color: #4b5563;
            font-size: 13px;
            line-height: 1.65;
            word-break: keep-all;
        }

        .flow-panel, .scope-panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 22px;
            padding: 24px 22px;
            box-shadow: 0 6px 18px rgba(15,23,42,0.045);
            margin-top: 2px;
        }

        .flow-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
        }

        .flow-item {
            position: relative;
            background: #fbfbfc;
            border: 1px solid #eef2f7;
            border-radius: 18px;
            padding: 18px 14px;
            min-height: 142px;
            text-align: center;
        }

        .flow-badge {
            position: absolute;
            top: -10px;
            left: 14px;
            width: 26px;
            height: 26px;
            border-radius: 999px;
            background: #ef233c;
            color: #ffffff;
            font-size: 12px;
            font-weight: 950;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .flow-icon {
            width: 50px;
            height: 50px;
            margin: 4px auto 11px auto;
            border-radius: 50%;
            background: #fff1f2;
            color: #ef233c;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }

        .flow-title {
            font-size: 14px;
            font-weight: 950;
            color: #111827;
            margin-bottom: 7px;
        }

        .flow-desc {
            font-size: 11.5px;
            color: #6b7280;
            line-height: 1.55;
            word-break: keep-all;
        }

        .scope-panel {
            border-radius: 20px;
            padding: 22px 22px 20px 22px;
            margin-top: 16px;
        }

        .scope-head {
            display: flex;
            align-items: center;
            gap: 14px;
            padding-bottom: 18px;
            border-bottom: 1px solid #eef2f7;
            margin-bottom: 20px;
        }

        .scope-head-icon {
            width: 54px;
            height: 54px;
            border-radius: 16px;
            background: #eff6ff;
            color: #2563eb;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 27px;
            flex-shrink: 0;
        }

        .scope-head-title {
            color: #111827;
            font-size: 17px;
            font-weight: 950;
            margin-bottom: 4px;
        }

        .scope-head-desc {
            color: #6b7280;
            font-size: 13px;
            line-height: 1.65;
            word-break: keep-all;
        }

        .scope-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
        }

        .scope-card {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: #fbfbfc;
            border: 1px solid #eef2f7;
            border-radius: 16px;
            padding: 15px 16px;
            min-height: 88px;
        }

        .scope-icon {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: #fff1f2;
            border: 1px solid #fee2e2;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 18px;
        }

        .scope-title {
            color: #111827;
            font-size: 13px;
            font-weight: 950;
            margin-bottom: 5px;
        }

        .scope-desc {
            color: #6b7280;
            font-size: 11.5px;
            line-height: 1.55;
            word-break: keep-all;
        }

        .success-standard-box {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 18px;
            padding: 16px 18px;
            margin-top: 14px;
            display: flex;
            gap: 14px;
            align-items: flex-start;
        }

        .success-standard-icon {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: #ffedd5;
            color: #f97316;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            flex-shrink: 0;
        }

        .success-standard-title {
            color: #9a3412;
            font-size: 14px;
            font-weight: 950;
            margin-bottom: 4px;
        }

        .success-standard-desc {
            color: #7c2d12;
            font-size: 13px;
            line-height: 1.7;
            word-break: keep-all;
        }

        .goal-callout {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 15px 18px;
            margin-top: 14px;
            display: flex;
            gap: 14px;
            align-items: center;
        }

        .goal-icon {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: #dbeafe;
            color: #2563eb;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            flex-shrink: 0;
        }

        .goal-text {
            color: #1f2937;
            font-size: 14px;
            line-height: 1.7;
            word-break: keep-all;
        }

        @media (max-width: 1100px) {
            .hero { grid-template-columns: 1fr; }
            .flow-grid { grid-template-columns: 1fr; }
            .scope-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="hero-kicker">기업 유튜브 운영 가이드라인 프로젝트</div>
                <div class="hero-title">
                    성공한 기업 유튜브의 특징을 분석하고,<br>
                    신규 기업을 위한 운영 가이드라인을 제시합니다.
                </div>
                <div class="hero-desc">
                    FnB와 IT 기업 유튜브 채널을 대상으로 롱폼, 썸네일, 숏츠, 댓글 데이터를 종합 분석합니다.
                    단순 조회수 비교가 아니라 참여율 기반 성공 패턴과 AI agent 진단 결과를 연결해
                    실무자가 바로 활용할 수 있는 콘텐츠 운영 전략을 도출합니다.
                </div>
                <div class="hero-actions">
                    <a href="?page=longform" target="_self"><div class="hero-primary">▶ 분석 페이지에서 결과 확인</div></a>
                    <a href="?page=guideline" target="_self"><div class="hero-secondary">▦ 최종 가이드라인으로 연결</div></a>
                </div>
            </div>
            <div class="hero-visual"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">핵심 분석 모듈</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">각 페이지는 하나의 분석 축을 담당하며, 마지막 가이드라인 페이지에서 통합됩니다.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _module_card("📺", "롱폼 분석", "영상 성과와 콘텐츠 구조를 기반으로 장기 콘텐츠 운영 전략을 확인합니다.")
    with c2:
        _module_card("🖼️", "썸네일 분석", "클릭을 유도하는 시각 요소와 제작 방향을 분석합니다.")
    with c3:
        _module_card("🎬", "숏츠 분석", "짧은 영상의 구성 요소와 성공 패턴을 분석합니다.")
    with c4:
        _module_card("💬", "댓글 분석", "댓글 감성, 키워드, 반응을 바탕으로 시청자 인사이트를 확인합니다.")
    with c5:
        _module_card("📘", "가이드라인", "분석 결과와 agent 결과를 통합해 실행 가능한 운영 전략을 제시합니다.")

    st.markdown('<div class="section-title">핵심 분석 흐름</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-panel">
            <div class="flow-grid">
                <div class="flow-item">
                    <div class="flow-badge">1</div>
                    <div class="flow-icon">🎯</div>
                    <div class="flow-title">성공 기준 정의</div>
                    <div class="flow-desc">참여율 중심으로 성공/실패 기준을 설정합니다.</div>
                </div>
                <div class="flow-item">
                    <div class="flow-badge">2</div>
                    <div class="flow-icon">⚖️</div>
                    <div class="flow-title">도메인별 비교</div>
                    <div class="flow-desc">FnB와 IT 기업의 콘텐츠 차이를 비교합니다.</div>
                </div>
                <div class="flow-item">
                    <div class="flow-badge">3</div>
                    <div class="flow-icon">▣</div>
                    <div class="flow-title">영상·썸네일·댓글 분석</div>
                    <div class="flow-desc">다양한 데이터 소스를 통합해 패턴을 도출합니다.</div>
                </div>
                <div class="flow-item">
                    <div class="flow-badge">4</div>
                    <div class="flow-icon">🤖</div>
                    <div class="flow-title">Agent 진단</div>
                    <div class="flow-desc">신규 채널과 영상의 개선점을 자동 진단합니다.</div>
                </div>
                <div class="flow-item">
                    <div class="flow-badge">5</div>
                    <div class="flow-icon">✅</div>
                    <div class="flow-title">가이드라인 제시</div>
                    <div class="flow-desc">실행 가능한 콘텐츠 운영 전략으로 정리합니다.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="scope-panel">
            <div class="scope-head">
                <div class="scope-head-icon">🎯</div>
                <div>
                    <div class="scope-head-title">프로젝트 목표</div>
                    <div class="scope-head-desc">
                        성공 기업의 콘텐츠 특징을 근거로 신규 기업 유튜브 채널이 적용할 수 있는 맞춤형 운영 가이드라인을 제시합니다.
                    </div>
                </div>
            </div>
            <div class="scope-grid">
                <div class="scope-card">
                    <div class="scope-icon">🏢</div>
                    <div>
                        <div class="scope-title">산업 분야</div>
                        <div class="scope-desc">FnB · IT 도메인 중심</div>
                    </div>
                </div>
                <div class="scope-card">
                    <div class="scope-icon">▶</div>
                    <div>
                        <div class="scope-title">콘텐츠 포맷</div>
                        <div class="scope-desc">롱폼 · 썸네일 · 숏츠</div>
                    </div>
                </div>
                <div class="scope-card">
                    <div class="scope-icon">💬</div>
                    <div>
                        <div class="scope-title">분석 데이터</div>
                        <div class="scope-desc">영상 성과 · 썸네일 · 댓글</div>
                    </div>
                </div>
                <div class="scope-card">
                    <div class="scope-icon">📈</div>
                    <div>
                        <div class="scope-title">성공 기준</div>
                        <div class="scope-desc">참여율 기반 성공 패턴</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="success-standard-box">
            <div class="success-standard-icon">📈</div>
            <div>
                <div class="success-standard-title">성공 기준 안내</div>
                <div class="success-standard-desc">
                    본 프로젝트는 채널 내 확산력(참여도·도달률)과 최근 평균 대비 반응(조회·좋아요·댓글)을 각각 통계적 가중치로 합산해, 두 지수를 모두 충족한 영상을 성공으로 분류했습니다.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="goal-callout">
            <div class="goal-icon">📌</div>
            <div class="goal-text">
                본 대시보드는 분석 결과를 나열하는 도구가 아니라,
                <b>성공 패턴 → Agent 진단 → 실행 가이드라인</b>으로 이어지는 기업 유튜브 전략 설계 도구입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
