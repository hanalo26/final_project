import streamlit as st

from longform_core import (
    apply_longform_minimal_style,
    ensure_longform_resources,
    page_home,
    page_domain,
    page_shap,
    page_strategy,
    page_waterfall,
    page_simulator,
    page_roadmap,
)

try:
    from longform_comment_dashboard_tab import page_longform_comment_analysis
except Exception as _comment_import_error:
    def page_longform_comment_analysis():
        st.error("롱폼 댓글 분석 탭을 불러오지 못했습니다.")
        st.exception(_comment_import_error)


DASHBOARD_SECTIONS = {
    "핵심 요약": {
        "desc": "전체 롱폼 영상의 평균 성공 확률, 최근 업로드 흐름, 도메인별 차이를 한 화면에서 확인합니다.",
        "render": page_home,
        "need_model": False,
    },
    "도메인별 분석": {
        "desc": "FnB와 IT 도메인별로 성공률, 콘텐츠 유형, CTA, 업로드 시간·요일 차이를 비교합니다.",
        "render": page_domain,
        "need_model": False,
    },
    "성공 요인": {
        "desc": "AI 모델의 성공 기여도 분석을 통해 롱폼 성과에 영향을 준 핵심 변수를 확인합니다.",
        "render": page_shap,
        "need_model": False,
    },
    "운영 전략": {
        "desc": "분석 결과를 바탕으로 롱폼 콘텐츠 기획, 설명란, CTA, 업로드 전략을 정리합니다.",
        "render": page_strategy,
        "need_model": False,
    },
    "영상 진단": {
        "desc": "개별 롱폼 영상의 성공 확률과 주요 기여 요인을 Waterfall 방식으로 확인합니다.",
        "render": page_waterfall,
        "need_model": True,
    },
    "AI 시뮬레이터": {
        "desc": "영상 제작 조건을 입력해 롱폼 성공 가능성과 개선 전략을 시뮬레이션합니다.",
        "render": page_simulator,
        "need_model": True,
    },
    "전략 로드맵": {
        "desc": "도메인별 롱폼 운영 전략을 기획 → 제작 → 업로드 → 배포 → 분석 흐름으로 정리합니다.",
        "render": page_roadmap,
        "need_model": False,
    },
    "댓글 반응 분석": {
        "desc": "롱폼 댓글 긍정 비율을 도메인·콘텐츠 유형·CTA 조건별로 분석하고 예측 시뮬레이션합니다.",
        "render": page_longform_comment_analysis,
        "need_model": False,
    },
}


def _render_tab_intro():
    st.markdown(
        """
        <div class="page-card" style="margin-bottom:16px;">
            <div style="font-size:23px;font-weight:900;color:#111827;margin-bottom:6px;">
                📊 롱폼 분석 결과 대시보드
            </div>
            <div style="font-size:14px;color:#6b7280;line-height:1.6;">
                롱폼 성과 분석, 성공 요인, 영상 진단, 댓글 반응, AI 시뮬레이터, 전략 로드맵을 한 곳에서 확인합니다.
                분석 결과와 진단형 기능을 함께 활용해 롱폼 운영 전략을 정리하세요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_selector():
    labels = list(DASHBOARD_SECTIONS.keys())
    try:
        return st.segmented_control(
            "롱폼 분석 항목",
            labels,
            default=labels[0],
            label_visibility="collapsed",
            key="longform_dashboard_section",
        )
    except Exception:
        return st.radio(
            "롱폼 분석 항목",
            labels,
            horizontal=True,
            label_visibility="collapsed",
            key="longform_dashboard_section_radio",
        )


def render_longform_dashboard():
    """app.py의 '롱폼 분석 결과' 탭에서 호출하는 함수."""
    apply_longform_minimal_style()
    _render_tab_intro()

    selected = _section_selector()
    info = DASHBOARD_SECTIONS[selected]

    try:
        # 기본 분석 섹션은 데이터만 로드하고,
        # 영상 진단 / AI 시뮬레이터처럼 모델이 필요한 섹션에서만 모델을 로드합니다.
        ensure_longform_resources(load_model=info.get("need_model", False))
    except Exception as e:
        st.error("롱폼 분석 데이터 또는 모델 로드 중 오류가 발생했습니다.")
        st.caption("DATA_PATH 경로, longform 결과 CSV/JSON 파일, joblib 모델 파일 위치를 확인해 주세요.")
        st.exception(e)
        return

    st.caption(info["desc"])
    st.markdown("---")

    try:
        info["render"]()
    except Exception as e:
        st.error(f"'{selected}' 섹션을 렌더링하는 중 오류가 발생했습니다.")
        st.exception(e)
