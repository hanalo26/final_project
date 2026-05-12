"""
thumbnail_analysis.py — 채널 분석 컴포넌트
메인 페이지에서 render()를 호출해 원하는 위치에 임베드합니다.

    from thunbnail_analysis_v1 import render as render_analysis
    render_analysis(FNB, IT, yt_key)
"""
import streamlit as st
import time
from utills_thumbnail import (
    inject_css, init_session_state,
    render_channel_result, run_channel_analysis,
    yt_search_channel_candidates, render_channel_candidate_picker,
    fmt_num, _load_bench_cache,
)

# ── 벤치마크 하드코딩 기본값 ──────────────────
_FNB_HARD = {
    "has_person":75,"person_cat":{"0명":25,"1명":43,"2명+":32},
    "has_text":65,"brand":55,"brightness":"보통-밝음","saturation":"높음","contrast":"높음",
    "design_quality":3.8,"text_len":17,
    "category":{"예능/콘텐츠형":45,"정보 전달형":30,"제품 홍보형":15,"리뷰/비교형":10},
    "color_tone":{"warm":55,"neutral":30,"cool":15},
}
_IT_HARD = {
    "has_person":68,"person_cat":{"0명":32,"1명":48,"2명+":20},
    "has_text":82,"brand":70,"brightness":"밝음","saturation":"보통","contrast":"높음",
    "design_quality":4.1,"text_len":22,
    "category":{"정보 전달형":50,"예능/콘텐츠형":25,"인터뷰/인물형":15,"브랜드 이미지형":10},
    "color_tone":{"neutral":45,"cool":40,"warm":15},
}


def get_bench_data(csv_path: str = "all_thumbnail.csv"):
    """벤치마크 데이터 로드 (CSV 없으면 하드코딩 기본값 반환)"""
    _csv_fnb, _csv_it = _load_bench_cache(csv_path)
    FNB = {**_FNB_HARD, **_csv_fnb} if _csv_fnb else _FNB_HARD
    IT  = {**_IT_HARD,  **_csv_it}  if _csv_it  else _IT_HARD
    return FNB, IT


def render(FNB: dict, IT: dict, yt_key: str):
    """
    채널 분석 UI를 현재 위치에 렌더링합니다.

    Args:
        FNB:    FnB 벤치마크 데이터  (get_bench_data() 로 얻거나 직접 전달)
        IT:     IT  벤치마크 데이터
        yt_key: YouTube Data API v3 키
    """
    guide_domain = st.radio(
        "업종 선택",
        ["🍔 FnB", "💻 IT"],
        horizontal=True,
        key="guide_domain_sel",
        label_visibility="collapsed",
    )
    _guide_is_fnb = guide_domain.startswith("🍔")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if _guide_is_fnb:
        st.markdown('<div class="sec-title"><span class="tbar" style="background:#ff0000"></span>FnB 기업 맞춤형 썸네일 전략 (롱폼 기준)</div>', unsafe_allow_html=True)

        if not yt_key:
            st.markdown('<div class="api-notice">💡 사이드바에서 YouTube Data API v3 키를 입력해주세요.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([4,1])
        with c1:
            fnb_q = st.text_input("채널명 또는 채널 ID",
                placeholder="예: CU씨유튜브 / @CUtube / UCxxxxxxxx",
                key="fnb_q", label_visibility="collapsed")
        with c2:
            fnb_go = st.button("🔍 채널 분석", key="fnb_go")

        if fnb_go:
            if not yt_key:
                st.error("사이드바에서 YouTube API 키를 입력해주세요")
            elif not fnb_q.strip():
                st.error("채널명을 입력해주세요")
            else:
                with st.spinner("공식 채널 후보를 찾는 중..."):
                    try:
                        st.session_state["fnb_channel_candidates"] = yt_search_channel_candidates(fnb_q, yt_key)
                        st.success("채널 후보를 찾았습니다. 아래에서 공식 채널을 선택한 뒤 분석을 실행하세요.")
                    except Exception as e:
                        st.error(f"오류: {e}")

        fnb_candidates = st.session_state.get("fnb_channel_candidates", [])
        if fnb_candidates:
            selected_cid = render_channel_candidate_picker("fnb", fnb_candidates)
            if st.button("✅ 선택한 FnB 채널 분석", key="fnb_analyze_selected"):
                with st.spinner("선택한 채널 데이터 수집 중 (롱폼/숏폼 구분 포함)..."):
                    try:
                        ch, vids, ana = run_channel_analysis(selected_cid, yt_key, FNB, "FnB")
                        st.session_state.fnb_channel = ch
                        st.session_state.fnb_videos = vids
                        st.session_state.fnb_analysis = ana
                        st.session_state.fnb_guideline = (
                            f"FnB 채널 '{ch['name']}' 롱폼 분석. "
                            "권장: 예능/콘텐츠형, Warm/Neutral 색상, 2인 이상 인물 구성, 브랜드 로고"
                        )
                        lf_cnt = ana.get("longform_count", 0)
                        sf_cnt = ana.get("shortform_count", 0)
                        st.session_state["fnb_report_requested"] = False
                        st.session_state["fnb_top5_analysis"] = None
                        st.success(f"✅ '{ch['name']}' 분석 완료! — 롱폼 {lf_cnt}개 · 숏폼 {sf_cnt}개 감지")
                    except Exception as e:
                        st.error(f"오류: {e}")

        if st.session_state.fnb_channel:
            render_channel_result("fnb", "#ff0000", FNB, "FnB")

    else:
        st.markdown('<div class="sec-title"><span class="tbar" style="background:#3ea6ff"></span>IT 기업 맞춤형 썸네일 전략 (롱폼 기준)</div>', unsafe_allow_html=True)

        if not yt_key:
            st.markdown('<div class="api-notice">💡 사이드바에서 YouTube Data API v3 키를 입력해주세요.</div>', unsafe_allow_html=True)

        ic1, ic2 = st.columns([4,1])
        with ic1:
            it_q = st.text_input("채널명 또는 채널 ID",
                placeholder="예: 삼성SDS / @samsungsds / UCxxxxxxxx",
                key="it_q", label_visibility="collapsed")
        with ic2:
            it_go = st.button("🔍 채널 분석", key="it_go")

        if it_go:
            if not yt_key:
                st.error("사이드바에서 YouTube API 키를 입력해주세요")
            elif not it_q.strip():
                st.error("채널명을 입력해주세요")
            else:
                with st.spinner("공식 채널 후보를 찾는 중..."):
                    try:
                        st.session_state["it_channel_candidates"] = yt_search_channel_candidates(it_q, yt_key)
                        st.success("채널 후보를 찾았습니다. 아래에서 공식 채널을 선택한 뒤 분석을 실행하세요.")
                    except Exception as e:
                        st.error(f"오류: {e}")

        it_candidates = st.session_state.get("it_channel_candidates", [])
        if it_candidates:
            selected_cid = render_channel_candidate_picker("it", it_candidates)
            if st.button("✅ 선택한 IT 채널 분석", key="it_analyze_selected"):
                with st.spinner("선택한 채널 데이터 수집 중 (롱폼/숏폼 구분 포함)..."):
                    try:
                        ch, vids, ana = run_channel_analysis(selected_cid, yt_key, IT, "IT")
                        st.session_state.it_channel = ch
                        st.session_state.it_videos = vids
                        st.session_state.it_analysis = ana
                        st.session_state.it_guideline = (
                            f"IT 채널 '{ch['name']}' 롱폼 분석. "
                            "권장: 정보 전달형, Neutral/Cool 색상, 전문가 인물, 핵심 텍스트"
                        )
                        lf_cnt = ana.get("longform_count", 0)
                        sf_cnt = ana.get("shortform_count", 0)
                        st.session_state["it_report_requested"] = False
                        st.session_state["it_top5_analysis"] = None
                        st.success(f"✅ '{ch['name']}' 분석 완료! — 롱폼 {lf_cnt}개 · 숏폼 {sf_cnt}개 감지")
                    except Exception as e:
                        st.error(f"오류: {e}")

        if st.session_state.it_channel:
            render_channel_result("it", "#3ea6ff", IT, "IT")


# ══════════════════════════════════════════════
# 단독 실행  streamlit run thunbnail_analysis_v1.py
# ══════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(
        page_title="채널 분석 — TubeStrategy",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_session_state()

    yt_key         = st.secrets["YOUTUBE_API_KEY"]
    vertex_project = st.secrets["GOOGLE_CLOUD_PROJECT"]
    vertex_location= st.secrets["GOOGLE_CLOUD_REGION"]
    st.session_state["vertex_project"]  = vertex_project
    st.session_state["vertex_location"] = vertex_location

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding:10px 0;'
        'border-bottom:1px solid #e8eaed;margin-bottom:14px">'
        '<span style="font-size:15px;font-weight:600">📋 채널 분석</span>'
        '<span style="font-size:12px;color:#6b7280">| FnB · IT 분야에서 제작되는 유튜브 동영상의 맞춤형 썸네일 전략 보고서</span>'
        '</div>', unsafe_allow_html=True)

    FNB, IT = get_bench_data()

    render(FNB, IT, yt_key)