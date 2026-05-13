"""
thunbnail_generation_v1.py — 썸네일 제작 컴포넌트
메인 페이지에서 render()를 호출해 원하는 위치에 임베드합니다.

    from thunbnail_generation_v1 import render as render_maker
    render_maker(FNB, IT, yt_key)
"""
import streamlit as st
import time
import re
import requests
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
from utills_thumbnail_v2 import (
    inject_css, init_session_state, _load_bench_cache,
    gemini_gen_prompt, gemini_gen_improvement_prompt, gemini_gen_thumbnail_analysis,
    gemini_gen_image, gemini_edit_image, _pts_to_prompt_hint,
    init_vertex, _get_genai_client, fmt_num,
    IMAGE_GEN_CONFIG
)
from google.genai import types as _genai_types

# ── 모델 상수 ──────────────────────────────────
GEMINI_PROMPT_MODEL     = st.secrets.get("GEMINI_PROMPT_MODEL",  "gemini-2.5-flash-lite")
GEMINI_VISION_MODEL     = st.secrets.get("GEMINI_VISION_MODEL",  "gemini-2.5-flash")
GEMINI_IMAGE_MODEL      = st.secrets.get("GEMINI_IMAGE_MODEL",   "gemini-3-pro-image-preview")
GEMINI_MULTIMODAL_MODEL = GEMINI_IMAGE_MODEL   # 표시용 alias (하위 호환)

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
    """벤치마크 데이터 로드. CSV 없으면 하드코딩 기본값 반환."""
    _csv_fnb, _csv_it = _load_bench_cache(csv_path)
    FNB = {**_FNB_HARD, **_csv_fnb} if _csv_fnb else _FNB_HARD
    IT  = {**_IT_HARD,  **_csv_it}  if _csv_it  else _IT_HARD
    return FNB, IT


def render(FNB: dict, IT: dict, yt_key: str = ""):
    """
    썸네일 제작 UI(개선+신규)를 현재 위치에 렌더링합니다.
    채널 분석 페이지와 세션 상태를 공유하므로 분석 후 사용하면
    가이드라인이 자동으로 적용됩니다.

    Args:
        FNB:    FnB 벤치마크 데이터
        IT:     IT  벤치마크 데이터
        yt_key: YouTube Data API v3 키 (확장성 위해 유지, 현재 미사용)
    """
    _guide_is_fnb = st.session_state.get("guide_domain_sel", "🍔 FnB").startswith("🍔")

    st.markdown('<div class="sec-title"><span class="tbar" style="background:#2ba640"></span>AI 썸네일 제작</div>', unsafe_allow_html=True)

    # 현재 선택된 도메인 가이드라인 자동 적용
    if _guide_is_fnb and st.session_state.fnb_channel and st.session_state.fnb_analysis:
        applied_guide = {
            "text": st.session_state.fnb_guideline or "",
            "bad": st.session_state.fnb_analysis.get("bad",[]),
            "act": st.session_state.fnb_analysis.get("act",[]),
            "domain": "FnB",
        }
    elif not _guide_is_fnb and st.session_state.it_channel and st.session_state.it_analysis:
        applied_guide = {
            "text": st.session_state.it_guideline or "",
            "bad": st.session_state.it_analysis.get("bad",[]),
            "act": st.session_state.it_analysis.get("act",[]),
            "domain": "IT",
        }
    else:
        applied_guide = None
        st.markdown('<div class="api-notice">💡 위에서 채널을 분석하면 가이드라인이 자동으로 적용됩니다.</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#e5e7eb'>", unsafe_allow_html=True)

    # ─── 서브 탭: 분석 기반 개선 / 새 썸네일 제작 ───
    sub_improve, sub_new = st.tabs(["🔧  분석 기반 개선", "✨  새로운 썸네일 제작"])

    # ──── 분석 기반 개선 탭 ────
    with sub_improve:
        st.markdown(
            '<div style="font-size:12px;color:#9ca3af;margin-bottom:12px">'
            '기존 썸네일을 선택하거나 업로드하면 <strong style="color:#1f2937">Gemini가 자동으로 분석</strong>하고, '
            '수정 지시를 입력하면 개선된 썸네일을 생성합니다.</div>',
            unsafe_allow_html=True)

        # ── 채널 분석 페이지에서 전달된 큐 감지 → 자동 세팅 ──
        _queue_on_load = st.session_state.get("thumb_analysis_queue")
        if _queue_on_load and st.session_state.get("jump_to_improve"):
            _qa = _queue_on_load.get("analysis") or {}
            _qe = _qa.get("elements", {})
            _qs = _qa.get("strengths", [])
            _qi = _qa.get("improvements", [])
            _qac = [i.get("action","") for i in _qi if isinstance(i,dict) and i.get("action")]
            _qdom = _queue_on_load.get("domain", "FnB")
            _qvt  = _queue_on_load.get("video",{}).get("title","")
            _is_fnb_q = (_qdom == "FnB")

            # 분석 결과를 imp_auto_analysis로 주입 (이후 프롬프트 자동생성에 활용)
            if _qa and not st.session_state.get("imp_auto_analysis"):
                st.session_state["imp_auto_analysis"] = _qa

            # 프롬프트 자동 생성
            if not st.session_state.get("imp_prompt"):
                _current_d = ", ".join(filter(None, [
                    _qe.get("main_objects",""), _qe.get("color_palette",""), _qe.get("person_details","")
                ]))[:180]
                _keep_d = _qs[0] if _qs else ""
                _domain_d = "FnB 푸드&베버리지 브랜드" if _is_fnb_q else "IT 기업"
                _auto_prompt = (
                    f"한국 {_domain_d} 유튜브 썸네일 개선. "
                    + (f"현재 구성: {_current_d}. " if _current_d else "")
                    + (f"유지할 강점: {_keep_d}. " if _keep_d else "")
                    + (f"개선 사항: {'; '.join(_qac[:3])}. " if _qac else "")
                    + ("따뜻하고 식욕을 돋우는 색감, 선명한 음식 사진, 프로페셔널 구도" if _is_fnb_q
                       else "깔끔하고 전문적인 디자인, 밝은 배경, 핵심 메시지 강조")
                )
                st.session_state["imp_prompt"] = _auto_prompt

            st.session_state["jump_to_improve"] = False  # 플래그 리셋

            st.markdown(
                f'<div class="guide-hint">📌 채널 분석 결과 자동 적용됨 — '
                f'<b>{_qvt[:40]}</b><br>'
                f'<span style="font-size:10px;color:#6b7280">개선 지시 프롬프트가 분석 결과를 기반으로 작성됐습니다.</span></div>',
                unsafe_allow_html=True)

        # ── 기존 썸네일 소스 선택 ──   # gemini-3-pro-image-preview 모델은 utills_thumbnail_v2.py 함수에 있음!
        src_mode = st.radio("기존 썸네일 가져오기",
            ["분석한 영상에서 선택", "YouTube URL 입력"],
            horizontal=True, key="imp_src_mode")

        # 분석된 영상 목록에서 선택
        _ref_thumb_url = ""
        _ref_thumb_img = None
        _ref_title = ""
        _ref_analysis = None

        if src_mode == "분석한 영상에서 선택":
            # FnB/IT 분석된 영상 목록 합치기
            all_lf = []
            if st.session_state.fnb_videos:
                for v in st.session_state.fnb_videos:
                    if v.get("verdict") in ("longform","unknown"):
                        all_lf.append({"label": f"[FnB] {v['title'][:40]}", "v": v, "domain": "FnB"})
                        
            if st.session_state.it_videos:
                for v in st.session_state.it_videos:
                    if v.get("verdict") in ("longform","unknown"):
                        all_lf.append({"label": f"[IT] {v['title'][:40]}", "v": v, "domain": "IT"})

            queue = st.session_state.get("thumb_analysis_queue")
            if queue:
                _ref_thumb_url = queue["video"].get("thumbnail_hq","")
                _ref_title     = queue["video"]["title"]
                _ref_analysis  = queue.get("analysis")
                _ref_analysis_domain = queue.get("domain","FnB")
                if _ref_thumb_url:
                    st.markdown(f'<div style="font-size:11px;color:#2563eb;margin-bottom:8px">📌 FnB/IT 탭에서 전달된 썸네일: {_ref_title[:50]}</div>', unsafe_allow_html=True)
            elif all_lf:
                sel_idx = st.selectbox("영상 선택", range(len(all_lf)),
                    format_func=lambda i: all_lf[i]["label"], key="imp_vid_sel")
                _sel_v = all_lf[sel_idx]["v"]
                _ref_thumb_url = _sel_v.get("thumbnail_hq","")
                _ref_title     = _sel_v["title"]
                _ref_analysis_domain = all_lf[sel_idx]["domain"]
                pfx = "fnb" if _ref_analysis_domain=="FnB" else "it"
                cached_ana = st.session_state.get(f"{pfx}_thumb_analysis")
                if cached_ana and cached_ana.get("video_id") == _sel_v["id"]:
                    _ref_analysis = cached_ana.get("data")
            else:
                st.markdown('<div class="api-notice">💡 FnB/IT 탭에서 채널을 먼저 분석해주세요.</div>', unsafe_allow_html=True)
                _ref_analysis_domain = "FnB"

        else:  # YouTube URL 입력
            _ref_analysis_domain = st.selectbox("도메인", ["FnB", "IT"], key="imp_url_domain")
            yt_url_input = st.text_input(
                "YouTube 영상 URL",
                placeholder="예: https://www.youtube.com/watch?v=xxxxxxxxxxx",
                key="imp_yt_url",
            )

            def _extract_video_id_from_url(url: str) -> str:
                """YouTube URL에서 video ID 추출"""
                url = url.strip()
                # youtu.be 단축 URL
                m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
                if m: return m.group(1)
                # youtube.com/watch?v=
                m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
                if m: return m.group(1)
                # youtube.com/shorts/
                m = re.search(r"/shorts/([A-Za-z0-9_-]{11})", url)
                if m: return m.group(1)
                return ""

            def _fetch_yt_thumbnail(video_id: str) -> tuple[str, str]:
                """YouTube video ID → (thumbnail_url, title)"""
                # 썸네일 URL: maxresdefault 우선, 없으면 hqdefault
                thumb_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
                # maxresdefault 존재 확인
                try:
                    r = requests.head(thumb_url, timeout=5)
                    if r.status_code != 200:
                        thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                except Exception:
                    thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

                # 제목은 YouTube oEmbed API로 가져오기 (API 키 불필요)
                title = video_id
                try:
                    oembed = requests.get(
                        f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
                        timeout=5,
                    )
                    if oembed.ok:
                        title = oembed.json().get("title", video_id)
                except Exception:
                    pass
                return thumb_url, title

            if yt_url_input.strip():
                _vid_id = _extract_video_id_from_url(yt_url_input)
                if _vid_id:
                    # URL이 바뀌었을 때만 재조회
                    if st.session_state.get("_imp_last_vid_id") != _vid_id:
                        with st.spinner("썸네일 불러오는 중..."):
                            _thumb_url, _vid_title = _fetch_yt_thumbnail(_vid_id)
                            st.session_state["_imp_last_vid_id"]    = _vid_id
                            st.session_state["_imp_last_thumb_url"] = _thumb_url
                            st.session_state["_imp_last_title"]     = _vid_title
                    _ref_thumb_url = st.session_state.get("_imp_last_thumb_url", "")
                    _ref_title     = st.session_state.get("_imp_last_title", _vid_id)
                    st.markdown(
                        f'<div class="guide-hint">✅ 영상 ID: <b>{_vid_id}</b><br>'
                        f'<span style="font-size:11px">{_ref_title[:60]}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div class="api-notice">⚠️ 올바른 YouTube URL을 입력해주세요.<br>예: https://www.youtube.com/watch?v=xxxx 또는 https://youtu.be/xxxx</div>', unsafe_allow_html=True)

        # ── 기존 썸네일 미리보기 + 자동 분석 + 수정 지시 ──
        if _ref_thumb_url or _ref_thumb_img:
            col_ref, col_edit = st.columns([1, 2])
            with col_ref:
                st.markdown('<div style="font-size:11px;color:#9ca3af;margin-bottom:4px">📌 기존 썸네일</div>', unsafe_allow_html=True)
                if _ref_thumb_url:
                    st.image(_ref_thumb_url, use_container_width=True)
                elif _ref_thumb_img:
                    st.image(_ref_thumb_img, use_container_width=True)
                if _ref_title:
                    st.markdown(f'<div style="font-size:9px;color:#9ca3af;margin-top:3px">{_ref_title[:45]}...</div>', unsafe_allow_html=True)

                # ── 썸네일 자동 분석 버튼 ──
                has_proj = bool(st.session_state.get("vertex_project",""))
                if st.button("🔍 썸네일 자동 분석 + 프롬프트 생성", key="imp_auto_analyze",
                             disabled=not has_proj, use_container_width=True,
                             help="Gemini Vision이 기존 썸네일을 분석하고 개선 프롬프트를 자동 작성합니다"):
                    with st.spinner(f"① {GEMINI_VISION_MODEL} Vision 분석 중... ② {GEMINI_PROMPT_MODEL} 프롬프트 생성 중..."):
                        try:
                            _p = st.session_state.get("vertex_project","")
                            _l = st.session_state.get("vertex_location","us-central1")
                            _bench = FNB if _ref_analysis_domain=="FnB" else IT

                            # 분석 영상 선택 / YouTube URL 모드 모두 URL 사용
                            _url_for_ana = _ref_thumb_url

                            # ── 분석 실행 ──
                            ana_result = gemini_gen_thumbnail_analysis(
                                _url_for_ana, _ref_title, 0,
                                _ref_analysis_domain, _bench, _p, _l
                            )
                            st.session_state["imp_auto_analysis"] = ana_result

                            # ── 분석 결과에서 정보 추출 ──
                            _elems  = ana_result.get("elements", {})
                            _bench_ = ana_result.get("benchmark", {})
                            _strgs  = ana_result.get("strengths", [])
                            _imprv  = ana_result.get("improvements", [])

                            # ── 분석 결과 기반 한국어 구조화 가이드 생성 ──
                            # 사용자가 보고 자유롭게 편집 가능, 백엔드에서 자동 영어 번역됨
                            _dom_label = "FnB 음식 브랜드" if _ref_analysis_domain == "FnB" else "IT 기술 브랜드"

                            _lines = [
                                f"[목표] 한국 {_dom_label} 유튜브 썸네일 개선",
                                "기존 첨부 썸네일 이미지를 참고하여 아래 분석 결과를 반영한 새 썸네일을 생성합니다.",
                                "",
                                "[기존 썸네일 현황 (Vision 분석)]",
                            ]
                            if _elems.get("main_objects"):
                                _lines.append(f"• 주요 피사체: {_elems['main_objects']}")
                            if _elems.get("color_palette"):
                                _lines.append(f"• 색감/톤: {_elems['color_palette']}")
                            if _elems.get("person_details"):
                                _lines.append(f"• 인물: {_elems['person_details']}")
                            if _elems.get("text_overlay") and _elems["text_overlay"] != "텍스트 없음":
                                _lines.append(f"• 텍스트: {_elems['text_overlay']}")
                            if _elems.get("brand_elements") and _elems["brand_elements"] not in ("없음", ""):
                                _lines.append(f"• 브랜드: {_elems['brand_elements']}")

                            if _strgs:
                                _lines.append("")
                                _lines.append("[유지할 강점]")
                                for _s in _strgs[:2]:
                                    _lines.append(f"• {_s}")

                            if _imprv:
                                _lines.append("")
                                _lines.append("[개선 방향]")
                                for _i, _tip in enumerate(_imprv[:3], 1):
                                    if isinstance(_tip, dict):
                                        _issue = _tip.get("issue", "").strip()
                                        _action = _tip.get("action", "").strip()
                                        if _issue and _action:
                                            _lines.append(f"{_i}. {_issue}")
                                            _lines.append(f"   → 개선: {_action}")
                                        elif _action:
                                            _lines.append(f"{_i}. {_action}")
                                    elif isinstance(_tip, str):
                                        _lines.append(f"{_i}. {_tip}")

                            _lines.append("")
                            _lines.append("[텍스트·로고 규칙 — 중요]")
                            _lines.append("• 기존 썸네일의 텍스트(글자)와 로고는 반드시 보존 — 완전 제거 금지")
                            _lines.append("• 단, 스타일 개선은 OK: 폰트 두께/크기/위치/색상/그림자/배경과의 어우러짐")
                            _lines.append("• 기존 텍스트 문구는 가독성 유지, 브랜드 로고는 식별 가능하게 유지")
                            _lines.append("• ⚠ 얼굴 보호: 텍스트/로고가 인물의 눈·코·입·턱을 절대 가리지 않도록 — 빈 공간 쪽 또는 턱 아래로 배치, 필요시 텍스트 크기를 줄여서라도 얼굴 위로 겹치지 말 것")
                            _lines.append("")
                            _lines.append("[제약 조건]")
                            _lines.append("• 16:9 가로 비율")
                            _lines.append("• 한국 브랜드 미감, 강력한 시각적 후킹")
                            _lines.append("• 고품질 전문 사진 스타일")
                            _lines.append("")
                            _lines.append("※ 위 내용을 자유롭게 수정/추가하세요. 한국어로 작성하면 자동으로 영어 번역되어 이미지 생성됩니다.")

                            _auto_prompt = "\n".join(_lines)

                            st.session_state["imp_prompt"] = _auto_prompt
                            st.rerun()
                        except Exception as e:
                            st.error(f"분석 실패: {e}")

                # 분석 결과 표시
                _auto_analysis = st.session_state.get("imp_auto_analysis") or _ref_analysis
                if _auto_analysis and isinstance(_auto_analysis, dict):
                    imprv_list = _auto_analysis.get("improvements",[])
                    if imprv_list:
                        st.markdown('<div style="font-size:10px;color:#ea580c;font-weight:600;margin:8px 0 4px">💡 분석된 개선점</div>', unsafe_allow_html=True)
                        for tip in imprv_list[:3]:
                            issue  = tip.get("issue","") if isinstance(tip,dict) else str(tip)
                            action = tip.get("action","") if isinstance(tip,dict) else ""
                            st.markdown(
                                f'<div style="background:#f0f2f5;border-left:3px solid #ff9944;border-radius:0 5px 5px 0;padding:5px 8px;margin-bottom:4px">' 
                                f'<div style="color:#ff7020;font-size:9px;font-weight:600">{issue}</div>' 
                                f'<div style="color:#374151;font-size:9px;margin-top:2px">→ {action}</div></div>',
                                unsafe_allow_html=True)

            with col_edit:
                st.markdown('<div style="font-size:11px;color:#9ca3af;margin-bottom:4px">✏️ 수정하고 싶은 내용을 직접 입력하세요</div>', unsafe_allow_html=True)
                imp_keywords = st.text_area(
                    "수정 지시",
                    placeholder="예: 인물 얼굴을 더 크게 클로즈업\n배경을 단순하게 정리\n브랜드 로고 오른쪽 하단에 배치\n전체적으로 더 밝고 따뜻한 색감으로",
                    key="imp_kw", height=120, label_visibility="collapsed")

                imp_kw_c1, imp_kw_c2 = st.columns([1,1])
                with imp_kw_c1:
                    imp_auto = st.button("✨ 프롬프트 자동생성", key="imp_auto",
                                        disabled=not st.session_state.get("vertex_project",""),
                                        use_container_width=True)
                with imp_kw_c2:
                    if st.button("🗑 초기화", key="imp_clear", use_container_width=True):
                        st.session_state["thumb_analysis_queue"] = None
                        st.session_state["imp_auto_analysis"] = None
                        st.session_state["imp_prompt"] = ""
                        st.rerun()

            # 프롬프트 자동생성 버튼 (Gemini + 분석 결과 + 사용자 수정 지시 통합)
            if imp_auto:
                _proj = st.session_state.get("vertex_project","")
                _loc  = st.session_state.get("vertex_location","us-central1")
                _cur_ana = st.session_state.get("imp_auto_analysis") or _ref_analysis

                try:
                    _elems = (_cur_ana or {}).get("elements", {})
                    _imprv = (_cur_ana or {}).get("improvements", [])
                    _strgs = (_cur_ana or {}).get("strengths", [])

                    current_summary = ", ".join(filter(None, [
                        _elems.get("main_objects", ""),
                        _elems.get("color_palette", ""),
                        _elems.get("person_details", ""),
                        _elems.get("brand_elements", ""),
                    ]))[:400]

                    improvement_hints = []
                    for tip in _imprv[:5]:
                        if isinstance(tip, dict):
                            if tip.get("prompt_hint"):
                                improvement_hints.append(tip["prompt_hint"])
                            elif tip.get("action"):
                                improvement_hints.append(tip["action"])
                        elif isinstance(tip, str):
                            improvement_hints.append(tip)

                    with st.spinner(f"{GEMINI_PROMPT_MODEL}로 개선 프롬프트 생성 중..."):
                        _new_prompt = gemini_gen_improvement_prompt(
                            user_instruction=imp_keywords.strip(),
                            domain=_ref_analysis_domain,
                            current_summary=current_summary,
                            strengths=_strgs,
                            improvement_hints=improvement_hints,
                            project_id=_proj,
                            location=_loc,
                        )

                    st.session_state["imp_prompt"] = _new_prompt
                    st.rerun()

                except Exception as e:
                    st.error(f"프롬프트 자동생성 실패: {e}")

            # 프롬프트 초기값 — 분석 결과 있으면 그걸 기반으로, 없으면 기본값
            imp_is_fnb = (_ref_analysis_domain == "FnB")
            _cur_ana_for_default = st.session_state.get("imp_auto_analysis") or _ref_analysis

            if _cur_ana_for_default and isinstance(_cur_ana_for_default, dict):
                _e = _cur_ana_for_default.get("elements", {})
                _i = _cur_ana_for_default.get("improvements", [])
                _s = _cur_ana_for_default.get("strengths", [])
                _current_d = ", ".join(filter(None, [
                    _e.get("main_objects",""), _e.get("color_palette",""), _e.get("person_details","")
                ]))[:180]
                _hints_d = [t.get("action","") for t in _i[:3] if isinstance(t,dict) and t.get("action")]
                _keep_d  = _s[0] if _s else ""
                _domain_d = "FnB 푸드&베버리지 브랜드" if imp_is_fnb else "IT 기업"
                _imp_default = (
                    f"한국 {_domain_d} 유튜브 썸네일 개선. "
                    + (f"현재 구성: {_current_d}. " if _current_d else "")
                    + (f"유지할 강점: {_keep_d}. " if _keep_d else "")
                    + (f"개선 사항: {'; '.join(_hints_d)}. " if _hints_d else "")
                    + ("따뜻하고 식욕을 돋우는 색감, 선명한 음식 사진, 프로페셔널 구도" if imp_is_fnb
                       else "깔끔하고 전문적인 디자인, 밝은 배경, 핵심 메시지 강조")
                )
            else:
                _imp_default = (
                    f"한국 {'FnB 브랜드 — 따뜻하고 식욕을 돋우는 색감, 선명한 음식 사진' if imp_is_fnb else 'IT 기업 — 깔끔하고 전문적인 디자인, 밝은 색감'} "
                    f"스타일의 유튜브 썸네일 개선. 인물 구도 강화, 브랜드 로고 배치, 시선을 끄는 비주얼 후킹 요소 추가"
                )

            if "imp_prompt" not in st.session_state or not st.session_state["imp_prompt"]:
                st.session_state["imp_prompt"] = _imp_default

            imp_final_prompt = st.text_area(
                "개선 지시 (한국어로 작성 → 자동 번역 후 생성)",
                key="imp_prompt", height=420,
                help=f"한국어로 자유롭게 작성하세요. 생성 시 {GEMINI_PROMPT_MODEL}가 영어로 번역해 {GEMINI_IMAGE_MODEL}에 전달합니다. 분석 결과 기반 가이드가 자동으로 채워지며 자유롭게 수정 가능합니다.")

            imp_gen_btn = st.button("🔧 개선 썸네일 생성", key="imp_gen",
                                   disabled=not st.session_state.get("vertex_project",""),
                                   use_container_width=True)

            if imp_gen_btn:
                if not _ref_thumb_url:
                    st.error("썸네일을 먼저 선택하거나 YouTube URL을 입력하세요.")
                else:
                    _proj = st.session_state.get("vertex_project","")
                    _loc  = st.session_state.get("vertex_location","us-central1")

                    # 분석 힌트 추출
                    _cur_analysis = st.session_state.get("imp_auto_analysis") or _ref_analysis
                    _hints_list = []
                    if _cur_analysis:
                        _hints_list = _cur_analysis.get("prompt_hints",[])
                        if not _hints_list:
                            for tip in _cur_analysis.get("improvements",[]):
                                if isinstance(tip,dict) and tip.get("prompt_hint"):
                                    _hints_list.append(tip["prompt_hint"])
                    analysis_hints = "; ".join(_hints_list[:3]) if _hints_list else ""

                    edit_instruction = imp_keywords.strip() or imp_final_prompt.strip()

                    with st.spinner(f"① 기존 썸네일 다운로드 중... ② {GEMINI_VISION_MODEL} 스타일 분석 중... ③ {GEMINI_IMAGE_MODEL} 개선 이미지 생성 중..."):
                        try:
                            # URL → bytes 먼저 다운로드
                            _r = requests.get(_ref_thumb_url, timeout=15)
                            _r.raise_for_status()
                            _thumb_bytes = _r.content

                            img_bytes, used_prompt = gemini_edit_image(
                                img_bytes_or_url=_thumb_bytes,   # bytes 전달
                                edit_prompt=edit_instruction,     # 키워드 인자
                                analysis_hints=analysis_hints,
                                domain=_ref_analysis_domain,
                                project_id=_proj,
                                location=_loc,
                            )
                            
                            img = Image.open(BytesIO(img_bytes))
                            st.session_state.generated_thumb = {
                                "bytes": img_bytes, "image": img,
                                "prompt": used_prompt,
                                "domain": _ref_analysis_domain,
                                "category": "스타일 개선",
                                "mode": "개선",
                                "source_thumb": _ref_thumb_url,
                                "source_title": _ref_title,
                                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            }
                            st.success("✅ 개선 썸네일 생성 완료!")
                            with st.expander("📄 사용된 최종 프롬프트 보기"):
                                st.code(used_prompt, language="text")
                            st.rerun()
                        except Exception as e:
                            st.error(f"생성 실패: {e}")
        else:
            if src_mode == "분석한 영상에서 선택" and not st.session_state.fnb_videos and not st.session_state.it_videos:
                pass
            elif src_mode == "YouTube URL 입력":
                st.markdown('<div class="api-notice">💡 YouTube 영상 URL을 입력하면 썸네일을 자동으로 불러옵니다.</div>', unsafe_allow_html=True)

    # ──── 새로운 썸네일 제작 탭 ────
    with sub_new:
        st.markdown('<div style="font-size:12px;font-weight:600;color:#1f2937;margin-bottom:10px">새로운 컨셉으로 썸네일을 제작합니다</div>', unsafe_allow_html=True)

        g1,g2,g3 = st.columns(3)
        with g1: t_domain   = st.selectbox("도메인",["FnB (식음료)","IT (기술/소프트웨어)"], key="t_domain")
        with g2: t_category = st.selectbox("썸네일 카테고리",["정보 전달형","예능/콘텐츠형","인터뷰/인물형","브랜드 이미지형","리뷰/비교형","제품 홍보형"], key="t_cat")
        with g3: t_color    = st.selectbox("색상 톤",["warm (따뜻한)","cool (차가운)","neutral (중간)"], key="t_color")
        is_fnb = "FnB" in t_domain
        dk = "FnB" if is_fnb else "IT"

        kw1, kw2 = st.columns([4,1])
        with kw1:
            keywords = st.text_input("핵심 키워드", placeholder="예: 신제품 도시락, 여름 한정 메뉴, 콜라보 이벤트", key="keywords", label_visibility="collapsed")
        with kw2:
            auto_btn = st.button("✨ 자동생성", key="auto_btn", disabled=not st.session_state.get("vertex_project",""))

        chips = (["신제품 리뷰","편의점 콜라보","여름 한정","인기 상품 TOP5","매운맛 챌린지","MD 추천"]
                 if is_fnb else ["AI 솔루션","클라우드 전환","신제품 발표","기술 트렌드","전문가 인터뷰","디지털 혁신"])
        st.markdown('<div style="margin-bottom:8px">' +
            " ".join([f'<span style="display:inline-block;background:#f3f4f6;border:1px solid #d1d5db;border-radius:50px;padding:3px 10px;font-size:11px;color:#4b5563;margin:2px">{c}</span>' for c in chips]) +
            '</div>', unsafe_allow_html=True)

        if auto_btn:
            if not keywords.strip():
                st.error("키워드를 먼저 입력해주세요")
            else:
                with st.spinner(f"{GEMINI_PROMPT_MODEL} 프롬프트 생성 중..."):
                    try:
                        ch_info = st.session_state.fnb_channel if is_fnb else st.session_state.it_channel
                        _proj = st.session_state.get("vertex_project","")
                        _loc  = st.session_state.get("vertex_location","us-central1")
                        en_p, ko_p = gemini_gen_prompt(keywords, dk, t_category, t_color, ch_info, _proj, _loc)
                        # 내부적으로 영어 프롬프트 보관, UI에는 한국어 요약 표시
                        st.session_state["generated_prompt"] = en_p   # 실제 생성에 사용
                        st.session_state["final_prompt_ko"] = ko_p    # UI 표시용
                        st.session_state["_prompt_source"] = "auto"   # 큐/기본값 덮어쓰기 방지
                        st.session_state["_last_fb"] = (dk, t_category)
                        st.success("✅ 프롬프트 생성 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"프롬프트 생성 실패: {e}")

        _fb = {
            ("FnB","정보 전달형"):   "편의점 신제품 소개 썸네일. 음식이 전면에 선명하게 배치되고, 따뜻하고 밝은 조명, 브랜드 로고 포함, 프로페셔널 푸드 사진",
            ("FnB","예능/콘텐츠형"): "한국 FnB 브랜드 예능 콘텐츠 썸네일. 음식에 반응하는 두 명의 인물, 편의점 배경, 따뜻하고 높은 대비, 생동감 있는 구도",
            ("IT","정보 전달형"):    "한국 IT 기업 정보 전달형 썸네일. 차분한 블루 계열 색감, 디지털 배경, 신뢰감 있는 전문적 디자인",
            ("IT","예능/콘텐츠형"):  "한국 IT 브랜드 유튜브 썸네일. 현대적인 사무실에서 테크 전문가, 쿨한 색상, 역동적이고 전문적인 구도",
        }
        fb = _fb.get((dk, t_category), f"한국 {dk} 브랜드 유튜브 썸네일. 시선을 끄는 전문적 구도, 고품질")

        # ── 텍스트박스 초기값 설정 로직 ──
        # 우선순위: ① 자동생성 결과 > ② 분석 큐 > ③ 도메인/카테고리 기본값
        # 자동생성 후에는 _prompt_source="auto"로 표시해 큐/기본값이 덮어쓰지 못하게 보호
        _prompt_source = st.session_state.get("_prompt_source", "")

        if _prompt_source == "auto":
            # 자동생성 결과 보호 — 도메인/카테고리가 바뀔 때만 초기화
            if st.session_state.get("_last_fb") != (dk, t_category):
                st.session_state["final_prompt_ko"] = fb
                st.session_state["generated_prompt"] = ""
                st.session_state["_prompt_source"] = ""
                st.session_state["_last_fb"] = (dk, t_category)

        else:
            # 분석 큐에서 영상 특징 가져오기 (개선 썸네일 제작하기 버튼으로 전달된 경우)
            _queue = st.session_state.get("thumb_analysis_queue")
            if _queue and _queue.get("analysis"):
                _qa  = _queue["analysis"]
                _qe  = _qa.get("elements", {})
                _qs  = _qa.get("strengths", [])
                _qi  = _qa.get("improvements", [])
                _qac = [i.get("action","") for i in _qi if isinstance(i,dict) and i.get("action")]
                _qvt = _queue.get("video",{}).get("title","")
                _queue_ko = (
                    f"'{_qvt[:25]}' 영상 특징 기반 신규 썸네일. "
                    + (f"현재 강점: {_qs[0]}. " if _qs else "")
                    + (f"개선 방향: {_qac[0]}. " if _qac else "")
                    + (f"구성: {_qe.get('main_objects','')[:60]}." if _qe.get('main_objects') else "")
                )
                if st.session_state.get("_last_fb") != (dk, t_category):
                    st.session_state["final_prompt_ko"] = _queue_ko
                    st.session_state["generated_prompt"] = ""
                    st.session_state["_last_fb"] = (dk, t_category)
            else:
                if "final_prompt_ko" not in st.session_state or st.session_state.get("_last_fb") != (dk, t_category):
                    st.session_state["final_prompt_ko"] = fb
                    st.session_state["generated_prompt"] = ""
                    st.session_state["_last_fb"] = (dk, t_category)

        final_prompt_ko = st.text_area(
            "썸네일 개념 설명 (한국어 → 자동 번역 후 생성)",
            key="final_prompt_ko", height=420,
            help=f"한국어로 자유롭게 작성하세요. ✨ 프롬프트 자동생성 버튼을 누르면 {GEMINI_PROMPT_MODEL}이 분석 결과·도메인·카테고리를 반영한 구조화 가이드를 채워줍니다. 자유롭게 수정한 뒤 🎨 생성 버튼을 누르면 {GEMINI_IMAGE_MODEL}로 전송됩니다.")
        final_prompt = final_prompt_ko

        gen_btn = st.button("🎨 새 썸네일 생성하기", key="gen_btn", disabled=not st.session_state.get("vertex_project",""), use_container_width=True)
        if gen_btn:
            if not final_prompt.strip():
                st.error("프롬프트를 입력해주세요")
            else:
                _proj = st.session_state.get("vertex_project","")
                _loc  = st.session_state.get("vertex_location","us-central1")
                with st.spinner(f"{GEMINI_MULTIMODAL_MODEL}로 새 썸네일 생성 중... (최대 60초)"):
                    try:
                        # 자동생성 버튼으로 만든 영어 프롬프트 있으면 재사용
                        _stored_en = st.session_state.get("generated_prompt","")
                        if _stored_en:
                            _translated = _stored_en
                        else:
                            # 한국어 → 시각 요소 보강 + 영어 변환 (단순 번역 X)
                            _tr_prompt = (
                                "You are a YouTube thumbnail visual prompt engineer.\n"
                                "Convert the following Korean thumbnail description into a detailed English image generation prompt.\n"
                                "- Translate AND expand: add visual details for any specific objects, foods, characters, or brands mentioned.\n"
                                "- For foods: describe appearance (color, texture, shape) in detail.\n"
                                "- For characters/IP (e.g. SpongeBob, Pikachu): describe their visual appearance specifically.\n"
                                "- For brand collabs: mention brand colors and visual identity.\n"
                                "- Output 80-120 words. Return ONLY the English prompt, no explanation.\n\n"
                                f"Korean: {final_prompt_ko.strip()}"
                            )
                            _tr_client = _get_genai_client(_proj, _loc)
                            _tr_resp = _tr_client.models.generate_content(
                                model=GEMINI_PROMPT_MODEL,
                                contents=[_tr_prompt],
                                config=_genai_types.GenerateContentConfig(
                                    temperature=IMAGE_GEN_CONFIG["translate_temperature"],
                                    max_output_tokens=IMAGE_GEN_CONFIG["translate_max_tokens"],
                                ),
                            )
                            _translated = (_tr_resp.text or "").strip()
                        ctx = "Korean food beverage brand, warm appetizing" if is_fnb else "Korean IT company, professional corporate"
                        full_p = (
                            f"{_translated}. "
                            f"Korean YouTube thumbnail, {t_category} style, {t_color} color tone, {ctx}. "
                            f"VIVID saturated colors, BRIGHT well-lit, HIGH CONTRAST, "
                            f"subject fills 60-80% of frame, dynamic eye-catching composition, "
                            f"clean background, wide horizontal 16:9 format. "
                            f"Text & logo policy: if the keywords suggest a hook word, "
                            f"include a short Korean text overlay (2-5 characters, bold sans-serif, "
                            f"heavy weight, white text with thick black outline or strong drop shadow, "
                            f"placed on the clean side of the frame). "
                            f"If a brand or product is named, include a small brand logo "
                            f"in the lower-left or lower-right corner. "
                            f"Otherwise leave ~30% of one side as clean space and do NOT invent random text. "
                            f"FACE PROTECTION (NON-NEGOTIABLE): text and logos must NEVER overlap any person's face — "
                            f"eyes, nose, mouth, and chin stay fully visible. Place text on the clean side opposite the face, "
                            f"or in the bottom strip below the chin. Shrink text before ever crossing the face."
                        )
                        if applied_guide and isinstance(applied_guide, dict):
                            full_p += f" Brand: {applied_guide['text'][:80]}"
                        img_bytes = gemini_gen_image(full_p, _proj, _loc)
                        img = Image.open(BytesIO(img_bytes))
                        st.session_state.generated_thumb = {
                            "bytes": img_bytes, "image": img,
                            "prompt": final_prompt, "domain": dk,
                            "category": t_category, "mode": "신규",
                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        st.success("✅ 썸네일 생성 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"생성 실패: {e}")

    # ─── 공통 결과 미리보기 ───
    if st.session_state.generated_thumb:
        t = st.session_state.generated_thumb
        st.markdown("<hr style='border-color:#e5e7eb'>", unsafe_allow_html=True)
        mode_badge = (f'<span class="badge badge-green">{t.get("mode","신규")}</span>'
                      if t.get("mode")=="신규"
                      else f'<span class="badge badge-orange">{t.get("mode","개선")}</span>')
        st.markdown(f'<div style="font-size:12px;font-weight:600;color:#1f2937;margin-bottom:8px">🖼 생성 결과 {mode_badge}</div>', unsafe_allow_html=True)

        # 개선 모드면 before/after 비교
        if t.get("mode") == "개선" and t.get("source_thumb"):
            _after_b64 = base64.b64encode(t["bytes"]).decode()
            _src_title = (t.get("source_title") or "")[:35]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
                f'<div style="flex:1;min-width:0">'
                f'<div style="font-size:10px;color:#9ca3af;margin-bottom:4px;text-align:center">BEFORE (원본)</div>'
                f'<img src="{t["source_thumb"]}" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;border:1px solid #e5e7eb">'
                f'<div style="font-size:9px;color:#9ca3af;margin-top:3px;text-align:center">{_src_title}...</div>'
                f'</div>'
                f'<div style="flex-shrink:0;display:flex;align-items:center;justify-content:center;'
                f'width:36px;height:36px;background:#f3f4f6;border-radius:50%;border:1px solid #d1d5db">'
                f'<span style="font-size:18px;color:#d97706;line-height:1">&#8594;</span>'
                f'</div>'
                f'<div style="flex:1;min-width:0">'
                f'<div style="font-size:10px;color:#16a34a;margin-bottom:4px;text-align:center">AFTER (AI 개선)</div>'
                f'<img src="data:image/png;base64,{_after_b64}" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;border:1px solid #2ba640">'
                f'<div style="font-size:9px;color:#9ca3af;margin-top:3px;text-align:center">{t["generated_at"]}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            p2_col = st.columns([1])[0]
            with p2_col:
                bc = "badge-red" if t["domain"]=="FnB" else "badge-blue"
                st.markdown(
                    f'<div class="yt-card" style="margin-top:8px"><div style="font-size:10px;color:#9ca3af;margin-bottom:6px">생성 정보</div>'
                    f'<span class="badge {bc}">{t["domain"]}</span>'
                    f'  <span style="font-size:11px;color:#4b5563">{t["category"]}</span>'
                    f'<div style="font-size:10px;color:#9ca3af;margin-top:6px">{t["generated_at"]} | {GEMINI_MULTIMODAL_MODEL}</div></div>',
                    unsafe_allow_html=True)
        else:
            p1, p2 = st.columns([3,1])
            with p1:
                st.image(t["image"], use_container_width=True)
                st.markdown(f'<div style="font-size:10px;color:#9ca3af;margin-top:4px">{t["generated_at"]} | {GEMINI_MULTIMODAL_MODEL}</div>', unsafe_allow_html=True)
            with p2:
                bc = "badge-red" if t["domain"]=="FnB" else "badge-blue"
                st.markdown(
                    f'<div class="yt-card"><div style="font-size:10px;color:#9ca3af;margin-bottom:8px">생성 정보</div>'
                    f'<span class="badge {bc}">{t["domain"]}</span>'
                    f'<div style="font-size:11px;color:#4b5563;margin-top:6px">{t["category"]}</div></div>',
                    unsafe_allow_html=True)

        # 저장 버튼 (공통)
        sv1, sv2 = st.columns([1,3])
        with sv1:
            buf = BytesIO(); t["image"].save(buf, "PNG")
            st.download_button("⬇ PNG 다운로드", buf.getvalue(),
                f"thumb_{int(time.time())}.png", "image/png", key="dl_thumb")
        with sv2:
            if st.button("💾 저장함에 저장", key="save_thumb"):
                buf2 = BytesIO(); t["image"].save(buf2,"PNG")
                st.session_state.saved_items.append({
                    "id": int(time.time()*1000), "type": "thumbnail",
                    "domain": t["domain"],
                    "title": f"{t.get('mode','신규')} {t['category']} 썸네일",
                    "prompt": t["prompt"],
                    "image_bytes": buf2.getvalue(),
                    "source_thumb": t.get("source_thumb",""),
                    "source_title": t.get("source_title",""),
                    "mode": t.get("mode","신규"),
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                st.success("저장 완료!")

# ══════════════════════════════════════════════
# 단독 실행  streamlit run thunbnail_generation_v1.py
# ══════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(
        page_title="썸네일 제작 — TubeStrategy",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    init_session_state()

    # 사이드바 완전히 숨김
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

    yt_key         = st.secrets["YOUTUBE_API_KEY"]
    vertex_project = st.secrets["GOOGLE_CLOUD_PROJECT"]
    vertex_location= st.secrets["GOOGLE_CLOUD_REGION"]
    st.session_state["vertex_project"]  = vertex_project
    st.session_state["vertex_location"] = vertex_location

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding:10px 0;'
        'border-bottom:1px solid #e8eaed;margin-bottom:14px">'
        '<span style="font-size:15px;font-weight:600">🎨 썸네일 제작</span>'
        '<span style="font-size:12px;color:#6b7280">| 개선 · 신규 생성</span>'
        '</div>', unsafe_allow_html=True)

    FNB, IT = get_bench_data()

    render(FNB, IT, yt_key)