import streamlit as st
from datetime import datetime
from io import BytesIO

# 팀원 원본 파일명 기준:
# - thunbnail_generation_v1.py
# - utills_thumbnail.py
#
# 주의: 파일명에 오타처럼 보이는 thunbnail / utills가 있지만,
# 현재 팀원 코드의 import 기준이므로 이름을 그대로 유지합니다.

try:
    from thunbnail_generation_v1 import render as render_original_thumbnail_creator
    from thunbnail_generation_v1 import get_bench_data
    from utills_thumbnail import init_session_state
except Exception as e:
    render_original_thumbnail_creator = None
    get_bench_data = None
    init_session_state = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


def _inject_thumbnail_creator_css():
    """원본 썸네일 제작 컴포넌트에서 쓰는 클래스만 최소 스타일링합니다.

    utills_thumbnail.inject_css()는 앱 전체 UI와 충돌할 수 있어 호출하지 않습니다.
    """
    st.markdown(
        """
        <style>
        .sec-title {
            font-size: 18px;
            font-weight: 950;
            color: #111827;
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 10px 0 14px 0;
            letter-spacing: -0.3px;
        }

        .tbar {
            width: 4px;
            height: 19px;
            border-radius: 999px;
            display: inline-block;
            flex-shrink: 0;
        }

        .api-notice {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 14px;
            padding: 12px 14px;
            font-size: 13px;
            color: #92400e;
            line-height: 1.7;
            margin-bottom: 12px;
            word-break: keep-all;
        }

        .guide-hint {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 12px 14px;
            font-size: 13px;
            color: #1e40af;
            line-height: 1.7;
            margin-bottom: 12px;
            word-break: keep-all;
        }

        .yt-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.055);
        }

        .stat-box {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 14px 10px;
            text-align: center;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.045);
        }

        .stat-val {
            font-size: 23px;
            font-weight: 950;
            margin-bottom: 3px;
            letter-spacing: -0.4px;
        }

        .stat-lbl {
            font-size: 12px;
            color: #6b7280;
            font-weight: 750;
        }

        .badge {
            display: inline-block;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 11.5px;
            font-weight: 850;
            margin: 2px;
        }

        .badge-red {
            background: #fff1f2;
            color: #ef233c;
            border: 1px solid #fecdd3;
        }

        .badge-blue {
            background: #eff6ff;
            color: #2563eb;
            border: 1px solid #bfdbfe;
        }

        .badge-green {
            background: #f0fdf4;
            color: #16a34a;
            border: 1px solid #bbf7d0;
        }

        .badge-orange {
            background: #fff7ed;
            color: #ea580c;
            border: 1px solid #fed7aa;
        }

        .badge-gray {
            background: #f9fafb;
            color: #6b7280;
            border: 1px solid #e5e7eb;
        }

        .good-point {
            background: #f0fdf4;
            border-left: 4px solid #16a34a;
            border-radius: 0 12px 12px 0;
            padding: 10px 12px;
            font-size: 13px;
            line-height: 1.65;
            margin-bottom: 8px;
            color: #14532d;
            word-break: keep-all;
        }

        .bad-point {
            background: #fff7ed;
            border-left: 4px solid #ea580c;
            border-radius: 0 12px 12px 0;
            padding: 10px 12px;
            font-size: 13px;
            line-height: 1.65;
            margin-bottom: 8px;
            color: #7c2d12;
            word-break: keep-all;
        }

        .action-point {
            background: #eff6ff;
            border-left: 4px solid #2563eb;
            border-radius: 0 12px 12px 0;
            padding: 10px 12px;
            font-size: 13px;
            line-height: 1.65;
            margin-bottom: 8px;
            color: #1e3a8a;
            word-break: keep-all;
        }

        .report-container {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 20px 22px;
            margin-top: 12px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.055);
        }

        .report-h1 {
            font-size: 19px;
            font-weight: 950;
            color: #111827;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }

        .report-h2 {
            font-size: 15px;
            font-weight: 950;
            color: #111827;
            margin: 18px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 7px;
        }

        .report-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 7px 0;
            border-bottom: 1px solid #f3f4f6;
            font-size: 13px;
        }

        .report-key {
            color: #6b7280;
        }

        .report-val {
            color: #111827;
            font-weight: 850;
        }

        .report-tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 850;
            margin: 2px;
        }

        .creator-download-box {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 14px 16px;
            margin-top: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header():
    st.markdown(
        """
        <div class="page-card" style="margin-bottom:16px;">
            <div style="font-size:24px;font-weight:950;color:#111827;letter-spacing:-0.6px;margin-bottom:6px;">
                썸네일 제작/개선
            </div>
            <div style="font-size:14px;color:#6b7280;line-height:1.65;word-break:keep-all;">
                기존 썸네일을 분석해 개선안을 만들거나, 키워드와 도메인 기준을 바탕으로 새로운 썸네일 이미지를 생성합니다.
                결과는 앱 내부 saved 폴더에 자동 저장하지 않고, PNG 또는 Markdown으로 직접 다운로드합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prepare_secrets():
    yt_key = st.secrets.get("YOUTUBE_API_KEY", "")
    vertex_project = st.secrets.get("GOOGLE_CLOUD_PROJECT", "")
    vertex_location = st.secrets.get("GOOGLE_CLOUD_REGION", "us-central1")

    st.session_state["vertex_project"] = vertex_project
    st.session_state["vertex_location"] = vertex_location

    return yt_key, vertex_project, vertex_location


def _render_secret_status(yt_key: str, vertex_project: str, vertex_location: str):
    missing = []
    if not vertex_project:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if not yt_key:
        missing.append("YOUTUBE_API_KEY")

    if missing:
        st.warning(
            "설정이 부족합니다: "
            + ", ".join(missing)
            + " 값이 `.streamlit/secrets.toml`에 있어야 정상 실행됩니다."
        )


def _short_repr(value, limit: int = 1400) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "..."


def _extract_saved_thumbnail_items() -> list[dict]:
    items = st.session_state.get("saved_items") or []
    if not isinstance(items, list):
        return []
    result = []
    for x in items:
        if not isinstance(x, dict):
            continue
        if x.get("type") == "guideline":
            continue
        if x.get("type") in {"thumbnail", "generated", "creator"} or any(
            k in x for k in ["image", "image_bytes", "prompt", "final_prompt", "generated_prompt", "source_title"]
        ):
            result.append(x)
    return result


def _extract_latest_thumbnail_item() -> dict | None:
    items = _extract_saved_thumbnail_items()
    return items[-1] if items else None


def _extract_generated_image_bytes():
    item = _extract_latest_thumbnail_item()
    if item and item.get("image_bytes"):
        return item.get("image_bytes")

    generated = st.session_state.get("generated_thumb")
    if isinstance(generated, dict):
        if generated.get("bytes"):
            return generated.get("bytes")
        if generated.get("image_bytes"):
            return generated.get("image_bytes")
        img = generated.get("image") or generated.get("img")
        if img is not None:
            try:
                buf = BytesIO()
                img.save(buf, "PNG")
                return buf.getvalue()
            except Exception:
                return None
    return None


def _infer_creator_context() -> tuple[str, str, str, str]:
    item = _extract_latest_thumbnail_item()
    if item:
        return (
            str(item.get("source_title") or item.get("title") or "썸네일 제작 결과"),
            str(item.get("domain") or ""),
            str(item.get("mode") or "개선"),
            str(item.get("prompt") or ""),
        )

    generated = st.session_state.get("generated_thumb")
    if isinstance(generated, dict):
        return (
            str(generated.get("source_title") or generated.get("title") or generated.get("category") or "썸네일 제작 결과"),
            str(generated.get("domain") or ""),
            str(generated.get("mode") or "개선"),
            str(generated.get("prompt") or ""),
        )

    prompt = st.session_state.get("generated_prompt") or st.session_state.get("final_prompt_ko") or st.session_state.get("imp_prompt") or ""
    return "썸네일 제작 결과", "", "개선", str(prompt)


def _build_creator_markdown() -> tuple[str, str]:
    keys = [
        "generated_thumb",
        "generated_prompt",
        "final_prompt_ko",
        "imp_prompt",
        "imp_auto_analysis",
        "thumb_analysis_queue",
        "selected_thumb",
        "current_thumbnail_result",
    ]
    found = {k: st.session_state.get(k) for k in keys if st.session_state.get(k)}
    item = _extract_latest_thumbnail_item()
    if item:
        found["latest_saved_thumbnail_item"] = item

    if not found:
        return "", ""

    source_title, domain, mode, prompt = _infer_creator_context()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_title = (source_title or "썸네일_제작_결과").replace("/", "_").replace("\\", "_")[:40]
    filename = f"{safe_title}_{domain or 'domain'}_썸네일_제작_보고서_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    md = [
        "# 썸네일 제작/개선 결과 보고서",
        "",
        f"> 대상/원본 제목: {source_title}",
        f"> 도메인: {domain or '-'}",
        f"> 제작 방식: {mode or '-'}",
        f"> 생성 시각: {now}",
        "",
        "## 1. 개선/생성 프롬프트",
        prompt or "-",
        "",
        "## 2. 세션 결과 요약",
        "```text",
        _short_repr(found, 5000),
        "```",
        "",
        "## 3. 활용 방법",
        "- PNG 이미지는 다운로드 버튼으로 별도 저장합니다.",
        "- 프롬프트는 이후 유사 썸네일 제작 시 참고합니다.",
        "- 최종 가이드라인 페이지에는 자동 반영되지 않습니다.",
    ]
    return "\n".join(md), filename


def _render_creator_downloads():
    md, md_filename = _build_creator_markdown()
    img_bytes = _extract_generated_image_bytes()

    if not md and not img_bytes:
        return

    st.markdown(
        """
        <div class="creator-download-box">
            <div style="font-size:16px;font-weight:950;color:#111827;margin-bottom:4px;">
                제작 결과 다운로드
            </div>
            <div style="font-size:12.5px;color:#6b7280;line-height:1.6;word-break:keep-all;">
                최근 생성/저장된 썸네일 결과를 직접 다운로드합니다.
                앱 내부 saved 폴더에는 자동 저장하지 않습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([0.34, 0.33, 0.33], gap="medium")
    if img_bytes:
        with cols[0]:
            st.download_button(
                "PNG 이미지 다운로드",
                data=img_bytes,
                file_name=f"thumbnail_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                mime="image/png",
                use_container_width=True,
            )
    if md:
        with cols[1]:
            st.download_button(
                "Markdown 보고서 다운로드",
                data=md,
                file_name=md_filename,
                mime="text/markdown",
                use_container_width=True,
            )
        with cols[2]:
            with st.expander("보고서 미리보기", expanded=False):
                st.markdown(md)


def render_thumbnail_creator_tab():
    _inject_thumbnail_creator_css()
    _render_header()

    if IMPORT_ERROR is not None:
        st.error("썸네일 제작 원본 파일을 불러오지 못했습니다.")
        st.exception(IMPORT_ERROR)
        return

    init_session_state()

    yt_key, vertex_project, vertex_location = _prepare_secrets()
    _render_secret_status(yt_key, vertex_project, vertex_location)

    FNB, IT = get_bench_data()
    render_original_thumbnail_creator(FNB, IT, yt_key)

    # 자동 saved/thumbnail_creator 저장은 하지 않습니다.
