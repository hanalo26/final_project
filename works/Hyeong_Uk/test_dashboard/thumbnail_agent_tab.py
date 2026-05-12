import streamlit as st
from datetime import datetime

# 팀원 원본 파일명 기준:
# - thunbnail_analysis_v1.py
# - utills_thumbnail.py
#
# 주의: 파일명에 오타처럼 보이는 thunbnail / utills가 있지만,
# 현재 팀원 코드의 import 기준이므로 이름을 그대로 유지합니다.

try:
    from thunbnail_analysis_v1 import render as render_original_thumbnail_agent
    from thunbnail_analysis_v1 import get_bench_data
    from utills_thumbnail import init_session_state
except Exception as e:
    render_original_thumbnail_agent = None
    get_bench_data = None
    init_session_state = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


def _inject_thumbnail_component_css():
    """원본 컴포넌트에서 사용하는 클래스만 최소 스타일링합니다.

    utills_thumbnail.inject_css()는 사이드바/탭/버튼 전체 스타일을 바꿔서
    app_test_dashboard.py의 공통 UI와 충돌할 수 있으므로 여기서는 호출하지 않습니다.
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

        .badge-yellow {
            background: #fefce8;
            color: #a16207;
            border: 1px solid #fde68a;
        }

        .badge-gray {
            background: #f9fafb;
            color: #6b7280;
            border: 1px solid #e5e7eb;
        }

        .badge-orange {
            background: #fff7ed;
            color: #ea580c;
            border: 1px solid #fed7aa;
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

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 16px;
        }

        .kpi-box {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px 10px;
            text-align: center;
        }

        .kpi-val {
            font-size: 19px;
            font-weight: 950;
            margin-bottom: 3px;
        }

        .kpi-lbl {
            font-size: 11px;
            color: #6b7280;
        }

        .strategy-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 9px 0;
            border-bottom: 1px solid #f3f4f6;
        }

        .strategy-num {
            width: 22px;
            height: 22px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 950;
            color: #fff;
            flex-shrink: 0;
            margin-top: 1px;
        }

        .analysis-modal {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 18px;
            margin-top: 12px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.055);
        }

        .longform-badge {
            background: #eff6ff;
            color: #2563eb;
            border: 1px solid #bfdbfe;
            padding: 2px 8px;
            border-radius: 7px;
            font-size: 11px;
            font-weight: 850;
        }

        .shortform-badge {
            background: #fff1f2;
            color: #ef233c;
            border: 1px solid #fecdd3;
            padding: 2px 8px;
            border-radius: 7px;
            font-size: 11px;
            font-weight: 850;
        }

        .download-guide-box {
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
                썸네일 전략 agent
            </div>
            <div style="font-size:14px;color:#6b7280;line-height:1.65;word-break:keep-all;">
                기업 유튜브 채널을 검색해 최근 롱폼 썸네일을 업종 기준과 비교하고,
                채널 단위 맞춤형 썸네일 전략 보고서를 생성합니다.
                결과는 앱 내부 저장소로 자동 저장하지 않고 필요한 경우 Markdown으로 직접 다운로드합니다.
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
    if not yt_key:
        missing.append("YOUTUBE_API_KEY")
    if not vertex_project:
        missing.append("GOOGLE_CLOUD_PROJECT")

    if missing:
        st.warning(
            "설정이 부족합니다: "
            + ", ".join(missing)
            + " 값이 `.streamlit/secrets.toml`에 있어야 정상 실행됩니다."
        )


def _short_text(value, limit: int = 1200) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "..."


def _latest_guideline_item() -> dict | None:
    items = st.session_state.get("saved_items") or []
    if not isinstance(items, list):
        return None
    guide_items = [x for x in items if isinstance(x, dict) and x.get("type") == "guideline"]
    return guide_items[-1] if guide_items else None


def _active_channel_context() -> tuple[str, str, dict, str]:
    """현재 화면에서 가장 최근/선택된 채널 분석 결과를 가져옵니다."""
    item = _latest_guideline_item()
    if item:
        domain = str(item.get("domain") or "")
        channel = str(item.get("channel") or item.get("channel_name") or item.get("title") or "썸네일 전략 보고서")
        summary = str(item.get("summary") or item.get("text") or item.get("content") or "")
        return domain, channel, item, summary

    # 원본 컴포넌트가 세션에 남기는 채널 단위 결과
    selected_domain = "FnB" if st.session_state.get("guide_domain_sel", "🍔 FnB").startswith("🍔") else "IT"
    if selected_domain == "FnB" and st.session_state.get("fnb_channel"):
        ch = st.session_state.get("fnb_channel") or {}
        ana = st.session_state.get("fnb_analysis") or {}
        channel_name = ch.get("name") or ch.get("title") or "FnB 채널"
        summary = st.session_state.get("fnb_guideline") or ""
        return "FnB", channel_name, {"channel": ch, "analysis": ana, "guideline": summary}, summary

    if selected_domain == "IT" and st.session_state.get("it_channel"):
        ch = st.session_state.get("it_channel") or {}
        ana = st.session_state.get("it_analysis") or {}
        channel_name = ch.get("name") or ch.get("title") or "IT 채널"
        summary = st.session_state.get("it_guideline") or ""
        return "IT", channel_name, {"channel": ch, "analysis": ana, "guideline": summary}, summary

    return "", "", {}, ""


def _make_thumbnail_strategy_report() -> tuple[str, str]:
    domain, channel_name, payload, summary = _active_channel_context()
    if not payload:
        return "", ""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"{channel_name}_{domain}_썸네일_전략_보고서".replace("/", "_").replace("\\", "_")
    filename = f"{title}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    md = [
        f"# {channel_name} 썸네일 전략 보고서",
        "",
        f"> 도메인: {domain or '-'}",
        f"> 생성 시각: {now}",
        "",
        "## 1. 보고서 요약",
        summary or "채널 단위 썸네일 분석 결과를 바탕으로 작성한 전략 보고서입니다.",
        "",
        "## 2. 원본 분석 결과",
        "```text",
        _short_text(payload, 4500),
        "```",
        "",
        "## 3. 활용 방법",
        "- 채널의 롱폼 썸네일 운영 기준을 점검합니다.",
        "- 썸네일 제작/개선 탭에서 개별 썸네일 개선 시 참고합니다.",
        "- 최종 가이드라인 페이지에는 자동 반영되지 않으며, 별도 다운로드 자료로 활용합니다.",
    ]
    return "\n".join(md), filename


def _render_report_download():
    md, filename = _make_thumbnail_strategy_report()
    if not md:
        return

    st.markdown(
        """
        <div class="download-guide-box">
            <div style="font-size:16px;font-weight:950;color:#111827;margin-bottom:4px;">
                썸네일 전략 보고서 다운로드
            </div>
            <div style="font-size:12.5px;color:#6b7280;line-height:1.6;word-break:keep-all;">
                현재 분석된 채널의 썸네일 전략 결과를 Markdown 파일로 직접 다운로드합니다.
                앱 내부 saved 폴더에는 저장하지 않습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([0.45, 0.55], gap="medium")
    with c1:
        st.download_button(
            "Markdown 보고서 다운로드",
            data=md,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )
    with c2:
        with st.expander("보고서 미리보기", expanded=False):
            st.markdown(md)


def _request_open_thumbnail_creator():
    """썸네일 제작 탭 이동 요청 플래그만 남깁니다.

    app_test_dashboard.py가 st.tabs를 사용하면 자동 전환은 어렵지만,
    제작 탭에서 이 세션 값을 활용할 수 있습니다.
    """
    if st.session_state.get("thumb_analysis_queue"):
        st.session_state["open_thumbnail_creator_tab"] = True
        st.session_state["selected_thumbnail_subtab"] = "썸네일 제작"


def render_thumbnail_agent_tab():
    _inject_thumbnail_component_css()
    _render_header()

    if IMPORT_ERROR is not None:
        st.error("썸네일 분석 agent 원본 파일을 불러오지 못했습니다.")
        st.exception(IMPORT_ERROR)
        return

    init_session_state()

    yt_key, vertex_project, vertex_location = _prepare_secrets()
    _render_secret_status(yt_key, vertex_project, vertex_location)

    FNB, IT = get_bench_data()
    render_original_thumbnail_agent(FNB, IT, yt_key)

    # 자동 saved 저장은 하지 않습니다.
    # 선택 썸네일 상세 분석도 별도 저장소로 넘기지 않습니다.
    _request_open_thumbnail_creator()
