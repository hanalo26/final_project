import streamlit as st

# ============================================================
# 기업 유튜브 성공 전략 대시보드 - Main App
# 위치 권장: works/Hyeong_Uk/test_dashboard/app_test_dashboard.py
# ============================================================

# -----------------------------
# 0. 페이지 import
# -----------------------------
# 아직 리팩토링되지 않은 탭 파일은 없어도 앱이 실행되도록 placeholder를 둡니다.

def _missing_tab(tab_name: str, module_name: str, function_name: str):
    def _render():
        st.markdown(
            f"""
            <div class="page-card">
                <div style="font-size:22px;font-weight:900;color:#111827;margin-bottom:8px;">
                    🚧 {tab_name}
                </div>
                <div style="font-size:14px;color:#6b7280;line-height:1.7;">
                    아직 <b>{module_name}.py</b> 파일의 <b>{function_name}()</b> 함수가 연결되지 않았습니다.<br>
                    해당 탭 파일을 <code>works/Hyeong_Uk/test_dashboard</code> 폴더에 추가하면 이 영역에 표시됩니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return _render

try:
    from landing_tab import render_landing_page
except Exception:
    render_landing_page = _missing_tab("랜딩", "landing_tab", "render_landing_page")

try:
    from longform_dashboard_tab import render_longform_dashboard
except Exception:
    render_longform_dashboard = _missing_tab("롱폼 분석 결과", "longform_dashboard_tab", "render_longform_dashboard")

# try:
#     from longform_agent_tab import render_longform_agent
# except Exception:
#     render_longform_agent = _missing_tab("롱폼 분석 agent", "longform_agent_tab", "render_longform_agent")

try:
    from longform_agent_tab import render_longform_agent_tab
except Exception:
    render_longform_agent_tab = _missing_tab(
        "롱폼 분석 agent",
        "longform_agent_tab",
        "render_longform_agent_tab",
    )

try:
    from thumbnail_dashboard_tab import render_thumbnail_dashboard
except Exception:
    render_thumbnail_dashboard = _missing_tab("썸네일 분석 결과", "thumbnail_dashboard_tab", "render_thumbnail_dashboard")

# try:
#     from thumbnail_agent_tab import render_thumbnail_agent
# except Exception:
#     render_thumbnail_agent = _missing_tab("썸네일 분석 agent", "thumbnail_agent_tab", "render_thumbnail_agent")

# try:
#     from thumbnail_creator_tab import render_thumbnail_creator
# except Exception:
#     render_thumbnail_creator = _missing_tab("썸네일 제작", "thumbnail_creator_tab", "render_thumbnail_creator")

try:
    from thumbnail_agent_tab import render_thumbnail_agent_tab
except Exception:
    render_thumbnail_agent_tab = _missing_tab(
        "썸네일 분석 agent",
        "thumbnail_agent_tab",
        "render_thumbnail_agent_tab"
    )

try:
    from thumbnail_creator_tab import render_thumbnail_creator_tab
except Exception:
    render_thumbnail_creator_tab = _missing_tab(
        "썸네일 제작",
        "thumbnail_creator_tab",
        "render_thumbnail_creator_tab"
    )

try:
    from shorts_dashboard_tab import render_shorts_dashboard
except Exception:
    render_shorts_dashboard = _missing_tab("숏츠 분석 결과", "shorts_dashboard_tab", "render_shorts_dashboard")

# try:
#     from shorts_agent_tab import render_shorts_agent
# except Exception:
#     render_shorts_agent = _missing_tab("숏츠 분석 agent", "shorts_agent_tab", "render_shorts_agent")

try:
    from shorts_agent_tab import render_shorts_agent_tab
except Exception:
    render_shorts_agent_tab = _missing_tab(
        "숏츠 분석 agent",
        "shorts_agent_tab",
        "render_shorts_agent_tab"
    )

try:
    from comment_dashboard_tab import render_comment_dashboard
except Exception:
    def render_comment_dashboard(mode: str = "longform"):
        name = "롱폼 댓글 분석" if mode == "longform" else "숏츠 댓글 분석"
        _missing_tab(name, "comment_dashboard_tab", "render_comment_dashboard")()

try:
    from shorts_comment_dashboard_tab import render_shorts_comment_dashboard
except Exception:
    render_shorts_comment_dashboard = _missing_tab("숏츠 댓글 분석", "shorts_comment_dashboard_tab", "render_shorts_comment_dashboard")

try:
    from guideline_tab import render_guideline_page
except Exception:
    render_guideline_page = _missing_tab("가이드라인", "guideline_tab", "render_guideline_page")


# -----------------------------
# 1. Streamlit 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기업 유튜브 성공 전략 대시보드",
    page_icon="▶️",
    layout="wide",
)


# -----------------------------
# 2. 공통 CSS
# -----------------------------
st.markdown(
    """
<style>
.stApp {
    background: #f8f9fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.85rem;
}

section[data-testid="stSidebar"] {
    background: #f1f3f6;
    border-right: 1px solid #e5e7eb;
}

/* 공통 상단 헤더 */
.main-app-header {
    background: #ffffff;
    border: 1px solid #e9edf3;
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    margin-bottom: 18px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

.main-app-title {
    font-size: 28px;
    font-weight: 900;
    color: #111827;
    letter-spacing: -0.7px;
    margin-bottom: 6px;
}

.main-app-subtitle {
    font-size: 14px;
    color: #6b7280;
    line-height: 1.5;
    text-align: left;
}

.page-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}

.yt-red {
    color: #ef233c;
}

/* 상단 탭 디자인 */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #e5e7eb;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding: 8px 14px;
    border-radius: 999px 999px 0 0;
    color: #6b7280;
    font-size: 14px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: #ef233c !important;
    border-bottom: 3px solid #ef233c;
    background: #fff1f2;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ef233c;
    background: #fff5f6;
}

/* Sidebar */
.side-title {
    font-size: 17px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 4px;
    letter-spacing: -0.3px;
}

.side-caption {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 18px;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 44px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background: transparent !important;
    color: #4b5563 !important;
    border: 0 !important;
    border-radius: 12px !important;
    padding: 11px 14px !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    box-shadow: none !important;
    margin-bottom: 6px !important;
}

section[data-testid="stSidebar"] .stButton > button div,
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button span {
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
    align-items: center !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #fee2e2 !important;
    color: #ef233c !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #fee2e2 !important;
    color: #ef233c !important;
    border-right: 4px solid #ef233c !important;
    font-weight: 900 !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] div,
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton > button[kind="primary"] span {
    color: #ef233c !important;
    font-weight: 900 !important;
}

/* Sidebar Info Card */
.sidebar-info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 16px 16px 18px 16px;
    margin-top: 14px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
}

.sidebar-info-visual {
    height: 86px;
    border-radius: 16px;
    background:
        radial-gradient(circle at 78% 22%, rgba(239, 35, 60, 0.20) 0%, rgba(239, 35, 60, 0.00) 32%),
        linear-gradient(135deg, #fff7f8 0%, #ffffff 100%);
    border: 1px solid #fee2e2;
    position: relative;
    overflow: hidden;
    margin-bottom: 14px;
}

.sidebar-info-visual::before {
    content: "";
    position: absolute;
    left: 18px;
    bottom: 18px;
    width: 92px;
    height: 45px;
    background:
        linear-gradient(to top, rgba(239,35,60,0.18) 0 28%, transparent 28% 100%),
        linear-gradient(90deg, transparent 0 8px, rgba(239,35,60,0.22) 8px 20px, transparent 20px 30px, rgba(239,35,60,0.30) 30px 44px, transparent 44px 54px, rgba(239,35,60,0.42) 54px 70px, transparent 70px);
    border-radius: 8px;
}

.sidebar-info-visual::after {
    content: "↗";
    position: absolute;
    right: 18px;
    top: 13px;
    color: #ef233c;
    font-size: 30px;
    font-weight: 900;
}

.sidebar-info-title {
    font-size: 14px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 7px;
    letter-spacing: -0.2px;
}

.sidebar-info-desc {
    font-size: 12px;
    line-height: 1.65;
    color: #4b5563;
    word-break: keep-all;
}

.sidebar-mini-list {
    margin-top: 12px;
    display: grid;
    gap: 6px;
}

.sidebar-mini-item {
    background: #f9fafb;
    border: 1px solid #eef2f7;
    border-radius: 10px;
    padding: 7px 9px;
    font-size: 11.5px;
    font-weight: 800;
    color: #374151;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# 3. 페이지 URL / 세션 상태 설정
# -----------------------------
PAGE_SLUG_MAP = {
    "landing": "Home",
    "longform": "롱폼 분석",
    "thumbnail": "썸네일 분석",
    "shorts": "숏츠 분석",
    "guideline": "가이드라인",
}

PAGE_TO_SLUG = {value: key for key, value in PAGE_SLUG_MAP.items()}

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

query_page = st.query_params.get("page")
if query_page in PAGE_SLUG_MAP:
    st.session_state["current_page"] = PAGE_SLUG_MAP[query_page]


# -----------------------------
# 4. 사이드바
# -----------------------------
def sidebar_button(label: str, icon: str, page_name: str):
    active = st.session_state["current_page"] == page_name

    clicked = st.button(
        f"{icon}  {label}",
        key=f"nav_{page_name}",
        use_container_width=True,
        type="primary" if active else "secondary",
    )

    if clicked:
        st.session_state["current_page"] = page_name
        st.query_params["page"] = PAGE_TO_SLUG[page_name]
        st.rerun()


def render_sidebar_info(page: str):
    info = {
        "Home": {
            "title": "데이터 기반 인사이트",
            "desc": "프로젝트 목적과 분석 흐름을 먼저 확인하고, 각 분석 페이지로 이동해 세부 결과를 살펴보세요.",
            "items": ["📌 프로젝트 개요", "🧩 분석 모듈 소개", "📘 최종 가이드라인 흐름"],
        },
        "롱폼 분석": {
            "title": "롱폼 분석 활용 방법",
            "desc": "롱폼 영상의 성과, 콘텐츠 구조, agent 진단, 댓글 반응을 함께 확인해 장기 콘텐츠 운영 전략을 정리하세요.",
            "items": ["📊 롱폼 분석 결과", " 롱폼 agent", "💬 롱폼 댓글 분석"],
        },
        "썸네일 분석": {
            "title": "썸네일 분석 활용 방법",
            "desc": "썸네일의 인물, 텍스트, 색감, 브랜드 노출 요소를 확인하고 클릭을 유도하는 시각 전략을 정리하세요.",
            "items": ["🖼️ 썸네일 분석 결과", " 썸네일 agent", "🎨 썸네일 제작/개선"],
        },
        "숏츠 분석": {
            "title": "숏츠 분석 활용 방법",
            "desc": "숏츠 영상 구성 요소와 댓글 반응, 최근 채널 진단 결과를 바탕으로 짧은 영상 운영 전략을 정리하세요.",
            "items": ["🎬 숏츠 분석 결과", " 숏츠 agent", "💬 숏츠 댓글 분석"],
        },
        "가이드라인": {
            "title": "가이드라인 활용 방법",
            "desc": "롱폼, 숏츠, 썸네일, 댓글 분석 결과에서 도출한 공통 패턴을 도메인별 운영 가이드라인으로 정리합니다.",
            "items": ["📘 도메인 선택", "📊 분석 기반 요약"],
        },
    }.get(page, {
        "title": "대시보드 활용 방법",
        "desc": "좌측 메뉴에서 분석 페이지를 선택해 세부 결과를 확인하세요.",
        "items": ["📌 분석 결과", " Agent", "📘 가이드라인"],
    })

    items_html = "".join([f'<div class="sidebar-mini-item">{item}</div>' for item in info["items"]])
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="sidebar-info-visual"></div>
            <div class="sidebar-info-title">{info["title"]}</div>
            <div class="sidebar-info-desc">{info["desc"]}</div>
            <div class="sidebar-mini-list">{items_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown(
        """
        <div class="side-title">▶ 기업 유튜브 성공 전략</div>
        <div class="side-caption">YouTube Strategy Dashboard</div>
        """,
        unsafe_allow_html=True,
    )

    sidebar_button("Home", "🏠", "Home")
    sidebar_button("롱폼 분석", "📺", "롱폼 분석")
    sidebar_button("썸네일 분석", "🖼️", "썸네일 분석")
    sidebar_button("숏츠 분석", "🎬", "숏츠 분석")
    sidebar_button("가이드라인", "📘", "가이드라인")

    st.markdown("---")
    render_sidebar_info(st.session_state["current_page"])


# -----------------------------
# 5. 공통 페이지 헤더
# -----------------------------
def render_page_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="main-app-header">
            <div class="main-app-title">{title}</div>
            <div class="main-app-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 6. 페이지 라우팅
# -----------------------------
page = st.session_state["current_page"]

if page == "Home":
    render_landing_page()

elif page == "롱폼 분석":
    render_page_header(
        "롱폼 분석",
        "롱폼 영상 성과, 콘텐츠 구조, agent 진단, 댓글 반응을 확인합니다.",
    )

    tab1, tab2 = st.tabs([
        "📊 롱폼 분석 결과",
        "🤖 롱폼 분석 agent",
    ])

    with tab1:
        render_longform_dashboard()

    with tab2:
        render_longform_agent_tab()


elif page == "썸네일 분석":
    render_page_header(
        "썸네일 분석",
        "기업 유튜브 썸네일의 클릭 유도 요소, 시각 구성, 분석 agent, 제작 기능을 확인합니다.",
    )

    tab1, tab2, tab3 = st.tabs([
        "🖼️ 썸네일 분석 결과",
        "🤖 썸네일 분석 agent",
        "🎨 썸네일 제작",
    ])

    with tab1:
        render_thumbnail_dashboard()

    with tab2:
        render_thumbnail_agent_tab()

    with tab3:
        render_thumbnail_creator_tab()

elif page == "숏츠 분석":
    render_page_header(
        "숏츠 분석",
        "숏츠 영상 성과, 영상 구성 요소, agent 진단, 댓글 반응을 확인합니다.",
    )

    tab1, tab2 = st.tabs([
        "🎬 숏츠 분석 결과",
        "🤖 숏츠 분석 agent",
    ])

    with tab1:
        render_shorts_dashboard()

    with tab2:
        render_shorts_agent_tab()


elif page == "가이드라인":
    render_guideline_page()
