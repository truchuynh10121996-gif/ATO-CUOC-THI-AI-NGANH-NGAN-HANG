"""
🌸 GIẢI PHÁP FRAUD DETECTION ATO 🌸
Phiên bản Demo cho cuộc thi Ý tưởng đổi mới sáng tạo số
Ngân hàng Nhà nước khu vực 13.

Entry point — sử dụng st.navigation để định tuyến giữa các trang.
"""
import streamlit as st

st.set_page_config(
    page_title="Fraud Detection ATO",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Định nghĩa các trang ─────────────────────────────────────────────
PAGES = {
    "Tổng quan": [
        st.Page("pages_app/home.py",        title="Trang chủ",                     icon="🏠", default=True),
    ],
    "3 Lớp Phòng Vệ": [
        st.Page("pages_app/tier1.py",       title="Tầng 1 — LightGBM Fraud",        icon="💳"),
        st.Page("pages_app/tier2.py",       title="Tầng 2 — Sinh trắc học hành vi", icon="🧠"),
        st.Page("pages_app/tier3.py",       title="Tầng 3 — rPPG chống DeepFake",   icon="🫀"),
    ],
    "Mở rộng": [
        st.Page("pages_app/end_to_end.py",  title="Demo End-to-End",                icon="🔗"),
        st.Page("pages_app/author.py",      title="Tác giả",                        icon="👤"),
    ],
}

pg = st.navigation(PAGES)

# Sidebar branding
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:14px 0 6px'>"
        "<div style='font-size:30px'>🌸</div>"
        "<div style='font-weight:800;font-size:17px;color:#A85070;line-height:1.2'>"
        "Fraud Detection<br>ATO</div>"
        "<div style='font-size:11px;color:#8C7785;margin-top:4px'>"
        "NHNN Khu vực 13 · Demo</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

pg.run()
