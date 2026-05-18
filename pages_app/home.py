"""🏠 Trang chủ — sơ đồ kiến trúc Siamese Network (MLP) & rPPG."""
import streamlit as st
from theme import inject_pastel_css, PALETTE

inject_pastel_css()

# ── Hero ───────────────────────────────────────────────────────────
st.markdown("<div class='hero-tag'>🌸 SINH TRẮC HỌC HÀNH VI · SIAMESE NETWORK</div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-title'>DEMO SIAMESE NETWORK (MLP)</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>"
    "Phiên bản Demo cho cuộc thi Ý tưởng đổi mới sáng tạo số<br>"
    "<b>Ngân hàng Nhà nước khu vực 13</b>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;font-size:16px;color:#5B4A55;max-width:780px;margin:auto'>"
    "Giải pháp <b>Sinh trắc học hành vi</b> sử dụng <b>Siamese Network (MLP)</b> "
    "làm kiến trúc trung tâm, học «<b>chữ ký gõ phím</b>» riêng của từng khách hàng — "
    "kết hợp công nghệ <b>rPPG</b> phát hiện sự sống để chống <b>DeepFake</b> "
    "và rủi ro <b>chiếm đoạt tài khoản (Account Takeover)</b>."
    "</p>",
    unsafe_allow_html=True,
)

st.write("")

# ── Sơ đồ kiến trúc Siamese Network (MLP) + rPPG ──────────────────
st.markdown("### 🗺️ Kiến trúc giải pháp")

DIAGRAM = """
digraph G {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Segoe UI",
          fontsize=12, fontcolor="#5B4A55", penwidth=2];
    edge [color="#E48BA7", penwidth=2, fontname="Segoe UI", fontcolor="#8C7785"];

    customer  [label="👤\\nKHÁCH HÀNG\\nđăng nhập / giao dịch",
               fillcolor="#FFEAF1", color="#E48BA7"];

    subgraph cluster_main {
        label="GIẢI PHÁP CHÍNH";
        labelloc="t"; fontsize=13; fontcolor="#A85070";
        style="rounded,filled"; fillcolor="#FFFAFC"; color="#F4D7E1";

        capture [label="⌨️\\nThu thập tín hiệu\\n(áp lực, gyro, dwell, flight)",
                 fillcolor="#FFF3E0", color="#F5C77E"];
        siamese [label="🧠 SIAMESE NETWORK (MLP)\\nHọc «chữ ký gõ phím»\\ncủa từng khách hàng",
                 fillcolor="#D7C4F2", color="#A993E0"];
        verify  [label="🔐\\nSo khớp embedding\\n(cosine similarity)",
                 fillcolor="#E6E0FA", color="#A993E0"];
    }

    subgraph cluster_ext {
        label="MỞ RỘNG";
        labelloc="t"; fontsize=12; fontcolor="#7FCBA9";
        style="rounded,dashed,filled"; fillcolor="#F4FCF7"; color="#BFE7D6";

        rppg [label="🫀 rPPG\\nChống DeepFake\\n(POS + Multi-ROI)",
              fillcolor="#BFE7D6", color="#7FCBA9"];
    }

    decision  [label="🛡️ KẾT LUẬN AI\\nAPPROVED / REVIEW / BLOCKED",
               shape=diamond, fillcolor="#C7E3F5", color="#7FB6DC"];
    bank      [label="🏦\\nHỆ THỐNG\\nCORE BANKING",
               fillcolor="#FFF7E6", color="#F5C77E"];

    customer -> capture [label="thao tác"];
    capture  -> siamese;
    siamese  -> verify [label="embedding"];
    verify   -> decision;
    verify   -> rppg [style=dashed, label="khi cần eKYC"];
    rppg     -> decision [style=dashed];
    decision -> bank;
}
"""
st.graphviz_chart(DIAGRAM, use_container_width=True)

st.caption(
    "💡 *Siamese Network (MLP) là kiến trúc trung tâm: học và so khớp «chữ ký hành vi» "
    "của khách hàng — phát hiện kẻ mạo danh dù biết mật khẩu/OTP. "
    "rPPG là lớp mở rộng cho luồng eKYC: phát hiện DeepFake/ảnh tĩnh bằng tín hiệu nhịp tim qua da.*"
)

st.write("")

# ── Cards giới thiệu ───────────────────────────────────────────────
st.markdown("### 🎯 Khám phá giải pháp")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
<div class='pastel-card' style='background:linear-gradient(135deg,#F5EFFF 0%,#EBE0FA 100%)'>
<h3>🧠 Siamese Network (MLP)</h3>
<b>Sinh trắc học hành vi — Giải pháp chính</b><br>
<span style='color:#8C7785;font-size:13px'>Siamese Network + MLP encoder</span><br><br>
Học «<b>chữ ký gõ phím</b>» riêng của từng khách hàng từ áp lực ngón tay,
tốc độ chạm, cảm biến gyro… Phát hiện kẻ mạo danh dù biết mật khẩu/OTP.
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Khám phá Siamese Network →", key="goto_siamese", use_container_width=True):
        st.switch_page("pages_app/tier2.py")

with c2:
    st.markdown(
        f"""
<div class='pastel-card' style='background:linear-gradient(135deg,#E6F7EF 0%,#D6F0E2 100%)'>
<h3>🫀 rPPG Anti-DeepFake</h3>
<b>Phát hiện sự sống — Mở rộng</b><br>
<span style='color:#8C7785;font-size:13px'>POS Algorithm + Multi-ROI</span><br><br>
Phân tích dao động vi tế của kênh màu trên da mặt để tái lập tín hiệu nhịp tim.
DeepFake và ảnh tĩnh thiếu dòng máu thật nên phổ tần số phẳng — bị chặn ngay tức thì.
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Khám phá rPPG →", key="goto_rppg", use_container_width=True):
        st.switch_page("pages_app/tier3.py")

with c3:
    st.markdown(
        f"""
<div class='pastel-card' style='background:linear-gradient(135deg,#FFF1E6 0%,#FFE6D6 100%)'>
<h3>💳 AI hỗ trợ Demo</h3>
<b>Toolbox bổ trợ — Mở rộng</b><br>
<span style='color:#8C7785;font-size:13px'>LightGBM · SHAP · Gemini</span><br><br>
Bộ công cụ AI hỗ trợ demo Siamese Network: phân tích giao dịch chuyển tiền,
chấm điểm rủi ro <b>0–100%</b> kèm <b>SHAP</b> giải thích và chatbot Gemini đối thoại.
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Khám phá AI hỗ trợ →", key="goto_ai_support", use_container_width=True):
        st.switch_page("pages_app/tier1.py")

st.write("")
