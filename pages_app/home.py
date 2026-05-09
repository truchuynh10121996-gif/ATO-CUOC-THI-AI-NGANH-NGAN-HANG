"""🏠 Trang chủ — sơ đồ kiến trúc 3 lớp phòng vệ."""
import streamlit as st
from theme import inject_pastel_css, PALETTE

inject_pastel_css()

# ── Hero ───────────────────────────────────────────────────────────
st.markdown("<div class='hero-tag'>🌸 BANKING FRAUD &amp; ATO PREVENTION</div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-title'>GIẢI PHÁP FRAUD DETECTION ATO</div>",
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
    "Giải pháp <b>3 lớp phòng vệ</b> kết hợp Machine Learning, sinh trắc học hành vi "
    "và công nghệ phát hiện sự sống <i>(rPPG)</i> — bảo vệ khách hàng trước rủi ro "
    "<b>chiếm đoạt tài khoản (Account Takeover)</b> và <b>DeepFake</b>."
    "</p>",
    unsafe_allow_html=True,
)

st.write("")

# ── Sơ đồ kiến trúc 3 lớp ──────────────────────────────────────────
st.markdown("### 🗺️ Kiến trúc 3 lớp phòng vệ")

DIAGRAM = """
digraph G {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Segoe UI",
          fontsize=12, fontcolor="#5B4A55", penwidth=2];
    edge [color="#E48BA7", penwidth=2, fontname="Segoe UI", fontcolor="#8C7785"];

    customer  [label="👤\\nKHÁCH HÀNG\\nthực hiện giao dịch",
               fillcolor="#FFEAF1", color="#E48BA7"];

    subgraph cluster_layers {
        label="3 LỚP PHÒNG VỆ AI";
        labelloc="t"; fontsize=13; fontcolor="#A85070";
        style="rounded,filled"; fillcolor="#FFFAFC"; color="#F4D7E1";

        tier1 [label="💳 TẦNG 1\\nPhát hiện giao dịch Fraud\\n(LightGBM)",
               fillcolor="#FFD6B8", color="#F5A678"];
        tier2 [label="🧠 TẦNG 2\\nSinh trắc học hành vi\\n(Siamese + MLP)",
               fillcolor="#D7C4F2", color="#A993E0"];
        tier3 [label="🫀 TẦNG 3\\nChống DeepFake bằng rPPG\\n(POS + Multi-ROI)",
               fillcolor="#BFE7D6", color="#7FCBA9"];
    }

    decision  [label="🛡️ KẾT LUẬN AI\\nAPPROVED / REVIEW / BLOCKED",
               shape=diamond, fillcolor="#C7E3F5", color="#7FB6DC"];
    bank      [label="🏦\\nHỆ THỐNG\\nCORE BANKING",
               fillcolor="#FFF7E6", color="#F5C77E"];

    customer -> tier1 [label="giao dịch"];
    tier1    -> tier2 [label="nếu nghi ngờ"];
    tier2    -> tier3 [label="nếu cần eKYC"];
    tier3    -> decision;
    tier1    -> decision [style=dashed, label="approve nhanh"];
    tier2    -> decision [style=dashed];
    decision -> bank;
}
"""
st.graphviz_chart(DIAGRAM, use_container_width=True)

st.caption(
    "💡 *Mỗi tầng là một câu hỏi độc lập: «Giao dịch này có lạ không?» → "
    "«Có đúng người chủ tài khoản đang thao tác không?» → «Đây có phải mặt người thật không?». "
    "Chỉ khi cả 3 đều OK, giao dịch mới được duyệt tự động.*"
)

st.write("")

# ── Cards 3 tầng ───────────────────────────────────────────────────
st.markdown("### 🎯 Khám phá từng tầng")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
<div class='pastel-card' style='background:linear-gradient(135deg,#FFF1E6 0%,#FFE6D6 100%)'>
<h3>💳 Tầng 1</h3>
<b>Phát hiện giao dịch Fraud ATO</b><br>
<span style='color:#8C7785;font-size:13px'>Model: LightGBM</span><br><br>
Phân tích hành vi <b>giao dịch</b>: tần suất, số tiền, người nhận mới, thiết bị, thời gian.
Mỗi giao dịch được chấm điểm rủi ro 0–100% kèm <b>SHAP</b> giải thích lý do.
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Khám phá Tầng 1 →", key="goto_t1", use_container_width=True):
        st.switch_page("pages_app/tier1.py")

with c2:
    st.markdown(
        f"""
<div class='pastel-card' style='background:linear-gradient(135deg,#F5EFFF 0%,#EBE0FA 100%)'>
<h3>🧠 Tầng 2</h3>
<b>Sinh trắc học hành vi</b><br>
<span style='color:#8C7785;font-size:13px'>Siamese Network + MLP</span><br><br>
Học «<b>chữ ký gõ phím</b>» riêng của từng khách hàng từ áp lực ngón tay,
tốc độ chạm, cảm biến gyro… Phát hiện kẻ mạo danh dù biết mật khẩu/OTP.
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Khám phá Tầng 2 →", key="goto_t2", use_container_width=True):
        st.switch_page("pages_app/tier2.py")

with c3:
    st.markdown(
        f"""
<div class='pastel-card' style='background:linear-gradient(135deg,#E6F7EF 0%,#D6F0E2 100%)'>
<h3>🫀 Tầng 3</h3>
<b>Chống DeepFake bằng rPPG</b><br>
<span style='color:#8C7785;font-size:13px'>POS Algorithm + Multi-ROI</span><br><br>
Phát hiện <b>nhịp tim qua da mặt</b> từ camera điện thoại trong eKYC.
DeepFake không có tín hiệu sự sống → chặn ngay.
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Khám phá Tầng 3 →", key="goto_t3", use_container_width=True):
        st.switch_page("pages_app/tier3.py")

st.write("")
st.divider()

# ── Highlights ─────────────────────────────────────────────────────
st.markdown("### ✨ Điểm sáng giải pháp")
h1, h2, h3, h4 = st.columns(4)
with h1:
    st.metric("Lớp phòng vệ", "3", help="3 model AI độc lập, tăng độ tin cậy")
with h2:
    st.metric("Giải thích AI", "SHAP + Gemini", help="Mọi quyết định đều giải thích được")
with h3:
    st.metric("Chặn DeepFake", "POS + HR", help="Phát hiện cả face-swap")
with h4:
    st.metric("Triển khai", "Realtime", help="Đáp ứng dưới 200ms/giao dịch")

st.write("")
st.info(
    "🎓 **Hướng dẫn cho ban giám khảo:** Sử dụng menu bên trái để duyệt qua 3 tầng. "
    "Mỗi tầng có 2–3 tab con: huấn luyện, demo trực tiếp, và phân tích AI bằng tiếng Việt."
)
