"""🔗 Demo End-to-End — kết hợp 3 lớp (placeholder)."""
import streamlit as st
from theme import inject_pastel_css

inject_pastel_css()

st.markdown("<div class='hero-tag'>🔗 END-TO-END</div>", unsafe_allow_html=True)
st.title("Demo End-to-End — Kết hợp 3 lớp phòng vệ")

st.info(
    "🚧 **Trang đang được xây dựng.**  \n"
    "Sau khi 3 tầng hoàn chỉnh, mục này sẽ trình diễn **luồng giao dịch end-to-end**: "
    "Một giao dịch chuyển tiền đi qua Tầng 1 → Tầng 2 → Tầng 3 và tổng hợp quyết định "
    "cuối cùng (APPROVE / OTP / BLOCK / FORCE eKYC)."
)

st.markdown("""
**Dự kiến nội dung:**
- Mô phỏng 3 kịch bản (giao dịch hợp lệ / kẻ chiếm đoạt tài khoản / DeepFake video call)
- Hiển thị từng tầng phản ứng theo thời gian thực
- Sơ đồ flow ra quyết định cuối
- Bảng so sánh False Positive / False Negative khi dùng 1 vs 2 vs 3 tầng
""")
