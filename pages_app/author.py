"""👤 Tác giả (placeholder)."""
import streamlit as st
from theme import inject_pastel_css

inject_pastel_css()

st.markdown("<div class='hero-tag'>👤 TÁC GIẢ</div>", unsafe_allow_html=True)
st.title("Về tác giả")

st.info("🚧 Trang đang được xây dựng. Sẽ cập nhật thông tin nhóm tác giả, "
        "đơn vị công tác, lời cảm ơn ở phiên bản tiếp theo.")
