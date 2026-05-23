"""Gemini chat helper — dùng chung cho cả 3 tầng.

CÁCH GẮN API KEY:
  - Tạo file .streamlit/secrets.toml với nội dung:
        GEMINI_API_KEY = "your_api_key_here"
  - Hoặc đặt biến môi trường:  export GEMINI_API_KEY="..."

Lấy key tại: https://aistudio.google.com/app/apikey  (miễn phí, có quota).
"""
import os
import time
import streamlit as st

DEFAULT_MODEL = "gemini-flash-latest"


def get_api_key() -> str | None:
    """Lấy API key từ st.secrets hoặc env var."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def get_model_name() -> str:
    """Lấy tên model từ st.secrets / env var, fallback về DEFAULT_MODEL."""
    try:
        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _build_system_prompt(role: str) -> str:
    return (
        "Bạn là trợ lý AI cho ngân hàng Việt Nam, chuyên giải thích kết quả "
        "phát hiện gian lận giao dịch (Account Takeover) cho NHÂN VIÊN nghiệp vụ "
        "và KHÁCH HÀNG. Hãy:\n"
        "1. Trả lời bằng TIẾNG VIỆT, ngôn ngữ phổ thông, dễ hiểu, NGẮN GỌN.\n"
        "2. KHÔNG dùng thuật ngữ Machine Learning (tránh: SHAP, feature, embedding, "
        "logit, threshold...). Nếu phải dùng, hãy giải thích bằng từ thường.\n"
        "3. Tập trung vào *vì sao* giao dịch/hành vi này đáng ngờ và *khuyến nghị "
        "hành động* cho khách hàng/nhân viên.\n"
        "4. Giọng văn thân thiện, chuyên nghiệp, có thể dùng emoji nhẹ.\n"
        f"5. Bối cảnh hiện tại: {role}\n"
    )


def stream_explanation(context: dict, user_question: str, role: str = "Phát hiện gian lận"):
    """
    Generator yield từng chunk text từ Gemini.
    `context` là dict đã chuẩn hoá (feature values, contributions, score...).
    """
    api_key = get_api_key()
    if not api_key:
        yield (
            "⚠️ **Chưa cấu hình GEMINI_API_KEY.**\n\n"
            "Để bật phân tích AI, vui lòng:\n"
            "1. Lấy API key miễn phí tại https://aistudio.google.com/app/apikey\n"
            "2. Tạo file `.streamlit/secrets.toml` với nội dung:\n"
            "```toml\nGEMINI_API_KEY = \"your_api_key\"\n```\n"
            "3. Hoặc đặt biến môi trường `GEMINI_API_KEY=...`\n\n"
            "_Sau đó khởi động lại app._"
        )
        return

    try:
        import google.generativeai as genai
    except ImportError:
        yield (
            "⚠️ Chưa cài thư viện `google-generativeai`.\n\n"
            "Chạy lệnh: `pip install google-generativeai`"
        )
        return

    try:
        genai.configure(api_key=api_key)

        # Đóng gói context thành đoạn văn có cấu trúc cho LLM
        ctx_lines = ["BỐI CẢNH GIAO DỊCH/HÀNH VI ĐÃ ĐƯỢC AI CHẤM:"]
        for k, v in context.items():
            if isinstance(v, (list, tuple)):
                ctx_lines.append(f"- {k}:")
                for item in v:
                    ctx_lines.append(f"    • {item}")
            else:
                ctx_lines.append(f"- {k}: {v}")
        context_block = "\n".join(ctx_lines)

        prompt = (
            _build_system_prompt(role)
            + "\n\n" + context_block
            + "\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n" + user_question
            + "\n\nTRẢ LỜI:"
        )

        model = genai.GenerativeModel(get_model_name())
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"❌ Lỗi gọi Gemini: `{e}`"


def render_chat_panel(
    context: dict,
    role: str,
    key_prefix: str,
    suggested_questions: list[str] | None = None,
    title: str = "💬 Phân tích của chuyên gia AI",
):
    """Vẽ khung chat dưới phần SHAP. Lưu lịch sử trong st.session_state."""
    st.markdown(f"#### {title}")
    st.caption(
        "Hỏi AI để được giải thích bằng ngôn ngữ phổ thông, dễ hiểu cho khách hàng. "
        "AI nhận đầy đủ thông tin giao dịch + mức độ ảnh hưởng từng yếu tố."
    )

    history_key = f"{key_prefix}_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Quick suggestion buttons.
    # Mỗi item có thể là chuỗi (label == prompt) hoặc tuple (label_hiển_thị, prompt_gửi_Gemini).
    if suggested_questions:
        cols = st.columns(len(suggested_questions))
        for i, q in enumerate(suggested_questions):
            if isinstance(q, (tuple, list)) and len(q) == 2:
                label, prompt_text = q
            else:
                label = prompt_text = q
            if cols[i].button(label, key=f"{key_prefix}_suggest_{i}", use_container_width=True):
                st.session_state[f"{key_prefix}_pending"] = prompt_text

    # Show history
    for msg in st.session_state[history_key]:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'><b>Bạn:</b> {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-bot'><b>🤖 Gemini:</b><br>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    # Pending question (from suggestion button)
    pending = st.session_state.pop(f"{key_prefix}_pending", None)

    user_input = st.chat_input("Nhập câu hỏi của bạn...", key=f"{key_prefix}_input")
    question = user_input or pending

    if question:
        st.session_state[history_key].append({"role": "user", "content": question})
        st.markdown(
            f"<div class='chat-user'><b>Bạn:</b> {question}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='chat-bot'><b>🤖 Gemini:</b><br>", unsafe_allow_html=True)
        placeholder = st.empty()
        full_text = ""
        for chunk in stream_explanation(context, question, role=role):
            full_text += chunk
            placeholder.markdown(full_text)
            time.sleep(0.005)
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state[history_key].append({"role": "bot", "content": full_text})
        st.rerun()

    if st.session_state[history_key]:
        if st.button("🗑️ Xoá lịch sử chat", key=f"{key_prefix}_clear"):
            st.session_state[history_key] = []
            st.rerun()
