---
title: Demo Siamese Network Sinh Trắc Học Hành Vi
emoji: 🧠
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Demo phát hiện gian lận ngân hàng bằng Siamese Network + rPPG
---

# 🧠 Demo Siamese Network — Sinh trắc học hành vi

Ứng dụng demo cho cuộc thi **Ý tưởng đổi mới sáng tạo số ngành Ngân hàng**
(Ngân hàng Nhà nước Khu vực 13).

## Các tầng giải pháp

- **Tầng 1 — LightGBM Fraud Detection**: phát hiện gian lận giao dịch
  (Account Takeover) bằng mô hình LightGBM + giải thích bằng SHAP.
- **Tầng 2 — Siamese Network (MLP)**: định danh người dùng dựa trên
  sinh trắc học hành vi.
- **Tầng 3 — rPPG**: chống DeepFake bằng tín hiệu remote-PPG.
- **Khung chat Gemini**: trợ lý AI hỗ trợ giải thích kết quả ở cả 3 tầng.

## Cấu hình

Vào **Settings → Variables and secrets** của Space và thêm:

| Tên | Loại | Bắt buộc | Mô tả |
|-----|------|----------|-------|
| `GEMINI_API_KEY` | Secret | ✅ | Lấy tại https://aistudio.google.com/app/apikey |
| `GEMINI_MODEL` | Variable | ❌ | Mặc định `gemini-flash-latest` |

## Chạy local

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Dán GEMINI_API_KEY vào file secrets.toml
streamlit run app.py
```
