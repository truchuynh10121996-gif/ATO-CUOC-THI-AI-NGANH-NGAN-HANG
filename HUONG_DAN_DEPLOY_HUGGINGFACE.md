# 🚀 Hướng dẫn Deploy lên Hugging Face Spaces (chi tiết cho người mới)

> **Mục tiêu:** Đưa toàn bộ ứng dụng Streamlit này lên web miễn phí, ai có link đều xem được.

---

## ✅ Tổng quan các bước

1. Tạo tài khoản Hugging Face (miễn phí)
2. Lấy `GEMINI_API_KEY` từ Google AI Studio (miễn phí)
3. Tạo một **Space** mới trên Hugging Face
4. Đẩy code lên Space
5. Cấu hình Secret (`GEMINI_API_KEY`)
6. Chờ build xong → app tự chạy

Mọi file cấu hình cần thiết (`README.md`, `packages.txt`, `requirements.txt`) **đã được tôi chuẩn bị sẵn**. Bạn chỉ việc làm theo các bước bên dưới.

---

## Bước 1 — Tạo tài khoản Hugging Face

1. Truy cập: <https://huggingface.co/join>
2. Đăng ký bằng email (hoặc nút "Continue with Google" cho nhanh)
3. Vào email xác nhận tài khoản
4. Đăng nhập → ghi nhớ **username** của bạn (ví dụ: `truchuynh10121996`)

---

## Bước 2 — Lấy GEMINI_API_KEY (miễn phí)

1. Truy cập: <https://aistudio.google.com/app/apikey>
2. Đăng nhập bằng Google
3. Bấm **"Create API key"** → chọn project (hoặc tạo mới)
4. Copy chuỗi key có dạng `AIzaSy...` → lưu vào Notepad tạm

> ⚠️ **Không** commit key này lên Git. Ở bước 5 ta sẽ dán vào ô **Secret** của Hugging Face.

---

## Bước 3 — Tạo Space mới trên Hugging Face

1. Truy cập: <https://huggingface.co/new-space>
2. Điền form:

   | Trường | Giá trị |
   |--------|---------|
   | **Owner** | Username của bạn |
   | **Space name** | `demo-siamese-banking` (tuỳ chọn, viết liền, không dấu) |
   | **License** | `mit` |
   | **Space SDK** | **Streamlit** ⭐ |
   | **Streamlit template** | Để mặc định / Blank |
   | **Space hardware** | **CPU basic — Free** (đủ dùng) |
   | **Public/Private** | **Public** (để Ban giám khảo xem được) |

3. Bấm **Create Space**.
4. Hugging Face sẽ tạo một Git repo trống tại địa chỉ kiểu:

   ```
   https://huggingface.co/spaces/<username>/demo-siamese-banking
   ```

---

## Bước 4 — Đẩy code lên Space

Có **2 cách**. Khuyến nghị **Cách A** (dễ nhất cho người mới).

### 🟢 Cách A — Dùng web UI Hugging Face (kéo-thả file)

1. Vào Space vừa tạo → tab **"Files"** (gần đầu trang).
2. Bấm **"Add file" → "Upload files"**.
3. Kéo-thả **tất cả** file/thư mục trong dự án này (trừ `.git/`, `__pycache__/`) lên:

   - `app.py`
   - `lib_gemini.py`, `lib_lightgbm.py`, `lib_personas.py`, `lib_rppg.py`, `lib_siamese.py`
   - `theme.py`
   - `requirements.txt`
   - `packages.txt`
   - `README.md`
   - `.gitignore`
   - Thư mục `.streamlit/` (giữ nguyên `config.toml`, **KHÔNG upload `secrets.toml`** nếu có)
   - Thư mục `pages_app/` (toàn bộ)

4. Ở ô **Commit message**, ghi: `Initial deploy`.
5. Bấm **Commit changes to main**.

> 💡 Mẹo: Có thể kéo-thả cả thư mục, web UI sẽ giữ nguyên cấu trúc.

### 🔵 Cách B — Dùng Git (cho ai quen dòng lệnh)

```bash
# 1. Tạo Access Token (Write) tại: https://huggingface.co/settings/tokens
#    → Bấm "Create new token" → Role: Write → Copy token (dạng hf_...)

# 2. Clone Space repo về máy
git clone https://huggingface.co/spaces/<username>/demo-siamese-banking
cd demo-siamese-banking

# 3. Copy toàn bộ file dự án vào thư mục này (trừ .git/)
# (Trên Linux/Mac)
rsync -av --exclude='.git' /đường/dẫn/dự-án/ ./

# 4. Push lên
git add .
git commit -m "Initial deploy"
git push
# Khi git hỏi password → DÁN ACCESS TOKEN (hf_...) thay vì mật khẩu HF
```

---

## Bước 5 — Cấu hình Secret `GEMINI_API_KEY`

> Đây là bước **bắt buộc**, nếu thiếu key thì khung chat AI sẽ báo lỗi.

1. Vào Space → tab **Settings** (trên cùng bên phải).
2. Cuộn xuống mục **"Variables and secrets"**.
3. Bấm **"New secret"**.
4. Điền:
   - **Name**: `GEMINI_API_KEY`
   - **Value**: dán chuỗi `AIzaSy...` đã copy ở Bước 2
5. Bấm **Save**.
6. (Tuỳ chọn) Tạo thêm **Variable** (không phải secret):
   - **Name**: `GEMINI_MODEL`
   - **Value**: `gemini-2.5-flash` (chỉ thêm nếu model mặc định bị lỗi quota)

7. Space sẽ **tự động restart** sau khi lưu secret.

---

## Bước 6 — Chờ build xong

1. Vào tab **"App"** của Space.
2. Thanh trạng thái sẽ chạy qua các giai đoạn:
   - 🟡 **Building** (5–15 phút lần đầu, do phải cài tensorflow/lightgbm/opencv)
   - 🟢 **Running**
3. Khi thấy app hiện ra với sidebar "Demo Siamese Network" → **THÀNH CÔNG** 🎉
4. Copy URL Space (dạng `https://<username>-demo-siamese-banking.hf.space`) để chia sẻ.

---

## 🛠️ Khi gặp lỗi — Cách đọc log

1. Vào Space → tab **"Logs"** (cạnh tab App).
2. Đọc log build hoặc log runtime.

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Khắc phục |
|-----|-------------|-----------|
| `ModuleNotFoundError: No module named 'cv2'` | Thiếu apt package | Đảm bảo file `packages.txt` có `libgl1` và `libglib2.0-0` |
| `Build timeout / out of memory` | Free tier RAM 16GB nhưng build tensorflow tốn nhiều | Đổi sang `tensorflow-cpu` (đã làm sẵn trong `requirements.txt`) |
| `GEMINI_API_KEY chưa cấu hình` | Quên thêm secret | Quay lại **Bước 5** |
| `numpy.dtype size changed` | numpy 2.x phá vỡ ABI | Đã pin `numpy<2.0.0` trong `requirements.txt` |
| App stuck ở trạng thái "Building" > 30 phút | Build quá lâu | Restart Space: Settings → "Factory reboot" |

---

## 🔄 Khi cần cập nhật code

- **Cách A (UI)**: Vào tab Files → bấm vào file → "Edit" → Commit.
- **Cách B (Git)**: `git pull && sửa code && git add . && git commit -m "..." && git push`.
- Space sẽ **tự rebuild** sau mỗi commit.

---

## 📊 Thông số hiệu năng dự kiến

- **Build time** lần đầu: ~10–15 phút (do cài tensorflow + lightgbm + opencv)
- **Build time** lần sau (cache): ~2–3 phút
- **Cold start**: ~30 giây
- **Disk size**: ~3–4 GB
- **RAM runtime**: ~1–2 GB (CPU basic Free có 16 GB ⇒ thoải mái)

---

## 🎓 Tài liệu chính thức

- Streamlit on HF Spaces: <https://huggingface.co/docs/hub/spaces-sdks-streamlit>
- Secrets & Variables: <https://huggingface.co/docs/hub/spaces-overview#managing-secrets>
- Hardware: <https://huggingface.co/docs/hub/spaces-gpus>

---

## ✨ Checklist trước khi nộp

- [ ] App đã chạy được tại URL `https://<username>-<space-name>.hf.space`
- [ ] Sidebar hiện đầy đủ các trang: Trang chủ, Siamese, rPPG, End-to-End, AI hỗ trợ, Tác giả
- [ ] Khung chat AI trả lời được câu hỏi (chứng tỏ `GEMINI_API_KEY` ok)
- [ ] Space đặt **Public**
- [ ] README.md hiển thị đúng tiêu đề & emoji 🧠

Chúc bạn deploy thành công! 🚀
