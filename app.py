import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fraud Detection ATO - Siamese Network",
    page_icon="🔐",
    layout="wide"
)

# ─────────────────────────────────────────────
#  Persona definitions (20 users)
# ─────────────────────────────────────────────
PERSONAS = [
    # (user_id, name, description, base_features dict)
    {"id": "U001", "name": "Nguyễn Văn An",     "desc": "Nhân viên văn phòng, gõ nhẹ nhàng",
     "avg_pressure": 0.35, "std_pressure": 0.03, "avg_touch_area": 120, "std_touch_area": 8,
     "avg_touch_duration": 85,  "std_touch_duration": 6,  "avg_inter_gap": 180, "std_inter_gap": 15,
     "avg_gyro_x": 0.05,  "std_gyro_x": 0.01, "avg_gyro_y": 0.03, "std_gyro_y": 0.01},

    {"id": "U002", "name": "Trần Thị Bích",     "desc": "Sinh viên, nhấn mạnh tay",
     "avg_pressure": 0.65, "std_pressure": 0.05, "avg_touch_area": 160, "std_touch_area": 12,
     "avg_touch_duration": 110, "std_touch_duration": 9,  "avg_inter_gap": 140, "std_inter_gap": 20,
     "avg_gyro_x": 0.12,  "std_gyro_x": 0.03, "avg_gyro_y": 0.08, "std_gyro_y": 0.02},

    {"id": "U003", "name": "Lê Minh Châu",      "desc": "Gõ nhanh, áp lực trung bình",
     "avg_pressure": 0.50, "std_pressure": 0.04, "avg_touch_area": 135, "std_touch_area": 10,
     "avg_touch_duration": 70,  "std_touch_duration": 5,  "avg_inter_gap": 110, "std_inter_gap": 12,
     "avg_gyro_x": 0.08,  "std_gyro_x": 0.02, "avg_gyro_y": 0.05, "std_gyro_y": 0.015},

    {"id": "U004", "name": "Phạm Đức Dũng",     "desc": "Ngón tay to, nhấn chậm",
     "avg_pressure": 0.72, "std_pressure": 0.06, "avg_touch_area": 185, "std_touch_area": 14,
     "avg_touch_duration": 130, "std_touch_duration": 10, "avg_inter_gap": 220, "std_inter_gap": 25,
     "avg_gyro_x": 0.04,  "std_gyro_x": 0.01, "avg_gyro_y": 0.02, "std_gyro_y": 0.01},

    {"id": "U005", "name": "Hoàng Thị Emm",     "desc": "Người cao tuổi, gõ chậm rãi",
     "avg_pressure": 0.40, "std_pressure": 0.05, "avg_touch_area": 145, "std_touch_area": 15,
     "avg_touch_duration": 160, "std_touch_duration": 18, "avg_inter_gap": 300, "std_inter_gap": 40,
     "avg_gyro_x": 0.15,  "std_gyro_x": 0.04, "avg_gyro_y": 0.12, "std_gyro_y": 0.03},

    {"id": "U006", "name": "Vũ Quang Phúc",     "desc": "Game thủ, phản xạ nhanh",
     "avg_pressure": 0.60, "std_pressure": 0.04, "avg_touch_area": 140, "std_touch_area": 9,
     "avg_touch_duration": 55,  "std_touch_duration": 4,  "avg_inter_gap": 90,  "std_inter_gap": 10,
     "avg_gyro_x": 0.20,  "std_gyro_x": 0.05, "avg_gyro_y": 0.15, "std_gyro_y": 0.04},

    {"id": "U007", "name": "Đặng Thị Giang",    "desc": "Nhân viên kế toán, gõ đều đặn",
     "avg_pressure": 0.45, "std_pressure": 0.02, "avg_touch_area": 125, "std_touch_area": 6,
     "avg_touch_duration": 90,  "std_touch_duration": 4,  "avg_inter_gap": 160, "std_inter_gap": 10,
     "avg_gyro_x": 0.06,  "std_gyro_x": 0.01, "avg_gyro_y": 0.04, "std_gyro_y": 0.01},

    {"id": "U008", "name": "Bùi Anh Hùng",      "desc": "Lập trình viên, gõ không nhìn bàn phím",
     "avg_pressure": 0.55, "std_pressure": 0.03, "avg_touch_area": 130, "std_touch_area": 8,
     "avg_touch_duration": 75,  "std_touch_duration": 5,  "avg_inter_gap": 100, "std_inter_gap": 11,
     "avg_gyro_x": 0.07,  "std_gyro_x": 0.02, "avg_gyro_y": 0.06, "std_gyro_y": 0.015},

    {"id": "U009", "name": "Ngô Thị Iris",      "desc": "Bác sĩ, tay nhỏ gọn",
     "avg_pressure": 0.30, "std_pressure": 0.03, "avg_touch_area": 100, "std_touch_area": 7,
     "avg_touch_duration": 80,  "std_touch_duration": 6,  "avg_inter_gap": 170, "std_inter_gap": 14,
     "avg_gyro_x": 0.05,  "std_gyro_x": 0.01, "avg_gyro_y": 0.03, "std_gyro_y": 0.01},

    {"id": "U010", "name": "Trịnh Văn Khoa",    "desc": "Sinh viên, hay dùng 2 tay",
     "avg_pressure": 0.48, "std_pressure": 0.06, "avg_touch_area": 115, "std_touch_area": 11,
     "avg_touch_duration": 65,  "std_touch_duration": 8,  "avg_inter_gap": 120, "std_inter_gap": 18,
     "avg_gyro_x": 0.10,  "std_gyro_x": 0.03, "avg_gyro_y": 0.07, "std_gyro_y": 0.02},

    {"id": "U011", "name": "Lý Thị Lan",        "desc": "Giáo viên, cẩn thận từng phím",
     "avg_pressure": 0.42, "std_pressure": 0.03, "avg_touch_area": 128, "std_touch_area": 7,
     "avg_touch_duration": 105, "std_touch_duration": 7,  "avg_inter_gap": 200, "std_inter_gap": 16,
     "avg_gyro_x": 0.06,  "std_gyro_x": 0.01, "avg_gyro_y": 0.04, "std_gyro_y": 0.01},

    {"id": "U012", "name": "Đinh Văn Mạnh",     "desc": "Tài xế, dùng điện thoại lúc nghỉ",
     "avg_pressure": 0.68, "std_pressure": 0.07, "avg_touch_area": 170, "std_touch_area": 16,
     "avg_touch_duration": 120, "std_touch_duration": 12, "avg_inter_gap": 250, "std_inter_gap": 35,
     "avg_gyro_x": 0.18,  "std_gyro_x": 0.05, "avg_gyro_y": 0.13, "std_gyro_y": 0.04},

    {"id": "U013", "name": "Phan Thị Nga",      "desc": "Người dùng một tay",
     "avg_pressure": 0.52, "std_pressure": 0.05, "avg_touch_area": 138, "std_touch_area": 12,
     "avg_touch_duration": 95,  "std_touch_duration": 8,  "avg_inter_gap": 155, "std_inter_gap": 18,
     "avg_gyro_x": 0.22,  "std_gyro_x": 0.06, "avg_gyro_y": 0.18, "std_gyro_y": 0.05},

    {"id": "U014", "name": "Cao Minh Oanh",     "desc": "Nhà thiết kế, thao tác nhanh nhẹn",
     "avg_pressure": 0.38, "std_pressure": 0.03, "avg_touch_area": 112, "std_touch_area": 8,
     "avg_touch_duration": 60,  "std_touch_duration": 5,  "avg_inter_gap": 95,  "std_inter_gap": 10,
     "avg_gyro_x": 0.09,  "std_gyro_x": 0.02, "avg_gyro_y": 0.06, "std_gyro_y": 0.015},

    {"id": "U015", "name": "Hà Văn Phong",      "desc": "Bảo vệ, ngón tay thô",
     "avg_pressure": 0.78, "std_pressure": 0.07, "avg_touch_area": 200, "std_touch_area": 18,
     "avg_touch_duration": 145, "std_touch_duration": 14, "avg_inter_gap": 280, "std_inter_gap": 38,
     "avg_gyro_x": 0.03,  "std_gyro_x": 0.01, "avg_gyro_y": 0.02, "std_gyro_y": 0.01},

    {"id": "U016", "name": "Trương Thị Quỳnh",  "desc": "Nhân viên ngân hàng, thao tác chính xác",
     "avg_pressure": 0.46, "std_pressure": 0.02, "avg_touch_area": 122, "std_touch_area": 5,
     "avg_touch_duration": 88,  "std_touch_duration": 4,  "avg_inter_gap": 165, "std_inter_gap": 9,
     "avg_gyro_x": 0.05,  "std_gyro_x": 0.01, "avg_gyro_y": 0.03, "std_gyro_y": 0.01},

    {"id": "U017", "name": "Mai Đình Rồng",     "desc": "Học sinh cấp 3, gõ bằng ngón cái",
     "avg_pressure": 0.58, "std_pressure": 0.05, "avg_touch_area": 148, "std_touch_area": 13,
     "avg_touch_duration": 72,  "std_touch_duration": 7,  "avg_inter_gap": 115, "std_inter_gap": 15,
     "avg_gyro_x": 0.14,  "std_gyro_x": 0.04, "avg_gyro_y": 0.10, "std_gyro_y": 0.03},

    {"id": "U018", "name": "Lưu Thị Sen",       "desc": "Nội trợ, thao tác chậm và cẩn thận",
     "avg_pressure": 0.43, "std_pressure": 0.04, "avg_touch_area": 132, "std_touch_area": 10,
     "avg_touch_duration": 118, "std_touch_duration": 11, "avg_inter_gap": 230, "std_inter_gap": 28,
     "avg_gyro_x": 0.11,  "std_gyro_x": 0.03, "avg_gyro_y": 0.08, "std_gyro_y": 0.02},

    {"id": "U019", "name": "Đỗ Quốc Tuấn",      "desc": "Kỹ sư, dùng cả 2 ngón trỏ",
     "avg_pressure": 0.53, "std_pressure": 0.04, "avg_touch_area": 142, "std_touch_area": 11,
     "avg_touch_duration": 82,  "std_touch_duration": 6,  "avg_inter_gap": 145, "std_inter_gap": 13,
     "avg_gyro_x": 0.07,  "std_gyro_x": 0.02, "avg_gyro_y": 0.05, "std_gyro_y": 0.015},

    {"id": "U020", "name": "Nguyễn Thị Uyên",   "desc": "Streamer, thao tác liên tục",
     "avg_pressure": 0.62, "std_pressure": 0.05, "avg_touch_area": 152, "std_touch_area": 11,
     "avg_touch_duration": 58,  "std_touch_duration": 4,  "avg_inter_gap": 85,  "std_inter_gap": 9,
     "avg_gyro_x": 0.25,  "std_gyro_x": 0.07, "avg_gyro_y": 0.20, "std_gyro_y": 0.06},
]

FEATURE_COLS = [
    "avg_pressure", "std_pressure",
    "avg_touch_area", "std_touch_area",
    "avg_touch_duration", "std_touch_duration",
    "avg_inter_gap", "std_inter_gap",
    "avg_gyro_x", "std_gyro_x",
    "avg_gyro_y", "std_gyro_y",
]

# ─────────────────────────────────────────────
#  Helper: generate noisy sessions for a persona
# ─────────────────────────────────────────────
def generate_sessions(persona: dict, n_sessions: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + hash(persona["id"]) % 10000)
    rows = []
    for i in range(n_sessions):
        row = {"user_id": persona["id"], "user_name": persona["name"],
               "session_id": f"{persona['id']}_S{i+1:03d}"}
        for feat in FEATURE_COLS:
            base = persona[feat]
            noise_ratio = 0.08 if feat.startswith("avg") else 0.12
            noise = rng.normal(0, abs(base) * noise_ratio)
            val = base + noise
            if feat.startswith("std"):
                val = max(val, 1e-4)
            row[feat] = round(val, 5)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  rPPG helpers
# ─────────────────────────────────────────────
def save_uploaded_video(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def extract_rppg_signal(video_path: str, max_frames: int = 300):
    import cv2
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_process = min(max_frames, total)

    green_vals = []
    n_detected = 0
    last_roi = None

    for _ in range(n_process):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Forehead ROI: top 30% of face, centre 50% width
            last_roi = (
                y + int(h * 0.08), y + int(h * 0.38),
                x + int(w * 0.25), x + int(w * 0.75),
            )
            n_detected += 1
        if last_roi:
            y1, y2, x1, x2 = last_roi
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                green_vals.append(float(np.mean(roi[:, :, 1])))  # BGR → index 1 = Green
            elif green_vals:
                green_vals.append(green_vals[-1])
        else:
            green_vals.append(128.0)

    cap.release()
    return np.array(green_vals, dtype=np.float32), fps, n_detected, len(green_vals)


def bandpass_filter_signal(signal: np.ndarray, fps: float,
                           low: float = 0.75, high: float = 4.0) -> np.ndarray:
    from scipy.signal import butter, filtfilt, detrend
    if len(signal) < 30:
        return signal
    sig = detrend(signal.astype(np.float64))
    nyq = fps / 2.0
    lo, hi = low / nyq, min(high / nyq, 0.98)
    if lo <= 0 or lo >= hi:
        return sig
    try:
        b, a = butter(3, [lo, hi], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def compute_vitals(raw_signal: np.ndarray, fps: float):
    """
    Quality = Peak Prominence — độ nổi bật của đỉnh nhịp tim trong phổ tần số.
      - Người thật: 1 đỉnh sắc nhọn tại tần số nhịp tim → peak >> mean → quality cao
      - Deepfake:   phổ phẳng như nhiễu → peak ≈ mean → quality thấp
    Dùng Welch PSD (mượt hơn periodogram) và dải HR thực tế 0.75–3.0 Hz (45–180 BPM).
    """
    from scipy.signal import welch, detrend
    import math
    if len(raw_signal) < 15:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    sig = detrend(raw_signal.astype(np.float64))

    # Welch cho PSD mượt hơn, ít bị nhiễu lẻ làm lệch kết quả
    nperseg = min(len(sig), 256)
    freqs, psd = welch(sig, fps, nperseg=nperseg)

    # Dải HR thực tế: 45–180 BPM = 0.75–3.0 Hz (bỏ 3.0–4.0 Hz vì không thực tế)
    hr_band = (freqs >= 0.75) & (freqs <= 3.0)
    if not np.any(hr_band):
        return 0.0, 0.0, freqs, psd

    psd_hr     = psd[hr_band]
    peak_power = float(np.max(psd_hr))
    mean_power = float(np.mean(psd_hr)) + 1e-12

    # Prominence ratio: người thật ratio cao (5x–15x), deepfake/nhiễu ratio ~1–2x
    prominence = peak_power / mean_power

    # Normalize về [0, 1] bằng log scale (nhạy ở vùng thấp, không bị bão hoà sớm)
    # ratio=1 → 0% | ratio=3 → ~44% | ratio=6 → ~73% | ratio=12 → 100%
    quality = float(np.clip(math.log(max(prominence, 1.0)) / math.log(12), 0.0, 1.0))

    peak_freq = float(freqs[hr_band][np.argmax(psd_hr)])
    hr_bpm    = peak_freq * 60.0
    return hr_bpm, quality, freqs, psd


def create_ecg_animation(time_ax: np.ndarray, filtered_sig: np.ndarray,
                          label: str, is_real: bool, skip: int = 3):
    """
    Tạo animated Plotly chart kiểu màn hình monitor bệnh viện.
    Tín hiệu được vẽ dần từ trái sang phải theo từng frame.
    """
    import plotly.graph_objects as go

    # Màu sắc theo loại video
    if is_real:
        line_color  = "#00ff88"   # Neon green — người thật
        bg_color    = "#000d00"   # Đen xanh rất đậm
        grid_color  = "#002200"
        glow_color  = "rgba(0,255,136,0.08)"
    else:
        line_color  = "#ff4444"   # Đỏ — deepfake
        bg_color    = "#0d0000"
        grid_color  = "#220000"
        glow_color  = "rgba(255,68,68,0.08)"

    t_sub = time_ax[::skip]
    s_sub = filtered_sig[::skip]
    n_sub = len(t_sub)

    y_min = float(s_sub.min()) * 1.6
    y_max = float(s_sub.max()) * 1.6
    if abs(y_max - y_min) < 1e-6:          # tín hiệu phẳng (deepfake)
        y_min, y_max = -1.0, 1.0

    # ── Tạo frames ──────────────────────────────────
    frames = []
    for i in range(1, n_sub + 1):
        frames.append(go.Frame(
            data=[
                # Vùng fill (glow effect dưới đường)
                go.Scatter(
                    x=t_sub[:i], y=s_sub[:i],
                    fill="tozeroy", fillcolor=glow_color,
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                ),
                # Đường tín hiệu chính
                go.Scatter(
                    x=t_sub[:i], y=s_sub[:i],
                    mode="lines",
                    line=dict(color=line_color, width=2.5),
                    name=label, showlegend=False,
                ),
                # Điểm sáng ở đầu đường (moving cursor)
                go.Scatter(
                    x=[t_sub[i - 1]], y=[s_sub[i - 1]],
                    mode="markers",
                    marker=dict(color="white", size=7, opacity=0.9),
                    showlegend=False,
                ),
            ],
            name=str(i),
        ))

    # ── Layout kiểu hospital monitor ────────────────
    verdict_text = "● LIVE SIGNAL DETECTED" if is_real else "● NO BIOLOGICAL SIGNAL"
    layout = go.Layout(
        title=dict(
            text=f"<b>{label}</b>  <span style='font-size:13px;color:{line_color}'>"
                 f"{verdict_text}</span>",
            font=dict(color=line_color, size=15, family="monospace"),
            x=0.02,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=line_color, family="monospace"),
        xaxis=dict(
            range=[float(t_sub[0]), float(t_sub[-1])],
            color=line_color,
            gridcolor=grid_color,
            showgrid=True,
            zeroline=False,
            title=dict(text="Thời gian (giây)", font=dict(size=11)),
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            range=[y_min, y_max],
            color=line_color,
            gridcolor=grid_color,
            showgrid=True,
            zeroline=True,
            zerolinecolor=grid_color,
            zerolinewidth=1,
            title=dict(text="Biên độ", font=dict(size=11)),
            tickfont=dict(size=10),
        ),
        height=280,
        margin=dict(l=55, r=20, t=50, b=45),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            bgcolor="#111111",
            bordercolor=line_color,
            font=dict(color=line_color, size=12),
            x=0.5, xanchor="center",
            y=1.22, yanchor="top",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 60, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0, "easing": "linear"},
                    }],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                ),
                dict(
                    label="↺ Reset",
                    method="animate",
                    args=[["1"], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(visible=False),
            pad=dict(b=5, t=5),
            bgcolor="#111111",
            bordercolor=line_color,
            tickcolor=line_color,
            font=dict(color=line_color, size=9),
            steps=[
                dict(
                    args=[[str(i)], {
                        "frame": {"duration": 60, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                    method="animate",
                    label="",
                )
                for i in range(1, n_sub + 1, max(1, n_sub // 50))
            ],
        )],
    )

    # Initial state: chỉ hiển thị 1 điểm đầu
    fig = go.Figure(
        data=[
            go.Scatter(x=t_sub[:1], y=s_sub[:1],
                       fill="tozeroy", fillcolor=glow_color,
                       line=dict(color="rgba(0,0,0,0)"), showlegend=False),
            go.Scatter(x=t_sub[:1], y=s_sub[:1],
                       mode="lines", line=dict(color=line_color, width=2.5),
                       showlegend=False),
            go.Scatter(x=[t_sub[0]], y=[s_sub[0]],
                       mode="markers", marker=dict(color="white", size=7),
                       showlegend=False),
        ],
        frames=frames,
        layout=layout,
    )
    return fig
@st.cache_resource
def build_siamese_model():
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model

    def build_mlp():
        inp = Input(shape=(12,))
        x = Dense(64, activation='relu')(inp)
        x = Dense(32, activation='relu')(x)
        x = Dense(16)(x)
        return Model(inp, x, name="shared_mlp")

    mlp = build_mlp()

    input_1 = Input(shape=(12,), name="session_A")
    input_2 = Input(shape=(12,), name="session_B")

    embed_1 = mlp(input_1)
    embed_2 = mlp(input_2)

    distance = Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True) + 1e-8),
        name="euclidean_distance"
    )([embed_1, embed_2])

    output = Dense(1, activation='sigmoid', name="similarity_score")(distance)

    model = Model([input_1, input_2], output, name="siamese_network")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, mlp


# ─────────────────────────────────────────────
#  Generate training pairs
# ─────────────────────────────────────────────
def generate_pairs(df: pd.DataFrame, n_same: int = 500, n_diff: int = 500, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique()
    pairs = []

    # Same-user pairs (label = 1)
    for _ in range(n_same):
        uid = rng.choice(users)
        sessions = df[df["user_id"] == uid]
        if len(sessions) < 2:
            continue
        idx = rng.choice(len(sessions), 2, replace=False)
        s1, s2 = sessions.iloc[idx[0]], sessions.iloc[idx[1]]
        row = {}
        for f in FEATURE_COLS:
            row[f"A_{f}"] = s1[f]
            row[f"B_{f}"] = s2[f]
        row["user_A"] = s1["user_id"]
        row["user_B"] = s2["user_id"]
        row["label"] = 1
        row["label_text"] = "Cùng người"
        pairs.append(row)

    # Different-user pairs (label = 0)
    for _ in range(n_diff):
        uid_a, uid_b = rng.choice(users, 2, replace=False)
        s1 = df[df["user_id"] == uid_a].iloc[rng.integers(len(df[df["user_id"] == uid_a]))]
        s2 = df[df["user_id"] == uid_b].iloc[rng.integers(len(df[df["user_id"] == uid_b]))]
        row = {}
        for f in FEATURE_COLS:
            row[f"A_{f}"] = s1[f]
            row[f"B_{f}"] = s2[f]
        row["user_A"] = uid_a
        row["user_B"] = uid_b
        row["label"] = 0
        row["label_text"] = "Khác người"
        pairs.append(row)

    return pd.DataFrame(pairs).sample(frac=1, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
st.title("🔐 Fraud Detection ATO — Siamese Network + MLP")
st.markdown(
    "> **Câu hỏi cốt lõi:** *User hiện tại có giống chính họ trong quá khứ không?*  \n"
    "> Ứng dụng sinh trắc học hành vi (behavioral biometrics) trên thiết bị di động."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tab 1 — Tạo Data Giả Lập",
    "🧠 Tab 2 — Siamese Network + MLP",
    "🫀 Tab 3 — Demo rPPG Deepfake",
    "🤖 Tab 4 — AI Deepfake Detection",
])

# ══════════════════════════════════════════════
#  TAB 1
# ══════════════════════════════════════════════
with tab1:
    st.header("📊 Tạo Bộ Dữ Liệu Giả Lập")
    st.markdown(
        "Sinh dữ liệu hành vi cho **20 user** với **12 features** mỗi người.  \n"
        "Mỗi user sẽ có nhiều session với noise thực tế."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_sessions = st.slider("Số session mỗi user", 20, 50, 25)
    with col2:
        noise_seed = st.number_input("Random seed", 0, 9999, 42)
    with col3:
        show_persona = st.checkbox("Hiện bảng persona", value=True)

    if show_persona:
        persona_df = pd.DataFrame([
            {"User ID": p["id"], "Tên": p["name"], "Đặc điểm": p["desc"]}
            for p in PERSONAS
        ])
        st.dataframe(persona_df, use_container_width=True, hide_index=True)

    if st.button("🚀 Tạo Dữ Liệu", type="primary", key="gen_data"):
        with st.spinner("Đang tạo dữ liệu..."):
            dfs = []
            for p in PERSONAS:
                dfs.append(generate_sessions(p, n_sessions, seed=int(noise_seed)))
            full_df = pd.concat(dfs, ignore_index=True)
            st.session_state["raw_df"] = full_df

        st.success(f"✅ Đã tạo **{len(full_df)} sessions** cho 20 users!")

        # Preview
        st.subheader("Xem trước dữ liệu")
        st.dataframe(full_df.head(30), use_container_width=True, hide_index=True)

        # Statistics
        st.subheader("Thống kê tổng quan")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tổng sessions", len(full_df))
        col_b.metric("Số users", full_df["user_id"].nunique())
        col_c.metric("Số features", len(FEATURE_COLS))

        # Distribution plot
        st.subheader("Phân bố avg_pressure theo từng user")
        fig, ax = plt.subplots(figsize=(14, 5))
        for p in PERSONAS:
            uid_df = full_df[full_df["user_id"] == p["id"]]
            ax.scatter(
                [p["id"]] * len(uid_df), uid_df["avg_pressure"],
                alpha=0.4, s=20, label=p["id"]
            )
        ax.set_xlabel("User ID")
        ax.set_ylabel("avg_pressure")
        ax.set_title("Phân bố avg_pressure — mỗi chấm là 1 session")
        plt.xticks(rotation=45, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Download raw CSV
        csv_raw = full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Tải xuống Raw Data CSV",
            data=csv_raw,
            file_name="ato_behavioral_data.csv",
            mime="text/csv",
        )

    # ── Tạo pair dataset ──────────────────────────────────
    st.divider()
    st.subheader("🔗 Tạo Bộ Dữ Liệu Cặp (Pair Dataset)")
    st.markdown("Ghép các session thành cặp để train Siamese Network.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        n_same_pairs = st.slider("Số cặp cùng người (label=1)", 200, 1000, 500, step=50)
    with col_p2:
        n_diff_pairs = st.slider("Số cặp khác người (label=0)", 200, 1000, 500, step=50)

    if st.button("🔗 Tạo Pair Dataset", key="gen_pairs"):
        if "raw_df" not in st.session_state:
            st.warning("⚠️ Hãy tạo Raw Data trước!")
        else:
            with st.spinner("Đang tạo cặp dữ liệu..."):
                pair_df = generate_pairs(
                    st.session_state["raw_df"], n_same_pairs, n_diff_pairs
                )
                st.session_state["pair_df"] = pair_df

            st.success(f"✅ Đã tạo **{len(pair_df)} cặp** ({n_same_pairs} cùng + {n_diff_pairs} khác)!")
            st.dataframe(pair_df.head(20), use_container_width=True, hide_index=True)

            col_d1, col_d2 = st.columns(2)
            col_d1.metric("Cặp cùng người", int((pair_df["label"] == 1).sum()))
            col_d2.metric("Cặp khác người", int((pair_df["label"] == 0).sum()))

            csv_pair = pair_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Tải xuống Pair Dataset CSV",
                data=csv_pair,
                file_name="ato_pair_dataset.csv",
                mime="text/csv",
            )

# ══════════════════════════════════════════════
#  TAB 2
# ══════════════════════════════════════════════
with tab2:
    st.header("🧠 Siamese Network + MLP Training & Demo")

    # ── Architecture diagram ─────────────────────────────
    st.subheader("Kiến Trúc Tổng Thể")
    st.code(
        """
Session_A → [12 features] → ┐
                              ├─ Shared MLP (64→32→16) → Embedding_A ─┐
Session_B → [12 features] → ┘                                          ├→ Euclidean Distance → Sigmoid → Score [0,1]
                                              Embedding_B ─────────────┘

Score ≥ 0.5  →  ✅ Cùng người (Legitimate)
Score  < 0.5 →  ❌ Khác người (FRAUD / ATO)
        """,
        language="text",
    )

    # ── Training section ─────────────────────────────────
    st.subheader("🏋️ Huấn Luyện Mô Hình")

    if "pair_df" not in st.session_state:
        st.info("ℹ️ Vui lòng tạo Pair Dataset ở Tab 1 trước.")
    else:
        pair_df = st.session_state["pair_df"]

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            epochs = st.slider("Số epochs", 5, 100, 30)
        with col_t2:
            batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
        with col_t3:
            test_split = st.slider("Test split %", 10, 40, 20)

        if st.button("🏋️ Train Siamese Network", type="primary", key="train_btn"):
            import tensorflow as tf
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            with st.spinner("Đang xây dựng và train mô hình..."):
                feat_A = [f"A_{f}" for f in FEATURE_COLS]
                feat_B = [f"B_{f}" for f in FEATURE_COLS]

                X_A = pair_df[feat_A].values.astype(np.float32)
                X_B = pair_df[feat_B].values.astype(np.float32)
                y   = pair_df["label"].values.astype(np.float32)

                # Normalize all features together
                all_X = np.vstack([X_A, X_B])
                scaler = StandardScaler()
                scaler.fit(all_X)
                X_A_scaled = scaler.transform(X_A)
                X_B_scaled = scaler.transform(X_B)

                (XA_tr, XA_te, XB_tr, XB_te, y_tr, y_te) = train_test_split(
                    X_A_scaled, X_B_scaled, y,
                    test_size=test_split / 100, random_state=42, stratify=y
                )

                model, mlp_backbone = build_siamese_model()
                st.session_state["scaler"] = scaler
                st.session_state["mlp_backbone"] = mlp_backbone
                st.session_state["model"] = model

                history_store = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
                progress_bar = st.progress(0)
                status_text  = st.empty()

                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        history_store["loss"].append(logs.get("loss", 0))
                        history_store["val_loss"].append(logs.get("val_loss", 0))
                        history_store["accuracy"].append(logs.get("accuracy", 0))
                        history_store["val_accuracy"].append(logs.get("val_accuracy", 0))
                        pct = int((epoch + 1) / epochs * 100)
                        progress_bar.progress(pct)
                        status_text.text(
                            f"Epoch {epoch+1}/{epochs} — "
                            f"loss: {logs.get('loss',0):.4f} | "
                            f"val_loss: {logs.get('val_loss',0):.4f} | "
                            f"val_acc: {logs.get('val_accuracy',0):.4f}"
                        )

                model.fit(
                    [XA_tr, XB_tr], y_tr,
                    validation_data=([XA_te, XB_te], y_te),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[StreamlitCallback()],
                    verbose=0,
                )

                st.session_state["XA_te"] = XA_te
                st.session_state["XB_te"] = XB_te
                st.session_state["y_te"]  = y_te
                st.session_state["history"] = history_store

            st.success("✅ Huấn luyện hoàn tất!")

            # Training curves
            st.subheader("📈 Đường Cong Học Tập")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            h = history_store
            ax1.plot(h["loss"],     label="Train Loss",      color="#e74c3c")
            ax1.plot(h["val_loss"], label="Validation Loss",  color="#e67e22", linestyle="--")
            ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

            ax2.plot(h["accuracy"],     label="Train Accuracy",     color="#2980b9")
            ax2.plot(h["val_accuracy"], label="Validation Accuracy", color="#27ae60", linestyle="--")
            ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Evaluation
            from sklearn.metrics import (
                classification_report, confusion_matrix, roc_auc_score
            )
            preds = model.predict([XA_te, XB_te], verbose=0).flatten()
            pred_labels = (preds >= 0.5).astype(int)

            auc = roc_auc_score(y_te, preds)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("AUC-ROC", f"{auc:.4f}")
            col_m2.metric("Accuracy", f"{np.mean(pred_labels == y_te):.4f}")
            col_m3.metric("Test samples", len(y_te))

            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_te, pred_labels)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["Khác người (0)", "Cùng người (1)"],
                yticklabels=["Khác người (0)", "Cùng người (1)"],
            )
            ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

            # Score distribution
            st.subheader("Phân bố Score dự đoán")
            fig_s, ax_s = plt.subplots(figsize=(8, 4))
            ax_s.hist(preds[y_te == 1], bins=30, alpha=0.6, label="Cùng người (1)", color="#27ae60")
            ax_s.hist(preds[y_te == 0], bins=30, alpha=0.6, label="Khác người (0)", color="#e74c3c")
            ax_s.axvline(0.5, color="black", linestyle="--", label="Ngưỡng 0.5")
            ax_s.set_xlabel("Score"); ax_s.set_ylabel("Số lượng")
            ax_s.set_title("Phân bố Score — mô hình càng tốt khi 2 đỉnh tách xa nhau")
            ax_s.legend()
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close()

    # ── Live Demo ────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Demo Live — Kiểm Tra Một Session Mới")
    st.markdown(
        "Chọn **user gốc** (reference) và nhập hoặc sinh ngẫu nhiên một session mới. "
        "Mô hình sẽ cho biết session mới có phải cùng người không."
    )

    if "model" not in st.session_state:
        st.info("ℹ️ Vui lòng train mô hình trước.")
    else:
        model  = st.session_state["model"]
        scaler = st.session_state["scaler"]
        raw_df = st.session_state.get("raw_df", None)

        if raw_df is None:
            st.warning("Cần có Raw Data từ Tab 1.")
        else:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                ref_user_id = st.selectbox(
                    "Chọn User Gốc (Reference)",
                    options=[p["id"] for p in PERSONAS],
                    format_func=lambda x: f"{x} — {next(p['name'] for p in PERSONAS if p['id']==x)}"
                )
            with col_d2:
                test_scenario = st.radio(
                    "Kịch bản test",
                    ["✅ Cùng người (nhập nhiễu nhẹ)", "❌ Kẻ gian mạo danh (user khác)"],
                    horizontal=True
                )

            if st.button("🔍 Kiểm Tra Ngay", key="demo_btn"):
                ref_sessions = raw_df[raw_df["user_id"] == ref_user_id]
                ref_vec = ref_sessions[FEATURE_COLS].mean().values.astype(np.float32)

                if "Cùng người" in test_scenario:
                    rng_d = np.random.default_rng(77)
                    noise = rng_d.normal(0, np.abs(ref_vec) * 0.07)
                    new_vec = (ref_vec + noise).astype(np.float32)
                    true_label = "Cùng người"
                else:
                    other_users = [p for p in PERSONAS if p["id"] != ref_user_id]
                    imposter = np.random.choice(other_users)
                    imp_sessions = raw_df[raw_df["user_id"] == imposter["id"]]
                    new_vec = imp_sessions[FEATURE_COLS].mean().values.astype(np.float32)
                    true_label = f"Kẻ gian ({imposter['name']})"

                ref_scaled = scaler.transform(ref_vec.reshape(1, -1))
                new_scaled = scaler.transform(new_vec.reshape(1, -1))

                score = float(model.predict(
                    [ref_scaled, new_scaled], verbose=0
                ).flatten()[0])

                mlp = st.session_state["mlp_backbone"]
                emb_ref = mlp.predict(ref_scaled, verbose=0).flatten()
                emb_new = mlp.predict(new_scaled, verbose=0).flatten()
                dist    = float(np.sqrt(np.sum((emb_ref - emb_new) ** 2)))

                st.markdown("---")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Similarity Score", f"{score:.4f}", help="Gần 1 = cùng người")
                col_r2.metric("Euclidean Distance", f"{dist:.4f}", help="Gần 0 = cùng người")
                verdict = "✅ Hợp lệ (Cùng người)" if score >= 0.5 else "❌ FRAUD / ATO (Khác người)"
                col_r3.metric("Kết quả", verdict)

                verdict_color = "#27ae60" if score >= 0.5 else "#e74c3c"
                st.markdown(
                    f"<div style='background:{verdict_color};color:white;padding:16px;"
                    f"border-radius:8px;font-size:20px;text-align:center'>"
                    f"{verdict} &nbsp;|&nbsp; Ground truth: <b>{true_label}</b></div>",
                    unsafe_allow_html=True,
                )

                # Embedding visualization
                st.subheader("So sánh Embedding Vector (16 chiều)")
                fig_e, ax_e = plt.subplots(figsize=(10, 3))
                x_idx = np.arange(16)
                ax_e.bar(x_idx - 0.2, emb_ref, 0.4, label="Reference (user gốc)", color="#2980b9", alpha=0.8)
                ax_e.bar(x_idx + 0.2, emb_new, 0.4, label="Session mới",          color="#e74c3c", alpha=0.8)
                ax_e.set_xlabel("Dimension"); ax_e.set_ylabel("Giá trị")
                ax_e.set_title("Embedding 16D — càng giống nhau thì score càng cao")
                ax_e.legend()
                plt.tight_layout()
                st.pyplot(fig_e)
                plt.close()

    # ── Export pair data ─────────────────────────────────
    st.divider()
    st.subheader("⬇️ Export Pair Data (đã dùng để train)")
    if "pair_df" in st.session_state:
        csv_pair = st.session_state["pair_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Tải Pair Dataset CSV",
            data=csv_pair,
            file_name="ato_pair_dataset.csv",
            mime="text/csv",
        )
        st.dataframe(
            st.session_state["pair_df"][["user_A", "user_B", "label", "label_text"]
            ].value_counts().reset_index(name="count"),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Tạo Pair Dataset ở Tab 1 trước.")

# ══════════════════════════════════════════════
#  TAB 3 — rPPG DEEPFAKE DETECTION
# ══════════════════════════════════════════════
with tab3:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.header("🫀 rPPG — Phát Hiện Deepfake Qua Tín Hiệu Sự Sống")
    st.markdown(
        "**Nguyên lý cốt lõi:**  \n"
        "Mặt người thật → máu lưu thông → kênh **GREEN** của camera "
        "dao động nhẹ theo từng nhịp tim.  \n"
        "Deepfake là ảnh/video tổng hợp → **không có tín hiệu sinh học** → "
        "GREEN channel phẳng hoặc nhiễu loạn ngẫu nhiên."
    )

    st.code(
        """
Pipeline rPPG:
  Video → Phát hiện khuôn mặt (Haar Cascade)
        → Cắt vùng trán (ROI: top 30% khuôn mặt)
        → Trung bình kênh GREEN mỗi frame
        → Tín hiệu thô theo thời gian
        → Bandpass filter  0.75 – 4.0 Hz  (tương đương 45–240 BPM)
        → Phân tích phổ tần số (FFT)

  Người thật  →  sóng tuần hoàn rõ ràng  →  đỉnh FFT sắc nét   ✅
  Deepfake    →  tín hiệu phẳng / nhiễu  →  không có đỉnh FFT  ❌
        """,
        language="text",
    )

    # ── Upload section ────────────────────────────────────
    st.subheader("📤 Tải Lên Video")
    st.caption(
        "Hỗ trợ MP4 / AVI / MOV / MKV. "
        "Khuyến nghị: video 10–30 giây, khuôn mặt nhìn thẳng, ánh sáng đủ."
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown(
            "<div style='background:#eafaf1;border-left:4px solid #27ae60;"
            "padding:10px;border-radius:4px'><b>🟢 Video Người Thật</b></div>",
            unsafe_allow_html=True,
        )
        real_file = st.file_uploader(
            "Chọn video người thật", type=["mp4", "avi", "mov", "mkv"], key="real_vid"
        )
    with col_u2:
        st.markdown(
            "<div style='background:#fdedec;border-left:4px solid #e74c3c;"
            "padding:10px;border-radius:4px'><b>🔴 Video Deepfake</b></div>",
            unsafe_allow_html=True,
        )
        fake_file = st.file_uploader(
            "Chọn video deepfake", type=["mp4", "avi", "mov", "mkv"], key="fake_vid"
        )

    # ── Config ───────────────────────────────────────────
    st.divider()
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        max_frames = st.slider(
            "Số frame tối đa xử lý", 100, 600, 300, step=50,
            help="Nhiều frame = chính xác hơn nhưng mất thêm vài giây"
        )
    with col_cfg2:
        quality_threshold = st.slider(
            "Ngưỡng 'Có Sự Sống' (%)", 5, 50, 28,
            help="Người thật: thường 30–95% | Deepfake (nhiễu): thường 15–25%. Mặc định 28% là điểm tách tốt."
        )

    # ── Run analysis ─────────────────────────────────────
    if st.button("🔬 Phân Tích rPPG", type="primary", key="rppg_btn"):
        if real_file is None and fake_file is None:
            st.warning("⚠️ Vui lòng tải lên ít nhất 1 video.")
        else:
            VIDEO_CONFIGS = []
            if real_file:
                VIDEO_CONFIGS.append(("Người Thật", real_file,
                                      "#27ae60", "rgba(39,174,96,0.12)"))
            if fake_file:
                VIDEO_CONFIGS.append(("Deepfake", fake_file,
                                      "#e74c3c", "rgba(231,76,60,0.12)"))

            results = {}
            for label, vfile, color, fill in VIDEO_CONFIGS:
                with st.spinner(f"Đang phân tích **{label}** — trích xuất tín hiệu rPPG..."):
                    tmp_path = save_uploaded_video(vfile)
                    try:
                        raw, fps, n_det, n_tot = extract_rppg_signal(tmp_path, max_frames)
                    except ModuleNotFoundError as e:
                        st.error(
                            f"❌ **Thiếu thư viện:** `{e.name}`\n\n"
                            "Vui lòng cài đặt bằng lệnh:\n"
                            "```\npip install opencv-python-headless scipy plotly\n```\n"
                            "Sau đó khởi động lại ứng dụng."
                        )
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xử lý video **{label}**: {e}")
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                        break
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    filtered          = bandpass_filter_signal(raw, fps)
                    # Quality tính trên tín hiệu thô (detrended) — không phải filtered
                    hr, quality, freqs, psd = compute_vitals(raw, fps)
                    time_ax           = np.arange(len(raw)) / fps
                    results[label]    = dict(
                        raw=raw, filtered=filtered, fps=fps,
                        time=time_ax, freqs=freqs, psd=psd,
                        hr=hr, quality=quality,
                        n_det=n_det, n_tot=n_tot,
                        color=color, fill=fill,
                    )

            st.success("✅ Phân tích hoàn tất!")

            # ── Metric cards ─────────────────────────────
            st.subheader("📊 Kết Quả Tổng Quan")
            metric_cols = st.columns(len(results))
            for i, (label, r) in enumerate(results.items()):
                is_alive = r["quality"] * 100 >= quality_threshold
                verdict  = "✅ CÓ SỰ SỐNG" if is_alive else "❌ KHÔNG CÓ SỰ SỐNG"
                bg       = "#eafaf1" if is_alive else "#fdedec"
                border   = "#27ae60" if is_alive else "#e74c3c"
                hr_display = "N/A" if r["hr"] < 30 else f"{r['hr']:.0f} BPM"
                metric_cols[i].markdown(
                    f"<div style='background:{bg};border:2px solid {border};"
                    f"border-radius:10px;padding:16px;text-align:center'>"
                    f"<div style='font-size:18px;font-weight:bold'>{label}</div>"
                    f"<div style='font-size:28px;font-weight:bold;color:{border}'>{verdict}</div>"
                    f"<hr style='margin:8px 0'>"
                    f"<div>💓 Nhịp tim ước tính: <b>{hr_display}</b></div>"
                    f"<div>📶 Chỉ số sự sống: <b>{r['quality']*100:.1f}%</b></div>"
                    f"<div>🎯 Phát hiện mặt: <b>{r['n_det']}/{r['n_tot']} frame</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Chart 1: Raw GREEN signal ─────────────────
            st.subheader("📈 Tín Hiệu Kênh GREEN Thô")
            st.caption(
                "Giá trị trung bình kênh GREEN vùng trán mỗi frame. "
                "Người thật: dao động nhẹ đều đặn. Deepfake: phẳng hoặc nhảy bất thường."
            )
            fig1 = go.Figure()
            for label, r in results.items():
                fig1.add_trace(go.Scatter(
                    x=r["time"], y=r["raw"],
                    name=label,
                    line=dict(color=r["color"], width=1.5),
                    mode="lines",
                    hovertemplate="t=%{x:.2f}s  GREEN=%{y:.2f}<extra>" + label + "</extra>",
                ))
            fig1.update_layout(
                xaxis_title="Thời gian (giây)",
                yaxis_title="Giá trị kênh GREEN (0–255)",
                height=320,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=20, t=10, b=50),
            )
            st.plotly_chart(fig1, use_container_width=True)

            # ── Chart 2: Filtered signal ──────────────────
            st.subheader("💓 Tín Hiệu Sau Lọc Bandpass (0.75 – 4.0 Hz)")
            st.caption(
                "Sau khi loại bỏ nhiễu DC và tần số ngoài dải nhịp tim. "
                "Người thật: sóng hình sin đẹp, chu kỳ ~0.8–1.2 giây. "
                "Deepfake: đường gần phẳng, không có hình dạng sinh học."
            )
            fig2 = go.Figure()
            for label, r in results.items():
                fig2.add_trace(go.Scatter(
                    x=r["time"], y=r["filtered"],
                    name=label,
                    line=dict(color=r["color"], width=2.5),
                    fill="tozeroy",
                    fillcolor=r["fill"],
                    mode="lines",
                    hovertemplate="t=%{x:.2f}s  amp=%{y:.4f}<extra>" + label + "</extra>",
                ))
            fig2.update_layout(
                xaxis_title="Thời gian (giây)",
                yaxis_title="Biên độ tín hiệu",
                height=340,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=20, t=10, b=50),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── Chart 3: FFT Power Spectrum ───────────────
            st.subheader("🔊 Phổ Tần Số FFT — Dấu Vân Tay Sinh Học")
            st.caption(
                "Người thật → đỉnh nhọn rõ ràng trong vùng 45–240 BPM (vùng xanh lá).  \n"
                "Deepfake  → phổ phẳng, không có đỉnh — bằng chứng thiếu tín hiệu sinh học."
            )
            fig3 = go.Figure()
            # Highlight heart rate band
            fig3.add_vrect(
                x0=45, x1=240,
                fillcolor="rgba(39,174,96,0.07)",
                line_width=0,
                annotation_text="Vùng nhịp tim bình thường",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="#27ae60",
            )
            for label, r in results.items():
                # Show only 0–5 Hz range (0–300 BPM)
                mask = (r["freqs"] > 0) & (r["freqs"] <= 5.0)
                bpm_axis = r["freqs"][mask] * 60.0
                fig3.add_trace(go.Scatter(
                    x=bpm_axis,
                    y=r["psd"][mask],
                    name=label,
                    line=dict(color=r["color"], width=2.5),
                    fill="tozeroy",
                    fillcolor=r["fill"],
                    mode="lines",
                    hovertemplate="BPM=%{x:.1f}  Power=%{y:.4e}<extra>" + label + "</extra>",
                ))
                # Mark peak in heart rate band
                hr_mask = (bpm_axis >= 45) & (bpm_axis <= 240)
                if np.any(hr_mask):
                    peak_bpm   = float(bpm_axis[hr_mask][np.argmax(r["psd"][mask][hr_mask])])
                    peak_power = float(np.max(r["psd"][mask][hr_mask]))
                    fig3.add_trace(go.Scatter(
                        x=[peak_bpm], y=[peak_power],
                        mode="markers+text",
                        marker=dict(color=r["color"], size=12, symbol="star"),
                        text=[f"  {peak_bpm:.0f} BPM"],
                        textposition="middle right",
                        textfont=dict(color=r["color"], size=12),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
            fig3.update_layout(
                xaxis_title="Nhịp tim tương đương (BPM)",
                yaxis_title="Công suất tín hiệu",
                height=380,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=20, t=10, b=50),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # ── Chart 4: Signal Quality Bar ───────────────
            if len(results) == 2:
                st.subheader("📊 So Sánh Chỉ Số Tín Hiệu Sự Sống")
                st.caption(
                    "Tỉ lệ % năng lượng tín hiệu nằm trong băng nhịp tim (0.75–4 Hz). "
                    "Ngưỡng phân loại hiện tại được đánh dấu bằng đường đứt nét."
                )
                labels_bar  = list(results.keys())
                quality_bar = [r["quality"] * 100 for r in results.values()]
                colors_bar  = [r["color"] for r in results.values()]

                fig4 = go.Figure(go.Bar(
                    x=labels_bar,
                    y=quality_bar,
                    marker_color=colors_bar,
                    text=[f"{q:.1f}%" for q in quality_bar],
                    textposition="outside",
                    textfont=dict(size=16, color="black"),
                ))
                fig4.add_hline(
                    y=quality_threshold,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Ngưỡng phân loại: {quality_threshold}%",
                    annotation_position="top right",
                    annotation_font_color="orange",
                )
                fig4.update_layout(
                    yaxis_title="Chỉ số sự sống (%)",
                    height=320,
                    margin=dict(l=50, r=20, t=10, b=50),
                    yaxis_range=[0, max(max(quality_bar) * 1.3, quality_threshold * 1.5)],
                )
                st.plotly_chart(fig4, use_container_width=True)

            # ── Animated ECG monitor ──────────────────────
            st.divider()
            st.subheader("🏥 Hospital Monitor — Sóng Tim Trực Quan")
            st.caption(
                "Nhấn **▶ Play** để xem tín hiệu rPPG vẽ dần theo thời gian thực.  \n"
                "🟢 Người thật: sóng dao động đều đặn theo nhịp tim.  "
                "🔴 Deepfake: đường gần phẳng, không có nhịp sinh học."
            )
            ecg_cols = st.columns(len(results))
            for i, (label, r) in enumerate(results.items()):
                is_real_signal = r["quality"] * 100 >= quality_threshold
                skip = max(2, len(r["filtered"]) // 120)   # ~120 frames tối đa
                fig_ecg = create_ecg_animation(
                    r["time"], r["filtered"], label, is_real_signal, skip=skip
                )
                ecg_cols[i].plotly_chart(fig_ecg, use_container_width=True)

            # ── Final verdict banner ──────────────────────
            st.divider()
            st.subheader("🏁 Kết Luận Cuối Cùng")
            verdict_cols = st.columns(len(results))
            for i, (label, r) in enumerate(results.items()):
                is_alive = r["quality"] * 100 >= quality_threshold
                icon     = "✅" if is_alive else "❌"
                verdict  = "CÓ TÍN HIỆU SỰ SỐNG<br>→ NGƯỜI THẬT" if is_alive else "KHÔNG CÓ TÍN HIỆU<br>→ KHẢ NĂNG DEEPFAKE"
                bg       = "#1e8449" if is_alive else "#c0392b"
                hr_text  = f"{r['hr']:.0f} BPM" if r["hr"] >= 30 else "Không xác định"
                verdict_cols[i].markdown(
                    f"<div style='background:{bg};color:white;padding:24px 16px;"
                    f"border-radius:12px;text-align:center;font-size:16px'>"
                    f"<div style='font-size:22px;font-weight:bold;margin-bottom:8px'>"
                    f"{icon} {label}</div>"
                    f"<div style='font-size:26px;font-weight:bold;line-height:1.4'>"
                    f"{verdict}</div>"
                    f"<hr style='border-color:rgba(255,255,255,0.3);margin:12px 0'>"
                    f"<div>💓 Nhịp tim: <b>{hr_text}</b></div>"
                    f"<div>📶 Chỉ số sự sống: <b>{r['quality']*100:.1f}%</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Explanation ───────────────────────────────
            st.divider()
            with st.expander("ℹ️ Tại sao Deepfake không có tín hiệu rPPG?", expanded=False):
                st.markdown(
                    """
**rPPG (Remote Photoplethysmography)** hoạt động dựa trên nguyên lý:
khi tim đập, lượng máu dưới da thay đổi → hấp thụ ánh sáng thay đổi nhẹ
→ màu sắc pixel vùng da dao động với tần số = nhịp tim.

**Tại sao Deepfake thất bại:**
- Deepfake được tạo ra bằng cách **ánh xạ khuôn mặt** từ video nguồn sang video đích
- Quá trình này làm **trung bình hoặc xáo trộn** tín hiệu màu sắc vi tế theo từng pixel
- Kết quả: tín hiệu sinh học bị **hủy hoàn toàn hoặc mất tính chu kỳ**
- Ngay cả Deepfake chất lượng cao nhất vẫn rất khó giữ lại đúng tần số nhịp tim

**Ứng dụng trong ngân hàng:**
Lớp kiểm tra rPPG được đặt ngay tại bước **xác thực video call / eKYC**,
trước khi cho phép giao dịch lớn — chặn kẻ gian dùng video deepfake giả mạo khách hàng.
                    """
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — AI DEEPFAKE DETECTION (Pretrained Deep Learning)
#  ❗ Module này hoàn toàn ĐỘC LẬP với Tab 3.
#     Có thể xoá Tab 3 mà không ảnh hưởng tới Tab 4.
# ══════════════════════════════════════════════════════════════════════════════

# ---- Cấu hình Tab 4 (chỉnh ở đây để swap model) -----------------------------
TAB4_MODEL_NAME      = "prithivMLmods/Deep-Fake-Detector-Model"   # SigLIP-based, ~365MB
TAB4_BATCH_SIZE      = 4
TAB4_FACE_MIN_SIZE   = 80
TAB4_FACE_MARGIN     = 0.20    # mở rộng ROI 20% quanh mặt
TAB4_MAX_FRAMES      = 60


@st.cache_resource(show_spinner="🔄 Đang tải AI model deepfake detection (lần đầu sẽ tải ~365MB)...")
def tab4_load_model():
    """Load pretrained model & processor 1 lần duy nhất, cache lại."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch

    processor = AutoImageProcessor.from_pretrained(TAB4_MODEL_NAME)
    model     = AutoModelForImageClassification.from_pretrained(TAB4_MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    # Tự động dò index của nhãn "fake" trong id2label
    id2label = model.config.id2label
    fake_idx = None
    for idx, label in id2label.items():
        if any(k in str(label).lower() for k in ("fake", "deepfake", "synthetic", "ai")):
            fake_idx = int(idx)
            break
    if fake_idx is None:
        fake_idx = 1   # fallback an toàn

    return processor, model, device, fake_idx, id2label


def tab4_save_video(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def tab4_extract_face_frames(video_path: str, n_frames: int = 24):
    """
    Trích xuất n_frames đều nhau theo thời gian, phát hiện và crop khuôn mặt.

    Returns:
        faces       : list[PIL.Image]  — danh sách mặt đã crop
        timestamps  : list[float]       — mốc thời gian (giây) của từng frame
        fps         : float
        total_frames: int
        n_attempted : int               — số frame đã thử đọc
    """
    import cv2
    from PIL import Image

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], [], fps, 0, 0

    sample_indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    faces, timestamps = [], []
    n_attempted = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        n_attempted += 1
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(TAB4_FACE_MIN_SIZE, TAB4_FACE_MIN_SIZE),
        )
        if len(detected) == 0:
            continue
        x, y, w, h = max(detected, key=lambda f: f[2] * f[3])
        m = int(TAB4_FACE_MARGIN * max(w, h))
        x1 = max(0, x - m); y1 = max(0, y - m)
        x2 = min(frame.shape[1], x + w + m); y2 = min(frame.shape[0], y + h + m)
        face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        faces.append(Image.fromarray(face_rgb))
        timestamps.append(float(idx) / fps)

    cap.release()
    return faces, timestamps, fps, total, n_attempted


def tab4_predict_faces(faces, processor, model, device, fake_idx, batch_size=TAB4_BATCH_SIZE):
    """Inference batch các PIL faces, trả về list xác suất fake (0..1)."""
    import torch
    if not faces:
        return []
    out = []
    for i in range(0, len(faces), batch_size):
        batch = faces[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        out.extend(probs[:, fake_idx].tolist())
    return out


# ══════════════════════════════════════════════
#  TAB 4 — UI
# ══════════════════════════════════════════════
with tab4:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.header("🤖 AI Deepfake Detection — Pretrained Deep Learning")
    st.markdown(
        "Sử dụng **mô hình deep learning đã được huấn luyện sẵn** (pretrained) trên "
        "dataset deepfake quy mô lớn để phân loại real/fake **trên từng frame video**, "
        "sau đó tổng hợp thành verdict cuối cùng.\n\n"
        "**Pipeline:** Upload video → Trích xuất N frame đều theo thời gian → "
        "Phát hiện & crop khuôn mặt → AI inference → Tổng hợp xác suất → Verdict + biểu đồ"
    )

    with st.expander("ℹ️ Thông tin model & yêu cầu hệ thống", expanded=False):
        st.markdown(f"""
- **Model**: `{TAB4_MODEL_NAME}` (HuggingFace)
- **Backbone**: Vision Transformer (SigLIP) — fine-tune cho bài toán deepfake detection
- **Kích thước weights**: ~365MB (tải tự động lần đầu, sau đó cache)
- **RAM tối thiểu**: 6GB | **Khuyến nghị**: 8GB+
- **GPU**: tự động dùng nếu có CUDA, không bắt buộc
- **Tốc độ inference (CPU)**: ~0.3–1.5 giây/frame
- **Định dạng video hỗ trợ**: MP4, AVI, MOV, MKV
        """)

    st.divider()

    # ── Upload 2 video song song ─────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            "<div style='background:#eafaf1;border-left:4px solid #27ae60;"
            "padding:10px;border-radius:4px'><b>🟢 Video Người Thật</b></div>",
            unsafe_allow_html=True,
        )
        t4_real_file = st.file_uploader(
            "Chọn video thật", type=["mp4", "avi", "mov", "mkv"], key="t4_real"
        )
    with col_r:
        st.markdown(
            "<div style='background:#fdedec;border-left:4px solid #e74c3c;"
            "padding:10px;border-radius:4px'><b>🔴 Video Deepfake</b></div>",
            unsafe_allow_html=True,
        )
        t4_fake_file = st.file_uploader(
            "Chọn video deepfake", type=["mp4", "avi", "mov", "mkv"], key="t4_fake"
        )

    # ── Cấu hình ─────────────────────────────────────────
    st.divider()
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        t4_n_frames = st.slider(
            "Số frame trích xuất / video", 10, TAB4_MAX_FRAMES, 24, step=2,
            help="Nhiều frame ⇒ dự đoán ổn định hơn nhưng inference lâu hơn",
        )
    with cfg2:
        t4_threshold = st.slider(
            "Ngưỡng phân loại fake (%)", 30, 90, 50, step=5,
            help="Frame có P(fake) ≥ ngưỡng → coi là frame deepfake. "
                 "Verdict toàn video dựa vào % frame fake và mean P(fake).",
        )

    # ── Run analysis ─────────────────────────────────────
    if st.button("🚀 Phân Tích Bằng AI", type="primary", key="t4_run"):
        if t4_real_file is None and t4_fake_file is None:
            st.warning("⚠️ Vui lòng upload ít nhất 1 video.")
        else:
            # Load model
            try:
                processor, model, device, fake_idx, id2label = tab4_load_model()
            except ModuleNotFoundError as e:
                st.error(
                    f"❌ **Thiếu thư viện:** `{e.name}`\n\n"
                    "Cài đặt bằng:\n"
                    "```\npip install torch transformers Pillow timm\n```"
                )
                st.stop()
            except Exception as e:
                st.error(f"❌ Lỗi khi tải model: {e}")
                st.stop()

            st.success(f"✅ Đã tải model trên `{device.upper()}` "
                       f"| Labels: `{id2label}` | Fake-idx: `{fake_idx}`")

            # Build danh sách video cần xử lý
            VIDEOS = []
            if t4_real_file:
                VIDEOS.append(("Người Thật",  t4_real_file, "#27ae60", "rgba(39,174,96,0.15)"))
            if t4_fake_file:
                VIDEOS.append(("Deepfake",    t4_fake_file, "#e74c3c", "rgba(231,76,60,0.15)"))

            results = {}
            for label, vfile, color, fill in VIDEOS:
                with st.spinner(f"⏳ Đang xử lý **{label}** — trích xuất frame & inference AI..."):
                    tmp_path = tab4_save_video(vfile)
                    try:
                        faces, ts, fps, total, n_try = tab4_extract_face_frames(tmp_path, t4_n_frames)
                    except Exception as e:
                        st.error(f"❌ Lỗi trích xuất frame ({label}): {e}")
                        try: os.unlink(tmp_path)
                        except: pass
                        continue
                    try: os.unlink(tmp_path)
                    except: pass

                    if not faces:
                        st.warning(
                            f"⚠️ Không phát hiện được khuôn mặt nào trong video **{label}**. "
                            "Kiểm tra lại: ánh sáng, góc mặt, độ phân giải."
                        )
                        continue

                    fake_probs = tab4_predict_faces(
                        faces, processor, model, device, fake_idx
                    )
                    fake_arr = np.array(fake_probs)

                    results[label] = dict(
                        faces=faces, timestamps=ts, fake_probs=fake_arr,
                        fps=fps, total=total, n_try=n_try, n_face=len(faces),
                        color=color, fill=fill,
                    )

            if not results:
                st.error("❌ Không có kết quả để hiển thị.")
                st.stop()

            st.success(f"✅ Phân tích hoàn tất {len(results)} video!")

            # ── Verdict cards ────────────────────────────
            st.subheader("📊 Kết Quả Tổng Quan")
            v_cols = st.columns(len(results))
            for i, (label, r) in enumerate(results.items()):
                mean_p   = float(np.mean(r["fake_probs"]))
                max_p    = float(np.max(r["fake_probs"]))
                pct_fake = float(np.mean(r["fake_probs"] >= t4_threshold/100.0)) * 100.0
                is_fake  = mean_p * 100.0 >= t4_threshold
                verdict  = "🔴 DEEPFAKE" if is_fake else "🟢 NGƯỜI THẬT"
                bg       = "#fdedec" if is_fake else "#eafaf1"
                bd       = "#e74c3c" if is_fake else "#27ae60"
                v_cols[i].markdown(
                    f"<div style='background:{bg};border:2px solid {bd};border-radius:10px;"
                    f"padding:18px;text-align:center'>"
                    f"<div style='font-size:18px;font-weight:bold'>{label}</div>"
                    f"<div style='font-size:30px;font-weight:bold;color:{bd};margin:6px 0'>{verdict}</div>"
                    f"<hr style='margin:8px 0'>"
                    f"<div>📈 Mean P(fake): <b>{mean_p*100:.1f}%</b></div>"
                    f"<div>🔝 Max  P(fake): <b>{max_p*100:.1f}%</b></div>"
                    f"<div>🎯 % frame fake: <b>{pct_fake:.0f}%</b></div>"
                    f"<div>👤 Mặt phát hiện: <b>{r['n_face']}/{r['n_try']}</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Chart 1: Frame-by-frame timeline ─────────
            st.subheader("📈 Xác Suất Deepfake Theo Thời Gian (Frame-by-Frame)")
            st.caption(
                "Mỗi điểm = 1 frame được phân tích. Đường ngang đỏ = ngưỡng phân loại. "
                "Người thật: hầu hết điểm dưới ngưỡng. Deepfake: phần lớn điểm trên ngưỡng."
            )
            fig_tl = go.Figure()
            for label, r in results.items():
                fig_tl.add_trace(go.Scatter(
                    x=r["timestamps"], y=r["fake_probs"]*100,
                    name=label, mode="lines+markers",
                    line=dict(color=r["color"], width=2),
                    marker=dict(size=8, color=r["color"]),
                    fill="tozeroy", fillcolor=r["fill"],
                    hovertemplate="t=%{x:.2f}s<br>P(fake)=%{y:.1f}%<extra>" + label + "</extra>",
                ))
            fig_tl.add_hline(
                y=t4_threshold, line_dash="dash", line_color="#c0392b",
                annotation_text=f"Ngưỡng {t4_threshold}%", annotation_position="top right",
            )
            fig_tl.update_layout(
                xaxis_title="Thời gian (giây)",
                yaxis_title="P(fake) — Xác suất là deepfake (%)",
                yaxis_range=[0, 100],
                height=420, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

            # ── Chart 2: Histogram phân phối ─────────────
            st.subheader("📊 Phân Phối Xác Suất Fake")
            st.caption(
                "Histogram cho thấy mật độ phân phối P(fake) trên các frame. "
                "Người thật: dồn về phía trái. Deepfake: dồn về phía phải."
            )
            fig_hist = go.Figure()
            for label, r in results.items():
                fig_hist.add_trace(go.Histogram(
                    x=r["fake_probs"]*100, name=label, opacity=0.7,
                    marker_color=r["color"], nbinsx=20,
                ))
            fig_hist.add_vline(
                x=t4_threshold, line_dash="dash", line_color="#c0392b",
                annotation_text=f"Ngưỡng {t4_threshold}%",
            )
            fig_hist.update_layout(
                barmode="overlay",
                xaxis_title="P(fake) (%)", yaxis_title="Số frame",
                xaxis_range=[0, 100], height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # ── Chart 3: Bar chart so sánh các metric ────
            st.subheader("📐 So Sánh Các Chỉ Số Tổng Hợp")
            metrics_data = []
            for label, r in results.items():
                metrics_data.append({
                    "video": label,
                    "Mean P(fake) %":  float(np.mean(r["fake_probs"])) * 100,
                    "Max P(fake) %":   float(np.max(r["fake_probs"])) * 100,
                    "Median P(fake) %": float(np.median(r["fake_probs"])) * 100,
                    "% frame ≥ ngưỡng": float(np.mean(r["fake_probs"] >= t4_threshold/100.0)) * 100,
                })
            df_metrics = pd.DataFrame(metrics_data).set_index("video")

            fig_bar = go.Figure()
            metric_names = list(df_metrics.columns)
            for i, mname in enumerate(metric_names):
                fig_bar.add_trace(go.Bar(
                    x=df_metrics.index, y=df_metrics[mname],
                    name=mname,
                    text=[f"{v:.1f}%" for v in df_metrics[mname]],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                barmode="group", height=420,
                yaxis_title="Giá trị (%)", yaxis_range=[0, 110],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Sample faces ─────────────────────────────
            st.subheader("🖼️ Mẫu Khuôn Mặt Đã Phân Tích")
            st.caption("Hiển thị tối đa 6 mẫu mặt mỗi video kèm xác suất fake do AI dự đoán.")
            for label, r in results.items():
                st.markdown(f"**{label}**")
                n_show = min(6, len(r["faces"]))
                idxs = np.linspace(0, len(r["faces"]) - 1, n_show).astype(int)
                cols = st.columns(n_show)
                for ci, fi in enumerate(idxs):
                    p = float(r["fake_probs"][fi]) * 100
                    border = "#e74c3c" if p >= t4_threshold else "#27ae60"
                    cols[ci].image(r["faces"][fi], use_container_width=True)
                    cols[ci].markdown(
                        f"<div style='text-align:center;color:{border};font-weight:bold;"
                        f"border-top:2px solid {border};padding-top:4px'>"
                        f"P(fake)={p:.1f}%<br><small>t={r['timestamps'][fi]:.1f}s</small></div>",
                        unsafe_allow_html=True,
                    )

            # ── Bảng chi tiết ────────────────────────────
            st.divider()
            st.subheader("📋 Bảng Chi Tiết Theo Frame")
            for label, r in results.items():
                with st.expander(f"📄 Chi tiết — {label} ({r['n_face']} frame)", expanded=False):
                    df_detail = pd.DataFrame({
                        "Frame #":      np.arange(1, len(r["faces"]) + 1),
                        "Time (s)":     [f"{t:.2f}" for t in r["timestamps"]],
                        "P(fake) %":    [f"{p*100:.2f}" for p in r["fake_probs"]],
                        "P(real) %":    [f"{(1-p)*100:.2f}" for p in r["fake_probs"]],
                        "Verdict":      ["🔴 FAKE" if p*100 >= t4_threshold else "🟢 REAL"
                                         for p in r["fake_probs"]],
                    })
                    st.dataframe(df_detail, use_container_width=True, hide_index=True)

            # ── Diễn giải ────────────────────────────────
            st.divider()
            st.subheader("📖 Diễn Giải Kết Quả")
            with st.expander("🔍 Cách AI phát hiện deepfake & ý nghĩa từng chỉ số", expanded=True):
                st.markdown(f"""
**Cách hoạt động:**
1. Video được lấy mẫu **{t4_n_frames} frame** đều theo thời gian.
2. Mỗi frame được phát hiện khuôn mặt bằng **Haar Cascade**, sau đó crop kèm margin {int(TAB4_FACE_MARGIN*100)}%.
3. Mặt được resize/normalize về định dạng chuẩn của model (`{TAB4_MODEL_NAME}`).
4. Model trả về 2 xác suất: P(real) và P(fake) cho mỗi mặt.
5. Tổng hợp toàn video bằng **Mean P(fake)** so sánh với ngưỡng {t4_threshold}%.

**Các chỉ số:**
- **Mean P(fake)**: trung bình xác suất fake trên toàn bộ frame — chỉ số chính để verdict.
- **Max P(fake)**: frame "đáng ngờ nhất" — hữu ích để phát hiện deepfake một phần.
- **% frame ≥ ngưỡng**: tỷ lệ frame bị model gắn nhãn fake — đo độ nhất quán.
- **Median**: ít bị ảnh hưởng bởi vài frame outlier — dùng kiểm tra chéo Mean.

**Khi nào kết quả đáng tin?**
- Tỷ lệ phát hiện mặt cao (>80% frame).
- Khoảng cách Mean P(fake) giữa 2 video lớn (>30%).
- Phân phối tách bạch trên histogram.

**Khi nào nên nghi ngờ?**
- Tỷ lệ phát hiện mặt thấp → ánh sáng/góc mặt kém.
- Mean P(fake) gần ngưỡng → nằm trong vùng không chắc chắn.
- Phân phối overlap nhiều giữa 2 video → model không phân biệt được.
                """)

