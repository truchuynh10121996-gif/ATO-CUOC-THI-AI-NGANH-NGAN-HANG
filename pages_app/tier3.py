"""🫀 Tầng 3 — rPPG phát hiện nhịp tim qua da chống DeepFake."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from theme import inject_pastel_css, plotly_pastel_layout, PALETTE
from lib_rppg import (
    save_video, extract_multi_roi, combine_rois, bandpass,
    multiwindow_hr, quality, safe_unlink,
)
from lib_gemini import render_chat_panel

inject_pastel_css()

st.markdown("<div class='hero-tag'>🫀 TẦNG 3 · rPPG ANTI-DEEPFAKE</div>", unsafe_allow_html=True)
st.title("Công nghệ rPPG phát hiện nhịp tim qua da chống DeepFake")
st.caption(
    "Phát hiện sự sống thật từ camera điện thoại — DeepFake không có nhịp tim "
    "nên sẽ bị từ chối ngay tại bước eKYC."
)

tab_concept, tab_demo = st.tabs([
    "📚 Nguyên lý hoạt động rPPG",
    "🎬 Demo Công nghệ rPPG",
])

# ════════════════════════════════════════════════════════════════════
#  TAB 1 — NGUYÊN LÝ
# ════════════════════════════════════════════════════════════════════
with tab_concept:
    st.markdown("#### 💡 rPPG là gì?")
    st.markdown(
        "**rPPG** *(remote PhotoPlethysmoGraphy)* — kỹ thuật **đo nhịp tim từ xa** "
        "qua camera RGB thông thường, không cần thiết bị y tế."
    )

    # Infographic: 4 bước
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "💓", "Tim đập",
         "Mỗi nhịp tim → máu được bơm tới mao mạch da mặt → áp lực thay đổi tuần hoàn."),
        ("2️⃣", "🌈", "Da đổi màu vi tế",
         "Hemoglobin trong máu hấp thụ mạnh ánh sáng XANH (Green) → da hơi đậm/nhạt theo nhịp."),
        ("3️⃣", "📷", "Camera ghi",
         "Camera RGB điện thoại nhạy đủ để ghi lại sự dao động ~1% kênh Green ở mặt."),
        ("4️⃣", "🤖", "AI phân tích",
         "Lọc tần số 0.75-3 Hz (45-180 BPM) + POS + Multi-ROI → ra nhịp tim ước tính."),
    ]
    cols = [c1, c2, c3, c4]
    grad_colors = [
        ("#FFEAF1", "#F7B8C9"),
        ("#F5EFFF", "#D7C4F2"),
        ("#E6F7EF", "#BFE7D6"),
        ("#FFF7E6", "#FFD6B8"),
    ]
    for col, (num, ico, title, desc), (g1, g2) in zip(cols, steps, grad_colors):
        col.markdown(
            f"""
<div class='pastel-card' style='background:linear-gradient(135deg,{g1} 0%,{g2} 100%);text-align:center;min-height:230px'>
<div style='font-size:24px;color:#A85070;font-weight:700'>{num}</div>
<div style='font-size:42px;margin:6px 0'>{ico}</div>
<div style='font-weight:700;font-size:16px;color:#5B4A55'>{title}</div>
<div style='font-size:13px;color:#5B4A55;margin-top:6px'>{desc}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 🎯 Vì sao DeepFake không qua được rPPG?")

    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown(
            """
<div class='pastel-card' style='background:#E7F7EF'>
<h3 style='color:#3F8D6F'>✅ Người thật</h3>
<ul style='font-size:14px'>
<li>Có dòng máu thật chảy tới da mặt</li>
<li>Kênh Green dao động <b>tuần hoàn ~1Hz</b></li>
<li>Xuất hiện <b>đỉnh phổ rõ rệt</b> tại 0.75–3 Hz</li>
<li>HR ổn định qua các cửa sổ thời gian</li>
</ul>
</div>
            """,
            unsafe_allow_html=True,
        )
    with cd2:
        st.markdown(
            """
<div class='pastel-card' style='background:#FCE4EC'>
<h3 style='color:#A85070'>❌ DeepFake</h3>
<ul style='font-size:14px'>
<li><b>Fully-synthetic</b> (AI tạo từ 0): không có máu → tín hiệu phẳng</li>
<li><b>Face-swap</b> (ghép mặt lên video): tín hiệu có nhưng <b>HR nhảy loạn</b></li>
<li>Phổ tần số <b>phẳng hoặc nhiễu trắng</b></li>
<li>Cải tiến của Tầng 3: <b>HR Stability</b> bắt được cả face-swap</li>
</ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 🔬 Cải tiến thuật toán so với rPPG cơ bản")
    st.markdown(
        """
| Thuật toán | rPPG cơ bản | **Tầng 3 (nâng cao)** |
|---|---|---|
| **Tín hiệu** | Kênh GREEN đơn thuần | **POS algorithm** (Wang 2017) — kết hợp 3 kênh RGB |
| **Vùng phân tích** | Chỉ trán | **Trán + Má trái + Má phải** (multi-ROI) |
| **Chỉ số chất lượng** | Peak Prominence | **Prominence + HR Stability** |
| **Phát hiện face-swap** | ❌ Yếu | ✅ Có (qua HR Stability) |
| **Tăng SNR** | 1× | **3–5×** |
        """
    )

    with st.expander("📐 Chi tiết kỹ thuật POS Algorithm", expanded=False):
        st.markdown(
            """
**POS** *(Plane-Orthogonal-to-Skin)* — Wang et al. 2017:
```
S1 = Rₙ - Gₙ
S2 = Rₙ + Gₙ - 2·Bₙ
α  = std(S1) / std(S2)
H  = S1 + α·S2     ← tín hiệu rPPG cuối
```
Phép chiếu trực giao với mặt phẳng phản xạ da → loại bỏ nhiễu chuyển động và
thay đổi ánh sáng → tăng SNR 3-5× so với GREEN đơn thuần.

**Multi-Window HR Stability** — chìa khoá phát hiện face-swap:
- Người thật: HR std ~3-8 BPM → Stability cao
- Face-swap: HR std ~15-40 BPM → Stability thấp dù prominence vẫn cao
            """
        )

    st.info(
        "🏦 **Ứng dụng trong eKYC ngân hàng:** Lớp rPPG đặt tại bước **xác thực "
        "video call** trước giao dịch lớn. Phải đạt cả 2 ngưỡng "
        "*Prominence* + *Stability* mới được duyệt."
    )


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — DEMO
# ════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("#### 🎬 Demo phân tích rPPG")
    st.caption("Upload video người thật và/hoặc DeepFake để so sánh nhịp tim ước tính.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            "<div style='background:#E7F7EF;border-left:4px solid "
            f"{PALETTE['mint_dark']};padding:10px;border-radius:8px'>"
            "<b>🟢 Video Người Thật</b></div>",
            unsafe_allow_html=True,
        )
        v_real = st.file_uploader("Upload video thật",
                                  type=["mp4", "avi", "mov", "mkv"], key="t3_real")
    with col_r:
        st.markdown(
            "<div style='background:#FCE4EC;border-left:4px solid "
            f"{PALETTE['rose_dark']};padding:10px;border-radius:8px'>"
            "<b>🔴 Video DeepFake</b></div>",
            unsafe_allow_html=True,
        )
        v_fake = st.file_uploader("Upload video deepfake",
                                  type=["mp4", "avi", "mov", "mkv"], key="t3_fake")

    st.divider()
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        max_f = st.slider("Số frame tối đa", 100, 600, 300, step=50, key="t3_maxf")
    with cc2:
        win = st.slider("Cửa sổ HR (giây)", 2, 8, 4, key="t3_win")
    with cc3:
        thr = st.slider("Ngưỡng 'Có sự sống' (%)", 5, 70, 35, key="t3_thr")

    if st.button("🔬 Phân Tích rPPG", type="primary", key="t3_run"):
        if v_real is None and v_fake is None:
            st.warning("⚠️ Vui lòng upload ít nhất 1 video.")
            st.stop()

        VIDEOS = []
        if v_real:
            VIDEOS.append(("Người Thật", v_real, PALETTE["mint_dark"], "rgba(127,203,169,0.20)"))
        if v_fake:
            VIDEOS.append(("DeepFake", v_fake, PALETTE["rose_dark"], "rgba(228,139,167,0.20)"))

        results = {}
        for label, vfile, color, fill in VIDEOS:
            with st.spinner(f"⏳ Đang xử lý {label}…"):
                tmp = save_video(vfile)
                try:
                    rois, fps, n_det, n_fr = extract_multi_roi(tmp, max_f)
                except Exception as e:
                    st.error(f"❌ Lỗi xử lý {label}: {e}")
                    safe_unlink(tmp); continue
                safe_unlink(tmp)

                raw_sig = combine_rois(rois)
                filt_sig = bandpass(raw_sig, fps)
                hr_arr, t_arr = multiwindow_hr(filt_sig, fps, float(win), 1.0)
                hr_bpm, q_prom, q_stab, q_comb, freqs, psd = quality(filt_sig, fps, hr_arr)
                time_ax = np.arange(len(filt_sig)) / fps

                results[label] = dict(
                    raw=raw_sig, filt=filt_sig, time=time_ax,
                    hr_arr=hr_arr, t_arr=t_arr, freqs=freqs, psd=psd,
                    hr_bpm=hr_bpm, q_prom=q_prom, q_stab=q_stab, q_comb=q_comb,
                    n_det=n_det, n_fr=n_fr, fps=fps, color=color, fill=fill,
                )
        if not results:
            st.error("❌ Không có kết quả."); st.stop()

        st.session_state["t3_results"] = results
        st.session_state["t3_thr"] = thr
        st.session_state["t3_win"] = win

    # ── Hiển thị kết quả nếu có ───────────────────────────────────
    if "t3_results" in st.session_state:
        results = st.session_state["t3_results"]
        thr = st.session_state["t3_thr"]
        win = st.session_state["t3_win"]

        st.success(f"✅ Đã phân tích {len(results)} video.")

        # ── Verdict cards ────────────────────────────────────────
        st.markdown("##### 📊 Kết quả tổng quan")
        vcols = st.columns(len(results))
        for i, (label, r) in enumerate(results.items()):
            alive = r["q_comb"] * 100 >= thr
            verdict = "✅ CÓ SỰ SỐNG" if alive else "❌ KHÔNG CÓ SỰ SỐNG"
            bg = "#E7F7EF" if alive else "#FCE4EC"
            bd = PALETTE["mint_dark"] if alive else PALETTE["rose_dark"]
            hr_disp = "N/A" if r["hr_bpm"] < 30 else f"{r['hr_bpm']:.0f} BPM"
            hr_std = float(r["hr_arr"].std()) if len(r["hr_arr"]) >= 2 else 0.0
            vcols[i].markdown(
                f"""
<div style='background:{bg};border:2px solid {bd};border-radius:14px;padding:18px;text-align:center'>
<div style='font-size:17px;font-weight:bold'>{label}</div>
<div style='font-size:26px;font-weight:bold;color:{bd};margin:6px 0'>{verdict}</div>
<hr style='margin:8px 0;border-color:{bd};opacity:.3'>
<div>💓 Nhịp tim: <b>{hr_disp}</b></div>
<div>🔍 HR Stability: <b>{r['q_stab']*100:.1f}%</b> <small>(std={hr_std:.1f})</small></div>
<div>📶 Peak Prominence: <b>{r['q_prom']*100:.1f}%</b></div>
<div>⚡ Tổng hợp: <b>{r['q_comb']*100:.1f}%</b></div>
<div>🎯 Phát hiện mặt: {r['n_det']}/{r['n_fr']} frame</div>
</div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Chart 1: POS Signal ───────────────────────────────────
        st.markdown("##### 📈 Tín hiệu rPPG (POS Algorithm — Đa Vùng Da)")
        fig1 = go.Figure()
        for label, r in results.items():
            fig1.add_trace(go.Scatter(
                x=r["time"], y=r["filt"], name=label, mode="lines",
                line=dict(color=r["color"], width=2),
                fill="tozeroy", fillcolor=r["fill"],
            ))
        fig1.update_layout(xaxis_title="Thời gian (s)", yaxis_title="Biên độ POS",
                           hovermode="x unified",
                           legend=dict(orientation="h", y=1.05))
        plotly_pastel_layout(fig1, height=360)
        st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2: PSD ──────────────────────────────────────────
        st.markdown("##### 📊 Phổ tần số (PSD)")
        fig2 = go.Figure()
        for label, r in results.items():
            mask = (r["freqs"] > 0) & (r["freqs"] <= 4.0)
            fig2.add_trace(go.Scatter(
                x=r["freqs"][mask] * 60, y=r["psd"][mask],
                name=label, mode="lines",
                line=dict(color=r["color"], width=2),
                fill="tozeroy", fillcolor=r["fill"],
            ))
        fig2.add_vrect(x0=45, x1=180, fillcolor="rgba(245,199,126,0.20)",
                       layer="below", line_width=0,
                       annotation_text="Vùng nhịp tim (45-180 BPM)",
                       annotation_position="top left")
        fig2.update_layout(xaxis_title="Tần số (BPM)", yaxis_title="PSD",
                           hovermode="x unified",
                           legend=dict(orientation="h", y=1.05))
        plotly_pastel_layout(fig2, height=360)
        st.plotly_chart(fig2, use_container_width=True)

        # ── Chart 3: HR timeline ──────────────────────────────────
        st.markdown(f"##### ⏱️ Nhịp tim theo thời gian (Multi-Window {win}s)")
        fig3 = go.Figure()
        for label, r in results.items():
            if len(r["hr_arr"]) == 0:
                continue
            fig3.add_trace(go.Scatter(
                x=r["t_arr"], y=r["hr_arr"], name=label,
                mode="lines+markers",
                line=dict(color=r["color"], width=2.5),
                marker=dict(size=8, color=r["color"]),
            ))
        fig3.add_hrect(y0=45, y1=180, fillcolor="rgba(245,199,126,0.15)",
                       layer="below", line_width=0)
        fig3.update_layout(xaxis_title="Thời gian (s)", yaxis_title="HR (BPM)",
                           hovermode="x unified",
                           legend=dict(orientation="h", y=1.05))
        plotly_pastel_layout(fig3, height=380)
        st.plotly_chart(fig3, use_container_width=True)

        # ── Chart 4: Radar ────────────────────────────────────────
        st.markdown("##### 🕸️ Biểu đồ Radar — So sánh toàn diện")
        cats = ["Peak Prominence", "HR Stability", "Tổng hợp"]
        fig5 = go.Figure()
        for label, r in results.items():
            vals = [r["q_prom"]*100, r["q_stab"]*100, r["q_comb"]*100]
            fig5.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=label,
                line=dict(color=r["color"], width=2),
                fillcolor=r["fill"],
            ))
        fig5.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#F4D7E1", color=PALETTE["ink"]),
                bgcolor="#FFFAFC",
                angularaxis=dict(gridcolor="#F4D7E1", color=PALETTE["ink"]),
            ),
            paper_bgcolor="#FFFAFC", plot_bgcolor="#FFFAFC",
            height=420,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig5, use_container_width=True)

        # ── Bảng tổng hợp ─────────────────────────────────────────
        st.markdown("##### 📋 Bảng tổng hợp các chỉ số")
        rows = []
        for label, r in results.items():
            hr_std_v = float(r["hr_arr"].std()) if len(r["hr_arr"]) >= 2 else 0.0
            alive = r["q_comb"] * 100 >= thr
            rows.append({
                "Video": label,
                "HR (BPM)": f"{r['hr_bpm']:.0f}" if r['hr_bpm'] >= 30 else "N/A",
                "HR Std": f"{hr_std_v:.1f}",
                "Peak Prominence": f"{r['q_prom']*100:.1f}%",
                "HR Stability":   f"{r['q_stab']*100:.1f}%",
                "Tổng hợp":       f"{r['q_comb']*100:.1f}%",
                "Verdict":        "✅ Có sự sống" if alive else "❌ DeepFake",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Video"), use_container_width=True)

        # ── Khung chat Gemini ────────────────────────────────────
        st.divider()
        chat_context = {
            "Số video phân tích": len(results),
            **{
                f"[{label}] Verdict": (
                    "Có sự sống" if r["q_comb"]*100 >= thr else "Bị nghi DeepFake"
                ) for label, r in results.items()
            },
            **{
                f"[{label}] HR (BPM)": f"{r['hr_bpm']:.0f}" if r['hr_bpm'] >= 30 else "N/A"
                for label, r in results.items()
            },
            **{
                f"[{label}] Peak Prominence (%)": f"{r['q_prom']*100:.1f}"
                for label, r in results.items()
            },
            **{
                f"[{label}] HR Stability (%)": f"{r['q_stab']*100:.1f}"
                for label, r in results.items()
            },
            "Ngưỡng quyết định (%)": thr,
        }
        render_chat_panel(
            context=chat_context,
            role="Phát hiện DeepFake bằng rPPG (Tầng 3) trong eKYC ngân hàng",
            key_prefix="t3_chat",
            suggested_questions=[
                "Vì sao video này bị nghi DeepFake?",
                "Khách hàng cần làm gì khi bị từ chối eKYC?",
                "Giải thích đơn giản cho khách hàng nghĩa là gì?",
            ],
            title="💬 Phân tích của AI (Gemini) — Giải thích kết quả rPPG",
        )
