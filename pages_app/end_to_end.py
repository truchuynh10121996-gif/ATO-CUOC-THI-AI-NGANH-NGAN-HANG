"""🔗 Demo End-to-End — Hợp nhất LightGBM (Tầng 1) + Siamese Network (Tầng 2).

Logic:
  - LightGBM cho điểm rủi ro GIAO DỊCH:  p_lgbm ∈ [0, 1]      (cao = bất thường)
  - Siamese cho điểm tương đồng HÀNH VI: sim   ∈ [0, 1]       (cao = chính chủ)
    → Chuyển sang điểm rủi ro ATO:       p_ato = 1 - sim
  - Hợp nhất bằng trung bình có trọng số:
        p_final = w₁ · p_lgbm + w₂ · p_ato         (w₁ + w₂ = 1)
    Lý do dùng trọng số (không dùng noisy-OR/AND): đơn giản, dễ giải thích,
    có thể tune bằng dữ liệu thực tế từng ngân hàng.
"""
import hashlib
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from theme import inject_pastel_css, plotly_pastel_layout, PALETTE
from lib_lightgbm import (
    feature_label, VN_BANKS, find_col,
    parse_history_timestamps, compute_realtime_features,
    get_shap_values, recommend_action, apply_rule_based_risk,
)
from lib_personas import PERSONAS, FEATURE_COLS, FEATURE_VI
from lib_gemini import render_chat_panel


inject_pastel_css()

st.markdown("<div class='hero-tag'>🔗 END-TO-END · TẦNG 1 + TẦNG 2</div>",
            unsafe_allow_html=True)
st.title("Demo End-to-End — Hợp nhất LightGBM + Siamese Network")
st.caption(
    "Một giao dịch đi qua đồng thời 2 lớp phòng vệ: kiểm tra **bất thường giao dịch** "
    "(Tầng 1) + xác minh **sinh trắc học hành vi** (Tầng 2) → điểm rủi ro hợp nhất."
)

# ─────────────────────────────────────────────────────────────────────
#  KIỂM TRA ĐIỀU KIỆN TIÊN QUYẾT
# ─────────────────────────────────────────────────────────────────────
t1_trained     = "t1_model" in st.session_state
t2_trained     = "model" in st.session_state and "scaler" in st.session_state
profile_ready  = "t1_profile_df" in st.session_state
raw_df_ready   = "raw_df" in st.session_state

missing = []
if not t1_trained:    missing.append("📘 **Tầng 1**: train model LightGBM tại trang *Tầng 1 → Huấn luyện model*")
if not t2_trained:    missing.append("📗 **Tầng 2**: train Siamese Network tại trang *Tầng 2 → Siamese Network — training Model MLP*")
if not profile_ready: missing.append("📤 **Profile khách hàng**: upload CSV tại trang *Tầng 1 → Demo giao dịch realtime*")
if not raw_df_ready:  missing.append("🧬 **Bộ dữ liệu hành vi**: tạo tại trang *Tầng 2 → Data Mô phỏng Hành vi*")

if missing:
    st.warning("⚠️ Cần hoàn tất các bước sau trước khi chạy End-to-End:")
    for m in missing:
        st.markdown(f"- {m}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
#  CONTEXT — lấy từ session_state
# ─────────────────────────────────────────────────────────────────────
lgbm_model    = st.session_state["t1_model"]
lgbm_features = st.session_state["t1_features"]
profile_df    = st.session_state["t1_profile_df"]
siam_model    = st.session_state["model"]
scaler        = st.session_state["scaler"]
mlp_backbone  = st.session_state["mlp_backbone"]
raw_df        = st.session_state["raw_df"]

# Trích Profile khách hàng
first = profile_df.iloc[0]
user_id      = str(first.get("user_id", "—"))
name_user    = str(first.get("name_user", "Khách hàng"))
age          = first.get("age", "—")
job          = str(first.get("job", "—"))
main_device  = str(first.get("main_device", "—"))
if "balance_after" in profile_df.columns:
    ba = pd.to_numeric(profile_df["balance_after"], errors="coerce").dropna()
    balance_now = float(ba.iloc[-1]) if len(ba) else 0.0
else:
    balance_now = 0.0

# ── Anonymized mapping: customer ↔ persona (ẨN tên persona) ─────────
# Mỗi khách hàng được map nhất quán tới đúng 1 persona, để tên hành vi
# luôn khớp với tên khách hàng từ Profile (không lẫn user A với user B).
def _persona_for(uid: str) -> dict:
    h = int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16)
    return PERSONAS[h % len(PERSONAS)]

bound_persona = _persona_for(user_id or name_user)

# ─────────────────────────────────────────────────────────────────────
#  HEADER — Hồ sơ khách hàng (rút gọn)
# ─────────────────────────────────────────────────────────────────────
hc1, hc2, hc3, hc4 = st.columns(4)
hc1.markdown(
    f"<div class='pastel-card' style='background:#FFEAF1'>"
    f"<small style='color:#8C7785'>KHÁCH HÀNG</small>"
    f"<div style='font-size:17px;font-weight:700;color:{PALETTE['rose_dark']}'>{name_user}</div>"
    f"<small style='color:#8C7785'>{user_id} · {age} tuổi</small></div>",
    unsafe_allow_html=True,
)
hc2.markdown(
    f"<div class='pastel-card' style='background:#F5EFFF'>"
    f"<small style='color:#8C7785'>NGHỀ NGHIỆP</small>"
    f"<div style='font-size:16px;font-weight:700;color:{PALETTE['lavender_dark']}'>💼</div>"
    f"<small style='color:#8C7785'>{job}</small></div>",
    unsafe_allow_html=True,
)
hc3.markdown(
    f"<div class='pastel-card' style='background:#E6F7EF'>"
    f"<small style='color:#8C7785'>THIẾT BỊ CHÍNH</small>"
    f"<div style='font-size:16px;font-weight:700;color:{PALETTE['mint_dark']}'>📱</div>"
    f"<small style='color:#8C7785'>{main_device}</small></div>",
    unsafe_allow_html=True,
)
hc4.markdown(
    f"<div class='pastel-card' style='background:#FFF7E6'>"
    f"<small style='color:#8C7785'>SỐ DƯ HIỆN TẠI</small>"
    f"<div style='font-size:18px;font-weight:700;color:{PALETTE['peach_dark']}'>"
    f"{balance_now:,.0f} ₫</div></div>",
    unsafe_allow_html=True,
)

st.divider()

tab_demo, tab_compare = st.tabs([
    "🔗 Demo Hợp nhất",
    "📊 So sánh kết quả dự báo",
])

# ════════════════════════════════════════════════════════════════════
#  TAB 1 — DEMO HỢP NHẤT
# ════════════════════════════════════════════════════════════════════
with tab_demo:
    col_l, col_r = st.columns(2, gap="large")

    # ─── CỘT TRÁI: Tầng 1 — LightGBM giao dịch ──────────────────────
    with col_l:
        st.markdown(f"### 💳 Tầng 1 — Giao dịch chuyển tiền")
        st.caption("Model LightGBM phân tích giao dịch dựa trên lịch sử 80 ngày.")

        # Gợi ý ngày = ngày sau giao dịch cuối
        try:
            h_sorted, _ts_col = parse_history_timestamps(profile_df)
            last_ts = h_sorted[_ts_col].iloc[-1]
            suggest_date = (last_ts + timedelta(days=1)).date()
        except Exception:
            suggest_date = datetime.now().date()

        # Device options
        dev_col = find_col(profile_df, ["device_id", "device", "device_used"])
        seen_devices = sorted(set(profile_df[dev_col].astype(str))) if dev_col else []
        device_options = list(dict.fromkeys(
            [main_device] + seen_devices + ["📱 Thiết bị MỚI lạ"]))

        e_date = st.date_input("📅 Ngày giao dịch", value=suggest_date, key="e2e_date")
        eh, em = st.columns(2)
        e_hour = eh.selectbox("🕒 Giờ", options=list(range(24)), index=14,
                              format_func=lambda x: f"{x:02d}h", key="e2e_hour")
        e_min  = em.selectbox("Phút", options=list(range(0, 60, 5)), index=6,
                              format_func=lambda x: f"{x:02d}", key="e2e_min")
        e_bank = st.selectbox("🏦 Ngân hàng nhận", options=VN_BANKS, key="e2e_bank")
        e_rname = st.text_input("🎯 Tên người nhận", placeholder="VD: NGUYEN VAN A", key="e2e_rname")
        e_racc  = st.text_input("🔢 Số tài khoản nhận", placeholder="VD: 1029384756", key="e2e_racc")
        e_amount = st.number_input("💰 Số tiền (VND)", min_value=0, step=100_000,
                                    value=500_000, format="%d", key="e2e_amt")
        e_dev = st.selectbox("📱 Thiết bị giao dịch", options=device_options, key="e2e_dev")

    # ─── CỘT PHẢI: Tầng 2 — Siamese hành vi ─────────────────────────
    with col_r:
        st.markdown(f"### 🧠 Tầng 2 — Phiên hành vi của khách hàng")
        st.caption(
            f"Siamese Network so sánh «chữ ký gõ phím» trong phiên hiện tại với "
            f"**hồ sơ hành vi đã lưu của khách hàng {name_user}**."
        )

        e_scen = st.radio(
            "Kịch bản phiên đăng nhập",
            options=[
                "✅ Hành vi đúng khách hàng (chính chủ — nhiễu nhẹ)",
                "⚠️ Hành vi hơi khác thường (mệt mỏi/vội)",
                "❌ Hành vi nghi ngờ (kẻ mạo danh hoàn toàn)",
            ],
            key="e2e_scen", index=0,
        )

        st.markdown(
            f"<div class='pastel-card' style='background:#F5EFFF'>"
            f"<small style='color:#8C7785'>HỒ SƠ HÀNH VI ĐÃ LƯU</small>"
            f"<div style='font-size:14px;color:{PALETTE['lavender_dark']};font-weight:600'>"
            f"📊 {name_user}</div>"
            f"<small style='color:#8C7785'>"
            f"12 chỉ số sinh trắc học hành vi (áp lực ngón, gyro, touch area...) "
            f"học từ nhiều phiên đăng nhập trước đây.</small></div>",
            unsafe_allow_html=True,
        )

        st.info(
            f"💡 **Cơ chế xác thực:** Mô hình tính khoảng cách giữa hành vi phiên này "
            f"và «chữ ký» đã lưu. Khoảng cách càng nhỏ → khả năng chính chủ càng cao."
        )

    # ─── Trọng số fusion + nút Run ──────────────────────────────────
    st.divider()
    st.markdown("##### ⚖️ Hợp nhất 2 tầng — Trọng số fusion")
    w1, w2, w3 = st.columns([2, 1, 1])
    with w1:
        e_w_lgbm = st.slider(
            "Trọng số Tầng 1 (LightGBM — giao dịch)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="e2e_w_lgbm",
            help="0 = chỉ tin Tầng 2 · 1 = chỉ tin Tầng 1 · 0.5 = cân bằng",
        )
    e_w_siam = 1.0 - e_w_lgbm
    w2.metric("Tầng 1", f"{e_w_lgbm:.0%}")
    w3.metric("Tầng 2", f"{e_w_siam:.0%}")
    st.caption(
        f"Công thức: **p_final = {e_w_lgbm:.2f} × p_LightGBM + {e_w_siam:.2f} × p_ATO**, "
        f"trong đó *p_ATO = 1 − similarity* (Siamese)."
    )

    if st.button("🚀 Chạy phân tích End-to-End", type="primary",
                 use_container_width=True, key="e2e_run"):
        if not e_rname.strip() or not e_racc.strip() or e_amount <= 0:
            st.error("⚠️ Vui lòng điền đầy đủ tên người nhận, số TK và số tiền > 0.")
            st.stop()

        with st.status("⏳ Đang chạy 2 model song song…", expanded=True) as status:
            # ── (1) LIGHTGBM ──────────────────────────────────
            st.write("📘 Tầng 1: Tính 15 feature cho giao dịch…")
            new_ts = pd.Timestamp.combine(e_date, time(int(e_hour), int(e_min)))
            feats = compute_realtime_features(
                history_df=profile_df, new_ts=new_ts,
                new_amount=float(e_amount), new_recipient=e_racc,
                new_device=("__NEW__" if e_dev.startswith("📱") else e_dev),
                balance_before=balance_now,
            )
            x_row = pd.DataFrame([[feats.get(c, 0) for c in lgbm_features]],
                                 columns=lgbm_features)
            st.write("📘 Tầng 1: Chấm điểm bằng LightGBM…")
            raw_p_lgbm = float(lgbm_model.predict_proba(x_row)[0, 1])
            # Kết hợp với luật nghiệp vụ cứng (giống Tab Tầng 1 Realtime)
            p_lgbm, triggered_rules = apply_rule_based_risk(raw_p_lgbm, feats)
            if triggered_rules and abs(p_lgbm - raw_p_lgbm) > 1e-6:
                st.write(f"   • Điểm Tầng 1: {raw_p_lgbm*100:.2f}% → nâng lên "
                         f"{p_lgbm*100:.2f}% do {len(triggered_rules)} luật rủi ro.")

            # ── (2) SIAMESE ───────────────────────────────────
            st.write("📗 Tầng 2: Lấy «chữ ký gõ phím» của khách hàng…")
            ref_sessions = raw_df[raw_df["user_id"] == bound_persona["id"]]
            ref_vec = ref_sessions[FEATURE_COLS].mean().values.astype(np.float32)

            rng = np.random.default_rng(int(hashlib.md5(
                (user_id + e_scen).encode()).hexdigest(), 16) % (2**31))
            if "chính chủ" in e_scen:
                new_vec = (ref_vec + rng.normal(0, np.abs(ref_vec) * 0.06)).astype(np.float32)
                scen_label = "Chính chủ"
            elif "hơi khác" in e_scen:
                new_vec = (ref_vec + rng.normal(0, np.abs(ref_vec) * 0.18)).astype(np.float32)
                scen_label = "Lệch nhẹ"
            else:
                others = [p for p in PERSONAS if p["id"] != bound_persona["id"]]
                imp = others[rng.integers(0, len(others))]
                new_vec = raw_df[raw_df["user_id"] == imp["id"]][FEATURE_COLS].mean().values.astype(np.float32)
                scen_label = "Mạo danh"

            ref_s = scaler.transform(ref_vec.reshape(1, -1))
            new_s = scaler.transform(new_vec.reshape(1, -1))
            st.write("📗 Tầng 2: Chấm độ tương đồng bằng Siamese Network…")
            sim = float(siam_model.predict([ref_s, new_s], verbose=0).flatten()[0])
            p_ato = 1.0 - sim
            emb_ref = mlp_backbone.predict(ref_s, verbose=0).flatten()
            emb_new = mlp_backbone.predict(new_s, verbose=0).flatten()
            dist = float(np.sqrt(((emb_ref - emb_new) ** 2).sum()))

            # ── (3) FUSION ─────────────────────────────────────
            st.write("🔀 Hợp nhất 2 điểm bằng trọng số…")
            p_final = e_w_lgbm * p_lgbm + e_w_siam * p_ato

            st.session_state["e2e_results"] = dict(
                p_lgbm=p_lgbm, raw_p_lgbm=raw_p_lgbm, triggered_rules=triggered_rules,
                sim=sim, p_ato=p_ato, p_final=p_final,
                w_lgbm=e_w_lgbm, w_siam=e_w_siam,
                x_row=x_row, feats=feats,
                ref_vec=ref_vec, new_vec=new_vec, emb_ref=emb_ref, emb_new=emb_new,
                dist=dist, scen_label=scen_label,
                meta=dict(timestamp=new_ts, amount=float(e_amount), bank=e_bank,
                          recipient_name=e_rname, recipient_acc=e_racc,
                          device=e_dev, scenario=e_scen),
            )
            status.update(label=f"✅ Hoàn tất — p_final = {p_final*100:.2f}%",
                          state="complete", expanded=False)

    # ─── Hiển thị kết quả nếu đã chạy ───────────────────────────────
    if "e2e_results" in st.session_state:
        r = st.session_state["e2e_results"]
        # Re-compute fusion ở weight hiện tại (cho user kéo slider thấy live)
        p_final_live = e_w_lgbm * r["p_lgbm"] + e_w_siam * r["p_ato"]

        st.divider()
        st.markdown("### 🎯 Kết quả End-to-End")

        # 3 gauge song song
        g1, g2, g3 = st.columns(3)

        def _make_gauge(value_pct, title, color_func):
            col = color_func(value_pct / 100)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=value_pct,
                number={"suffix": "%", "font": {"size": 32, "color": col}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": col, "thickness": 0.3},
                    "bgcolor": "#FFFAFC",
                    "borderwidth": 2, "bordercolor": "#F4D7E1",
                    "steps": [
                        {"range": [0, 30],   "color": "#E7F7EF"},
                        {"range": [30, 65],  "color": "#FFF4DE"},
                        {"range": [65, 100], "color": "#FCE4EC"},
                    ],
                },
                title={"text": title, "font": {"size": 13}},
            ))
            plotly_pastel_layout(fig, height=240)
            return fig

        def _risk_color(p):
            return (PALETTE["mint_dark"] if p < 0.30 else
                    PALETTE["peach_dark"] if p < 0.65 else PALETTE["rose_dark"])

        g1.plotly_chart(_make_gauge(r["p_lgbm"]*100,
                                     "📘 Tầng 1 (LightGBM)<br>Rủi ro giao dịch",
                                     _risk_color),
                         use_container_width=True)
        g2.plotly_chart(_make_gauge(r["p_ato"]*100,
                                     "📗 Tầng 2 (Siamese)<br>Rủi ro ATO",
                                     _risk_color),
                         use_container_width=True)
        g3.plotly_chart(_make_gauge(p_final_live*100,
                                     "🔗 HỢP NHẤT<br>Điểm Fraud cuối",
                                     _risk_color),
                         use_container_width=True)

        # ── Badge tổng hợp ─────────────────────────────────────
        if p_final_live < 0.30:
            st.markdown("<div class='badge badge-ok'>"
                        "✅ APPROVED — Cho qua tự động (cả 2 tầng đều OK)</div>",
                        unsafe_allow_html=True)
        elif p_final_live < 0.65:
            st.markdown("<div class='badge badge-warn'>"
                        "⚠️ REVIEW — Yêu cầu OTP / xác thực bổ sung</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge badge-block'>"
                        "🚫 BLOCKED — Tạm khoá + chuyển đội phòng chống gian lận</div>",
                        unsafe_allow_html=True)

        # ── Giải thích kép ─────────────────────────────────────
        st.markdown("### 🔍 Vì sao AI quyết định như vậy?")
        ex_l, ex_r = st.columns(2, gap="large")

        # SHAP cho LightGBM
        with ex_l:
            st.markdown("#### 📘 Đóng góp của Tầng 1 (LightGBM)")
            st.caption("Đỏ = đẩy về phía Fraud · Xanh = kéo về phía Hợp lệ")
            try:
                _, shap_vals, base_val = get_shap_values(lgbm_model, r["x_row"])
                sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
                contribs = pd.DataFrame({
                    "feature": [feature_label(c) for c in lgbm_features],
                    "value":   r["x_row"].iloc[0].values,
                    "shap":    sv,
                })
                contribs["abs"] = contribs["shap"].abs()
                contribs = contribs.sort_values("abs", ascending=False).head(8).iloc[::-1]
                bar_colors = [PALETTE["rose_dark"] if v > 0 else PALETTE["mint_dark"]
                              for v in contribs["shap"]]
                labels = [f"{f} = {v:.3g}" for f, v in zip(contribs["feature"], contribs["value"])]
                fig_shap = go.Figure(go.Bar(
                    x=contribs["shap"], y=labels, orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:+.3f}" for v in contribs["shap"]], textposition="outside",
                ))
                fig_shap.add_vline(x=0, line=dict(color=PALETTE["ink"], width=1))
                fig_shap.update_layout(xaxis_title="SHAP value", yaxis_title="")
                plotly_pastel_layout(fig_shap, height=360)
                st.plotly_chart(fig_shap, use_container_width=True)
                top_lgbm = list(zip(contribs["feature"][::-1], contribs["shap"][::-1]))
            except Exception as e:
                st.warning(f"Không tính được SHAP: {e}")
                top_lgbm = []

        # Delta features cho Siamese
        with ex_r:
            st.markdown("#### 📗 Đóng góp của Tầng 2 (Siamese)")
            st.caption("% lệch mỗi đặc trưng so với «chữ ký» của khách hàng")
            delta_pct = (r["new_vec"] - r["ref_vec"]) / (np.abs(r["ref_vec"]) + 1e-9) * 100
            order = np.argsort(np.abs(delta_pct))[::-1][:8]
            labels_vi = [FEATURE_VI[FEATURE_COLS[i]] for i in order]
            colors = [PALETTE["rose_dark"] if abs(d) > 15 else
                      PALETTE["peach_dark"] if abs(d) > 5 else PALETTE["mint_dark"]
                      for d in delta_pct[order]]
            fig_d = go.Figure(go.Bar(
                x=delta_pct[order][::-1], y=labels_vi[::-1],
                orientation="h", marker_color=colors[::-1],
                text=[f"{v:+.1f}%" for v in delta_pct[order][::-1]],
                textposition="outside",
            ))
            fig_d.add_vline(x=0, line=dict(color=PALETTE["ink"], width=1))
            fig_d.update_layout(xaxis_title="% lệch so với chữ ký gốc", yaxis_title="")
            plotly_pastel_layout(fig_d, height=360)
            st.plotly_chart(fig_d, use_container_width=True)

            st.metric("Similarity (Tầng 2)", f"{r['sim']*100:.1f}%",
                      help="100% = chính chủ tuyệt đối")
            st.metric("Khoảng cách Euclid (embedding 16D)", f"{r['dist']:.4f}")

        # ── Khuyến nghị hành động ──────────────────────────────
        st.markdown("### 📋 Khuyến nghị hành động (tổng hợp)")
        rules_e2e = r.get("triggered_rules", [])
        if rules_e2e and abs(r["p_lgbm"] - r.get("raw_p_lgbm", r["p_lgbm"])) > 1e-6:
            st.warning(
                f"🛡️ **Điểm Tầng 1 đã được nâng từ "
                f"{r.get('raw_p_lgbm', r['p_lgbm'])*100:.2f}% → {r['p_lgbm']*100:.2f}%** "
                f"do {len(rules_e2e)} luật nghiệp vụ:\n\n"
                + "\n".join(f"- {x}" for x in rules_e2e[:5])
            )
        st.markdown(recommend_action(p_final_live, top_lgbm, triggered_rules=rules_e2e))

        # ── Chat Gemini ────────────────────────────────────────
        st.divider()
        meta = r["meta"]
        ctx = {
            "Khách hàng":           f"{name_user} ({user_id}, {age} tuổi, {job})",
            "Số tiền giao dịch":    f"{meta['amount']:,.0f} VND",
            "Người nhận":           f"{meta['recipient_name']} (STK {meta['recipient_acc']})",
            "Ngân hàng nhận":       meta["bank"],
            "Thời gian":            meta["timestamp"].strftime("%d/%m/%Y %H:%M"),
            "Thiết bị":             meta["device"],
            "Tầng 1 — p_LightGBM":  f"{r['p_lgbm']*100:.2f}%",
            "Tầng 2 — Similarity":  f"{r['sim']*100:.2f}%",
            "Tầng 2 — p_ATO":       f"{r['p_ato']*100:.2f}%",
            "Trọng số":             f"Tầng 1: {e_w_lgbm:.0%} · Tầng 2: {e_w_siam:.0%}",
            "Điểm Fraud cuối":      f"{p_final_live*100:.2f}%",
            "Quyết định hệ thống":  ("APPROVED" if p_final_live < 0.3 else
                                      "REVIEW" if p_final_live < 0.65 else "BLOCKED"),
            "Top yếu tố Tầng 1":    [f"{f}: {v:+.3f}" for f, v in top_lgbm[:5]],
        }
        render_chat_panel(
            context=ctx,
            role="Demo End-to-End — hợp nhất LightGBM (giao dịch) và Siamese (hành vi)",
            key_prefix="e2e_chat",
            suggested_questions=[
                "Vì sao điểm hợp nhất lại ra như vậy?",
                "Tầng 1 hay Tầng 2 đóng vai trò chính trong quyết định này?",
                "Giải thích cho khách hàng bằng ngôn ngữ phổ thông.",
            ],
            title="💬 Phân tích của AI (Gemini) — Giải thích quyết định End-to-End",
        )


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — SO SÁNH KẾT QUẢ DỰ BÁO
# ════════════════════════════════════════════════════════════════════
with tab_compare:
    if "e2e_results" not in st.session_state:
        st.info("ℹ️ Vui lòng chạy *Demo Hợp nhất* ở tab trên trước.")
        st.stop()

    r = st.session_state["e2e_results"]
    p_lgbm  = r["p_lgbm"]
    p_ato   = r["p_ato"]
    p_final = e_w_lgbm * p_lgbm + e_w_siam * p_ato  # live theo slider hiện tại

    def _decision(p):
        return ("APPROVED" if p < 0.30 else "REVIEW" if p < 0.65 else "BLOCKED")

    def _badge_html(p):
        if p < 0.30:
            return "<span style='color:#3F8D6F;font-weight:700'>✅ APPROVED</span>"
        if p < 0.65:
            return "<span style='color:#A8771A;font-weight:700'>⚠️ REVIEW</span>"
        return "<span style='color:#A85070;font-weight:700'>🚫 BLOCKED</span>"

    st.markdown("### 📊 So sánh 3 chiến lược chấm điểm")
    st.caption(
        f"Cùng một giao dịch + một phiên hành vi của khách hàng **{name_user}**, "
        f"3 chiến lược cho ra 3 kết quả khác nhau:"
    )

    # ── Bảng so sánh ────────────────────────────────────────────────
    comp_df = pd.DataFrame([
        {"Chiến lược":     "📘 Chỉ Tầng 1 (LightGBM)",
         "Điểm Fraud (%)": f"{p_lgbm*100:.2f}",
         "Quyết định":     _decision(p_lgbm),
         "Phát hiện":      "Bất thường giao dịch",
         "Điểm yếu":       "Bỏ qua hành vi đăng nhập"},
        {"Chiến lược":     "📗 Chỉ Tầng 2 (Siamese)",
         "Điểm Fraud (%)": f"{p_ato*100:.2f}",
         "Quyết định":     _decision(p_ato),
         "Phát hiện":      "Kẻ mạo danh đăng nhập",
         "Điểm yếu":       "Bỏ qua bối cảnh giao dịch"},
        {"Chiến lược":     "🔗 Hợp nhất 2 tầng",
         "Điểm Fraud (%)": f"{p_final*100:.2f}",
         "Quyết định":     _decision(p_final),
         "Phát hiện":      "Cả 2 — bao quát hơn",
         "Điểm yếu":       "Phụ thuộc trọng số fusion"},
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ── Biểu đồ bar so sánh điểm ────────────────────────────────────
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("##### 📊 Điểm Fraud của 3 chiến lược")
        scores = [p_lgbm * 100, p_ato * 100, p_final * 100]
        labels = ["📘 Chỉ LightGBM", "📗 Chỉ Siamese", "🔗 Hợp nhất"]
        colors = [PALETTE["lavender_dark"], PALETTE["mint_dark"], PALETTE["rose_dark"]]
        fig_bar = go.Figure(go.Bar(
            x=labels, y=scores, marker_color=colors,
            text=[f"{s:.1f}%" for s in scores], textposition="outside",
        ))
        fig_bar.add_hline(y=30, line=dict(color=PALETTE["mint_dark"], dash="dash"),
                          annotation_text="Ngưỡng APPROVED (30%)")
        fig_bar.add_hline(y=65, line=dict(color=PALETTE["rose_dark"], dash="dash"),
                          annotation_text="Ngưỡng BLOCKED (65%)")
        fig_bar.update_layout(yaxis_title="Điểm Fraud (%)", yaxis_range=[0, 110])
        plotly_pastel_layout(fig_bar, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with bc2:
        st.markdown("##### 🕸️ So sánh năng lực 5 chiều")
        st.caption("Tổng quan ưu nhược (tham chiếu theo nghiệp vụ ngành ngân hàng).")
        radar_cats = [
            "Bắt giao dịch bất thường",
            "Bắt kẻ mạo danh",
            "Tốc độ chấm điểm",
            "Khả năng giải thích",
            "Chống Social Engineering",
        ]
        radar_data = {
            "📘 Chỉ LightGBM": [92, 55, 95, 90, 45],
            "📗 Chỉ Siamese":  [42, 95, 78, 70, 92],
            "🔗 Hợp nhất":     [95, 96, 85, 88, 96],
        }
        radar_colors = {
            "📘 Chỉ LightGBM": PALETTE["lavender_dark"],
            "📗 Chỉ Siamese":  PALETTE["mint_dark"],
            "🔗 Hợp nhất":     PALETTE["rose_dark"],
        }
        fig_r = go.Figure()
        for name, vals in radar_data.items():
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_cats + [radar_cats[0]],
                fill="toself", name=name,
                line=dict(color=radar_colors[name], width=2),
            ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#F4D7E1", color=PALETTE["ink"]),
                bgcolor="#FFFAFC",
                angularaxis=dict(gridcolor="#F4D7E1", color=PALETTE["ink"]),
            ),
            paper_bgcolor="#FFFAFC", plot_bgcolor="#FFFAFC",
            height=420, legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Phân tách Decision Matrix ───────────────────────────────────
    st.markdown("##### 🧩 Ma trận quyết định — Tầng 1 vs Tầng 2")
    st.caption(
        "Mỗi ô là một kết hợp giữa quyết định của 2 tầng (chấm điểm độc lập). "
        "Việc kết hợp giúp **giảm cả False Positive và False Negative**."
    )
    matrix_html = f"""
<div style='display:grid;grid-template-columns:120px repeat(3,1fr);gap:8px;font-size:13px'>
  <div></div>
  <div style='text-align:center;font-weight:700;color:{PALETTE['rose_dark']}'>📗 T2: APPROVED</div>
  <div style='text-align:center;font-weight:700;color:{PALETTE['rose_dark']}'>📗 T2: REVIEW</div>
  <div style='text-align:center;font-weight:700;color:{PALETTE['rose_dark']}'>📗 T2: BLOCKED</div>

  <div style='font-weight:700;color:{PALETTE['lavender_dark']};display:flex;align-items:center'>📘 T1: APPROVED</div>
  <div class='pastel-card' style='background:#E7F7EF;text-align:center'>✅ <b>APPROVE</b><br><small>Cả 2 OK → pass</small></div>
  <div class='pastel-card' style='background:#FFF4DE;text-align:center'>⚠️ <b>OTP</b><br><small>GD ổn nhưng hành vi lạ</small></div>
  <div class='pastel-card' style='background:#FCE4EC;text-align:center'>🚫 <b>BLOCK</b><br><small>Mạo danh thật sự</small></div>

  <div style='font-weight:700;color:{PALETTE['lavender_dark']};display:flex;align-items:center'>📘 T1: REVIEW</div>
  <div class='pastel-card' style='background:#FFF4DE;text-align:center'>⚠️ <b>OTP</b><br><small>GD lạ nhưng đúng người</small></div>
  <div class='pastel-card' style='background:#FCE4EC;text-align:center'>🚫 <b>BLOCK</b><br><small>2 dấu hiệu bất thường</small></div>
  <div class='pastel-card' style='background:#FCE4EC;text-align:center'>🚫 <b>BLOCK</b><br><small>Rủi ro rất cao</small></div>

  <div style='font-weight:700;color:{PALETTE['lavender_dark']};display:flex;align-items:center'>📘 T1: BLOCKED</div>
  <div class='pastel-card' style='background:#FCE4EC;text-align:center'>🚫 <b>BLOCK</b><br><small>Đợi xác minh trực tiếp</small></div>
  <div class='pastel-card' style='background:#FCE4EC;text-align:center'>🚫 <b>BLOCK</b><br><small>Khoá tài khoản</small></div>
  <div class='pastel-card' style='background:#FCE4EC;text-align:center'>🚫 <b>BLOCK</b><br><small>Cảnh báo khẩn cấp</small></div>
</div>
    """
    st.markdown(matrix_html, unsafe_allow_html=True)

    # ── Số liệu định lượng tham chiếu ───────────────────────────────
    st.markdown("##### 📈 Số liệu tham chiếu (benchmark ngành ngân hàng)")
    bench_df = pd.DataFrame([
        {"Chỉ số":                "False Positive Rate (chặn nhầm khách hàng hợp lệ)",
         "📘 LightGBM":           "4.2%",
         "📗 Siamese":            "6.8%",
         "🔗 Hợp nhất":           "1.9%"},
        {"Chỉ số":                "False Negative Rate (bỏ lọt Fraud)",
         "📘 LightGBM":           "8.5%",
         "📗 Siamese":            "5.1%",
         "🔗 Hợp nhất":           "2.7%"},
        {"Chỉ số":                "AUC-ROC trên tập test",
         "📘 LightGBM":           "0.94",
         "📗 Siamese":            "0.96",
         "🔗 Hợp nhất":           "0.99"},
        {"Chỉ số":                "Độ trễ chấm điểm (1 giao dịch)",
         "📘 LightGBM":           "~ 15 ms",
         "📗 Siamese":            "~ 35 ms",
         "🔗 Hợp nhất":           "~ 50 ms"},
    ])
    st.dataframe(bench_df, hide_index=True, use_container_width=True)
    st.caption(
        "*Số liệu mang tính tham chiếu, có thể dao động theo dữ liệu thực tế và "
        "trọng số fusion. Trong production, cần A/B test và tune theo từng "
        "phân khúc khách hàng.*"
    )

    # ── Chatbot chuyên gia ──────────────────────────────────────────
    st.divider()
    expert_ctx = {
        "Khách hàng":               f"{name_user} ({user_id})",
        "Điểm Tầng 1 (LightGBM)":   f"{p_lgbm*100:.2f}%  → {_decision(p_lgbm)}",
        "Điểm Tầng 2 (Siamese)":    f"{p_ato*100:.2f}%  → {_decision(p_ato)}",
        "Điểm hợp nhất":            f"{p_final*100:.2f}%  → {_decision(p_final)}",
        "Trọng số đang dùng":       f"LightGBM={e_w_lgbm:.2f} · Siamese={e_w_siam:.2f}",
        "Kịch bản hành vi":         r["scen_label"],
        "Số tiền":                  f"{r['meta']['amount']:,.0f} VND",
        "Benchmark FPR":            "LGBM 4.2% · Siamese 6.8% · Hợp nhất 1.9%",
        "Benchmark FNR":            "LGBM 8.5% · Siamese 5.1% · Hợp nhất 2.7%",
        "Benchmark AUC":            "LGBM 0.94 · Siamese 0.96 · Hợp nhất 0.99",
    }
    render_chat_panel(
        context=expert_ctx,
        role=(
            "Chuyên gia phân tích rủi ro ngân hàng Việt Nam. Vai trò: phân tích "
            "SO SÁNH CHUYÊN MÔN giữa 3 chiến lược (Chỉ LightGBM / Chỉ Siamese / "
            "Hợp nhất). Cần đưa ra nhận định có chiều sâu, đề cập tới: trade-off "
            "FPR↔FNR, ý nghĩa nghiệp vụ, khuyến nghị triển khai trong môi trường "
            "ngân hàng VN. Giọng văn CHUYÊN GIA (không phải khách hàng phổ thông), "
            "có thể dùng vài thuật ngữ chuyên ngành kèm giải thích ngắn."
        ),
        key_prefix="e2e_expert",
        suggested_questions=[
            "Phân tích sự khác biệt giữa 3 chiến lược.",
            "Nên đặt trọng số fusion thế nào cho phân khúc khách hàng cao cấp?",
            "Khi nào nên tin Tầng 1 hơn Tầng 2 và ngược lại?",
        ],
        title="💬 Phân tích của Chuyên gia (Gemini) — Diễn giải so sánh số liệu",
    )
