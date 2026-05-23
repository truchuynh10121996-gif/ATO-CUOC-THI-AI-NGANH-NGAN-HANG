"""💳 Tầng 1 — Phát hiện giao dịch Fraud ATO bằng LightGBM."""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from theme import inject_pastel_css, matplotlib_pastel, plotly_pastel_layout, PALETTE, PALETTE_LIST
from datetime import datetime, time, timedelta
from lib_lightgbm import (
    DEFAULT_FRAUD_FEATURES, TARGET_COL, FEATURE_VI, feature_label,
    split_features, train_lightgbm, evaluate, get_shap_values,
    find_optimal_threshold, recommend_action,
    VN_BANKS, find_col, parse_history_timestamps, compute_realtime_features,
)
from lib_gemini import render_chat_panel

inject_pastel_css()
matplotlib_pastel()

st.markdown("<div class='hero-tag'>💳 AI HỖ TRỢ DEMO SIAMESE NETWORK</div>", unsafe_allow_html=True)
st.title("Phát hiện giao dịch Fraud ATO")
st.caption(
    "Mỗi giao dịch chuyển tiền được chấm rủi ro 0–100% bằng LightGBM — "
    "kèm SHAP giải thích vì sao và Gemini diễn giải bằng tiếng Việt phổ thông."
)

tab_train, tab_demo = st.tabs([
    "🏋️ Huấn luyện model",
    "⚡ Demo giao dịch realtime",
])

# ════════════════════════════════════════════════════════════════════
#  TAB 1 — HUẤN LUYỆN
# ════════════════════════════════════════════════════════════════════
with tab_train:
    with st.expander("📘 Vì sao chọn LightGBM?", expanded=False):
        st.markdown("""
**LightGBM** (Light Gradient Boosting Machine — Microsoft 2017) là thuật toán
gradient boosting trên cây quyết định, **dẫn đầu** các bài toán phát hiện gian
lận giao dịch ngân hàng nhờ:

| Ưu điểm | Ý nghĩa thực tế |
|---|---|
| **Cực nhanh** (10–20× XGBoost) | Chấm điểm <50ms cho 1 giao dịch realtime |
| **Xử lý mất cân bằng** (`scale_pos_weight`) | Fraud chỉ ~1% giao dịch — model vẫn học được |
| **Tự xử lý feature thiếu** | Không cần điền NaN |
| **TreeExplainer SHAP** rất nhanh | Giải thích từng giao dịch trong vài ms |
| **Robust** với outlier | Không bị giá trị bất thường làm lệch |

**Cách phát hiện Fraud:** model học từ dữ liệu lịch sử các giao dịch *thường* và
*gian lận*, sau đó với mỗi giao dịch mới sẽ tổng hợp ~15 dấu hiệu (số tiền,
tần suất, người nhận mới, thiết bị mới, giờ ban đêm…) để đưa ra điểm rủi ro.
        """)

    st.markdown("#### 📤 Nạp dữ liệu huấn luyện")
    st.caption(
        f"File CSV phải có cột **`{TARGET_COL}`** (0/1). "
        f"Các cột số khác đều được tự động dùng làm feature. "
        f"Bộ feature gốc gợi ý: `{', '.join(DEFAULT_FRAUD_FEATURES)}`"
    )

    up = st.file_uploader("Tải lên file CSV (dữ liệu giao dịch thật)",
                          type=["csv"], key="t1_csv")

    df = None
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.success(f"✅ Đã đọc {len(df):,} dòng từ `{up.name}`.")
        except Exception as e:
            st.error(f"❌ Không đọc được CSV: {e}")
    else:
        st.info("ℹ️ Vui lòng tải file CSV để bắt đầu huấn luyện.")

    if df is not None:
        st.markdown("#### 👀 Preview 6 dòng đầu")
        st.dataframe(df.head(6), use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng giao dịch", f"{len(df):,}")
        if TARGET_COL in df.columns:
            n_fraud = int((df[TARGET_COL] == 1).sum())
            c2.metric("Số ca Fraud", f"{n_fraud:,}", f"{n_fraud/len(df)*100:.2f}%")
            c3.metric("Số ca hợp lệ", f"{len(df)-n_fraud:,}")
        c4.metric("Số feature", f"{len([c for c in df.columns if c != TARGET_COL]):,}")

        st.markdown("#### ⚙️ Tham số huấn luyện")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            n_est = st.slider("n_estimators", 100, 800, 300, step=50)
        with cc2:
            lr = st.select_slider("learning_rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
        with cc3:
            n_leaves = st.slider("num_leaves", 15, 127, 31)
        with cc4:
            test_size = st.slider("Test split %", 10, 40, 20)

        if st.button("🚀 Huấn luyện LightGBM", type="primary", key="t1_train"):
            try:
                from sklearn.model_selection import train_test_split
            except ImportError:
                st.error("❌ Thiếu scikit-learn.")
                st.stop()

            # ── Hiệu ứng status box: hiển thị từng bước cho giám khảo
            with st.status("⏳ Đang huấn luyện LightGBM…", expanded=True) as status:
                st.write("📂 Bước 1/5 — Tách features và target…")
                X, y, feat_cols = split_features(df, TARGET_COL)

                st.write(f"✂️ Bước 2/5 — Chia train / valid / test ({100-test_size}/{test_size}%)…")
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
                )
                st.write(f"   • Train: {len(X_tr):,}  ·  Valid: {len(X_va):,}  ·  Test: {len(X_te):,}")

                st.write(f"🏋️ Bước 3/5 — Train với {n_est} cây, lr={lr}, leaves={n_leaves}…")
                model, history = train_lightgbm(
                    X_tr, y_tr, X_va, y_va,
                    n_estimators=n_est, learning_rate=float(lr), num_leaves=n_leaves,
                )
                st.write(f"   • Best iteration = **{model.best_iteration_}**")

                st.write("🎯 Bước 4/5 — Tìm ngưỡng tối ưu trên tập validation (F1-max)…")
                opt_thr = find_optimal_threshold(model, X_va, y_va)
                st.write(f"   • Ngưỡng tối ưu = **{opt_thr:.4f}**  "
                         f"(thay cho 0.5 mặc định để tránh P/R/F1 = 0%)")

                st.write("📊 Bước 5/5 — Đánh giá trên tập test…")
                metrics = evaluate(model, X_te, y_te, threshold=opt_thr)

                st.session_state["t1_model"] = model
                st.session_state["t1_features"] = feat_cols
                st.session_state["t1_X_te"] = X_te
                st.session_state["t1_y_te"] = y_te
                st.session_state["t1_metrics"] = metrics
                st.session_state["t1_history"] = history
                st.session_state["t1_opt_thr"] = opt_thr

                status.update(label=f"✅ Huấn luyện hoàn tất — F1 = {metrics['f1']*100:.2f}%",
                              state="complete", expanded=False)
            st.balloons()

    # ── Hiển thị kết quả nếu đã train ─────────────────────────────
    if "t1_metrics" in st.session_state:
        model = st.session_state["t1_model"]
        feat_cols = st.session_state["t1_features"]
        X_te = st.session_state["t1_X_te"]
        y_te = st.session_state["t1_y_te"]
        opt_thr = st.session_state.get("t1_opt_thr", 0.5)

        st.markdown("#### 📈 Kết quả đánh giá trên tập test")
        st.caption(
            f"🎯 Ngưỡng quyết định **được tối ưu tự động trên validation = {opt_thr:.4f}** "
            "(thay vì 0.5 mặc định). Bạn có thể điều chỉnh dưới đây để xem ảnh hưởng."
        )
        thr_used = st.slider(
            "Ngưỡng phân loại (threshold)", 0.05, 0.95, float(opt_thr),
            step=0.01, key="t1_thr_slider",
            help="Kéo để xem trade-off Precision ↔ Recall.",
        )
        # Re-evaluate ở threshold người dùng chọn
        m = evaluate(model, X_te, y_te, threshold=float(thr_used))
        st.session_state["t1_metrics"] = m

        mc = st.columns(5)
        mc[0].metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
        mc[1].metric("Precision", f"{m['precision']*100:.2f}%")
        mc[2].metric("Recall",    f"{m['recall']*100:.2f}%")
        mc[3].metric("F1-Score",  f"{m['f1']*100:.2f}%")
        mc[4].metric("AUC-ROC",   f"{m['auc_roc']:.4f}")

        st.markdown("#### 📊 Lưới biểu đồ đánh giá (2×2)")

        # ── Confusion matrix
        import seaborn as sns
        row1 = st.columns(2)
        with row1[0]:
            st.markdown("**Confusion Matrix**")
            fig_cm, ax = plt.subplots(figsize=(4.5, 3.6))
            sns.heatmap(
                m["cm"], annot=True, fmt="d",
                cmap=sns.light_palette(PALETTE["rose_dark"], as_cmap=True),
                ax=ax, cbar=False,
                xticklabels=["Hợp lệ", "Fraud"],
                yticklabels=["Hợp lệ", "Fraud"],
            )
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm); plt.close()

        with row1[1]:
            st.markdown("**ROC Curve**")
            fpr, tpr, _ = m["fpr_tpr"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=PALETTE["rose_dark"], width=3),
                fill="tozeroy", fillcolor="rgba(228,139,167,0.18)",
                name=f"AUC = {m['auc_roc']:.3f}",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color=PALETTE["ink_soft"], dash="dash", width=1.5),
                name="Random", showlegend=False,
            ))
            fig.update_layout(xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            plotly_pastel_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)

        row2 = st.columns(2)
        with row2[0]:
            st.markdown("**Precision-Recall Curve**")
            precs, recs, _ = m["pr_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recs, y=precs, mode="lines",
                line=dict(color=PALETTE["lavender_dark"], width=3),
                fill="tozeroy", fillcolor="rgba(169,147,224,0.18)",
                name=f"AP = {m['auc_pr']:.3f}",
            ))
            fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
            plotly_pastel_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with row2[1]:
            st.markdown("**Top 15 Feature Importance**")
            importances = pd.DataFrame({
                "feature": feat_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=True).tail(15)
            importances["label"] = importances["feature"].map(feature_label)
            fig = go.Figure(go.Bar(
                x=importances["importance"], y=importances["label"],
                orientation="h",
                marker=dict(
                    color=importances["importance"],
                    colorscale=[[0, PALETTE["mint"]], [.5, PALETTE["rose"]],
                                [1, PALETTE["lavender_dark"]]],
                ),
                text=importances["importance"], textposition="outside",
            ))
            fig.update_layout(xaxis_title="Importance", yaxis_title="")
            plotly_pastel_layout(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)

        # ── Bonus: Score distribution ─────────────────────────────
        st.markdown("#### 🎚️ Phân bố điểm rủi ro trên tập test")
        proba = m["proba"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=proba[y_te == 0], name="Hợp lệ", nbinsx=40,
            marker_color=PALETTE["mint_dark"], opacity=0.7,
        ))
        fig.add_trace(go.Histogram(
            x=proba[y_te == 1], name="Fraud", nbinsx=40,
            marker_color=PALETTE["rose_dark"], opacity=0.7,
        ))
        fig.add_vline(x=0.5, line=dict(color=PALETTE["ink"], dash="dash"),
                      annotation_text="Ngưỡng 0.5")
        fig.update_layout(barmode="overlay", xaxis_title="Probability fraud",
                          yaxis_title="Số giao dịch")
        plotly_pastel_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — DEMO REALTIME (upload Profile khách hàng + nhập giao dịch mới)
# ════════════════════════════════════════════════════════════════════
PROFILE_HIDE_COLS = ["user_id", "name_user", "job", "main_device", "age", "persona"]


with tab_demo:
    if "t1_model" not in st.session_state:
        st.warning("⚠️ Vui lòng huấn luyện model ở tab **Huấn luyện model** trước.")
        st.stop()

    model = st.session_state["t1_model"]
    feat_cols = st.session_state["t1_features"]

    st.markdown("#### 🏦 Mô phỏng một giao dịch chuyển tiền")

    # ───────────────────────────────────────────────────────────────
    #  PHẦN 1 — Upload Profile khách hàng (CSV 80 ngày giao dịch)
    # ───────────────────────────────────────────────────────────────
    profile_file = st.file_uploader(
        "📤 Tải lên CSV Profile khách hàng (chứa thông tin cá nhân + lịch sử giao dịch 80 ngày)",
        type=["csv"], key="t1_profile_csv",
    )

    if profile_file is None:
        st.info("ℹ️ Vui lòng tải file CSV Profile khách hàng để bắt đầu demo realtime.")
        st.stop()

    try:
        profile_df = pd.read_csv(profile_file)
    except Exception as e:
        st.error(f"❌ Không đọc được CSV: {e}")
        st.stop()

    if len(profile_df) == 0:
        st.error("❌ File CSV rỗng.")
        st.stop()

    st.session_state["t1_profile_df"] = profile_df

    # ── Trích thông tin Profile từ dòng đầu ──────────────────────
    first = profile_df.iloc[0]
    user_id      = str(first.get("user_id", "—"))
    name_user    = str(first.get("name_user", "—"))
    age          = first.get("age", "—")
    job          = str(first.get("job", "—"))
    main_device  = str(first.get("main_device", "—"))
    balance_before_first = float(first.get("balance_before", 0) or 0)

    # balance_after lấy từ dòng cuối có dữ liệu
    if "balance_after" in profile_df.columns:
        ba_series = pd.to_numeric(profile_df["balance_after"], errors="coerce").dropna()
        balance_after_last = float(ba_series.iloc[-1]) if len(ba_series) else 0.0
    else:
        balance_after_last = 0.0

    # ── Hiển thị Profile (4 + 2 cards) ───────────────────────────
    st.markdown("##### 👤 Hồ sơ khách hàng")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(
        f"<div class='pastel-card' style='background:#FFEAF1'>"
        f"<small style='color:#8C7785'>USER ID</small>"
        f"<div style='font-size:18px;font-weight:700;color:{PALETTE['rose_dark']}'>{user_id}</div>"
        f"<small style='color:#8C7785'>{name_user}</small></div>",
        unsafe_allow_html=True,
    )
    p2.markdown(
        f"<div class='pastel-card' style='background:#F5EFFF'>"
        f"<small style='color:#8C7785'>TUỔI · NGHỀ NGHIỆP</small>"
        f"<div style='font-size:18px;font-weight:700;color:{PALETTE['lavender_dark']}'>{age}</div>"
        f"<small style='color:#8C7785'>{job}</small></div>",
        unsafe_allow_html=True,
    )
    p3.markdown(
        f"<div class='pastel-card' style='background:#E6F7EF'>"
        f"<small style='color:#8C7785'>THIẾT BỊ CHÍNH</small>"
        f"<div style='font-size:18px;font-weight:700;color:{PALETTE['mint_dark']}'>📱</div>"
        f"<small style='color:#8C7785'>{main_device}</small></div>",
        unsafe_allow_html=True,
    )
    p4.markdown(
        f"<div class='pastel-card' style='background:#FFF7E6'>"
        f"<small style='color:#8C7785'>SỐ DỄ HIỆN TẠI</small>"
        f"<div style='font-size:18px;font-weight:700;color:{PALETTE['peach_dark']}'>"
        f"{balance_after_last:,.0f} ₫</div>"
        f"<small style='color:#8C7785'>(số dư đầu kỳ: {balance_before_first:,.0f} ₫)</small></div>",
        unsafe_allow_html=True,
    )

    # ── Bảng lịch sử giao dịch ───────────────────────────────────
    st.markdown("##### 📋 Lịch sử giao dịch 80 ngày")
    show_cols = [c for c in profile_df.columns if c not in PROFILE_HIDE_COLS]
    st.dataframe(
        profile_df[show_cols],
        use_container_width=True, hide_index=True, height=280,
    )

    st.divider()

    # ───────────────────────────────────────────────────────────────
    #  PHẦN 2 — Form nhập giao dịch mới
    # ───────────────────────────────────────────────────────────────
    st.markdown("##### 💸 Tạo giao dịch chuyển tiền mới")

    # Thiết bị mặc định = main_device
    dev_col = find_col(profile_df, ["device_id", "device", "device_used"])
    seen_devices = sorted(set(profile_df[dev_col].astype(str))) if dev_col else []
    device_options = list(dict.fromkeys([main_device] + seen_devices + ["📱 Thiết bị MỚI lạ"]))

    # Lấy ngày của giao dịch cuối làm gợi ý
    try:
        h_sorted, _ts_col = parse_history_timestamps(profile_df)
        last_ts = h_sorted[_ts_col].iloc[-1]
        suggest_date = (last_ts + timedelta(days=1)).date()
    except Exception:
        suggest_date = datetime.now().date()
        last_ts = None

    with st.form("t1_new_txn_form", clear_on_submit=False):
        f1, f2 = st.columns(2)
        with f1:
            txn_date = st.date_input("📅 Ngày giao dịch", value=suggest_date)
            t_h, t_m = st.columns(2)
            txn_hour = t_h.selectbox("🕒 Giờ", options=list(range(24)),
                                     index=14, format_func=lambda x: f"{x:02d}h")
            txn_min  = t_m.selectbox("Phút", options=list(range(0, 60, 5)),
                                     index=6, format_func=lambda x: f"{x:02d}")
            src_account = st.text_input("👤 Tài khoản nguồn", value=name_user, disabled=True)
            new_device = st.selectbox("📱 Thiết bị thực hiện", options=device_options,
                                       help="Chọn 'Thiết bị MỚI lạ' để mô phỏng kẻ gian dùng máy khác")
        with f2:
            bank = st.selectbox("🏦 Ngân hàng nhận", options=VN_BANKS)
            recipient_name = st.text_input("🎯 Tên người nhận", placeholder="VD: NGUYEN VAN A")
            recipient_acc  = st.text_input("🔢 Số tài khoản nhận", placeholder="VD: 1029384756")
            amount = st.number_input(
                "💰 Số tiền chuyển (VND)", min_value=0, step=100_000,
                value=500_000, format="%d",
            )

        submitted = st.form_submit_button("💸 Chuyển tiền", type="primary",
                                          use_container_width=True)

    if submitted:
        if not recipient_name.strip() or not recipient_acc.strip() or amount <= 0:
            st.error("⚠️ Vui lòng điền đầy đủ tên người nhận, số tài khoản và số tiền > 0.")
        else:
            new_ts = pd.Timestamp.combine(txn_date, time(int(txn_hour), int(txn_min)))
            try:
                feats = compute_realtime_features(
                    history_df=profile_df,
                    new_ts=new_ts,
                    new_amount=float(amount),
                    new_recipient=recipient_acc,
                    new_device=("__NEW__" if new_device.startswith("📱") else new_device),
                    balance_before=balance_after_last,
                )
                st.session_state["t1_realtime_features"] = feats
                st.session_state["t1_realtime_meta"] = {
                    "timestamp": new_ts, "src_account": name_user,
                    "bank": bank, "recipient_name": recipient_name,
                    "recipient_acc": recipient_acc, "amount": float(amount),
                    "device": new_device, "balance_before": balance_after_last,
                }
                # Reset prediction cũ khi tạo giao dịch mới
                st.session_state.pop("t1_realtime_proba", None)
            except Exception as e:
                st.error(f"❌ Không tính được feature: {e}")

    # ───────────────────────────────────────────────────────────────
    #  PHẦN 3 — Bảng feature đã tính + nút dự báo
    # ───────────────────────────────────────────────────────────────
    if "t1_realtime_features" in st.session_state:
        feats = st.session_state["t1_realtime_features"]
        meta  = st.session_state["t1_realtime_meta"]

        st.divider()
        st.markdown("##### 🧮 Bảng đặc trưng đã tính cho giao dịch realtime")
        st.caption(f"Giao dịch: **{meta['amount']:,.0f} VNĐ** → "
                   f"{meta['recipient_name']} ({meta['bank']}) — "
                   f"{meta['timestamp'].strftime('%d/%m/%Y %H:%M')}")

        feat_show = pd.DataFrame([
            {"Đặc trưng": feature_label(k),
             "Giá trị":  f"{v:,.4f}" if isinstance(v, float) else f"{v:,}"}
            for k, v in feats.items()
        ])
        st.dataframe(feat_show, hide_index=True, use_container_width=True, height=400)

        if st.button("🤖 Dự báo Fraud", type="primary", key="t1_predict_btn"):
            x_row = pd.DataFrame([[feats.get(c, 0) for c in feat_cols]],
                                 columns=feat_cols)
            with st.spinner("Đang chấm điểm bằng LightGBM…"):
                proba = float(model.predict_proba(x_row)[0, 1])
            st.session_state["t1_realtime_proba"] = proba
            st.session_state["t1_realtime_x_row"] = x_row

    # ───────────────────────────────────────────────────────────────
    #  PHẦN 4 — Kết quả AI + SHAP + khuyến nghị + chat (REALTIME)
    # ───────────────────────────────────────────────────────────────
    if "t1_realtime_proba" in st.session_state:
        proba = st.session_state["t1_realtime_proba"]
        x_row = st.session_state["t1_realtime_x_row"]
        feats = st.session_state["t1_realtime_features"]
        meta  = st.session_state["t1_realtime_meta"]

        st.divider()
        st.markdown("##### 🤖 Kết quả AI")

        gauge_color = (PALETTE["mint_dark"] if proba < 0.30 else
                       PALETTE["peach_dark"] if proba < 0.65 else
                       PALETTE["rose_dark"])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 42, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PALETTE["ink_soft"]},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "#FFFAFC",
                "borderwidth": 2, "bordercolor": "#F4D7E1",
                "steps": [
                    {"range": [0, 30],  "color": "#E7F7EF"},
                    {"range": [30, 65], "color": "#FFF4DE"},
                    {"range": [65, 100],"color": "#FCE4EC"},
                ],
                "threshold": {
                    "line": {"color": PALETTE["ink"], "width": 3},
                    "thickness": 0.75, "value": proba * 100,
                },
            },
            title={"text": "Xác suất Fraud", "font": {"size": 14}},
        ))
        plotly_pastel_layout(gauge, height=260)

        gc1, gc2 = st.columns([1, 1])
        with gc1:
            st.plotly_chart(gauge, use_container_width=True)
        with gc2:
            st.markdown(f"<br>", unsafe_allow_html=True)
            if proba < 0.30:
                st.markdown("<div class='badge badge-ok'>✅ APPROVED — Cho qua tự động</div>",
                            unsafe_allow_html=True)
            elif proba < 0.65:
                st.markdown("<div class='badge badge-warn'>⚠️ REVIEW — Yêu cầu xác thực bổ sung</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge badge-block'>🚫 BLOCKED — Tạm khoá giao dịch</div>",
                            unsafe_allow_html=True)
            st.metric("Số tiền giao dịch", f"{meta['amount']:,.0f} ₫")
            st.metric("Người nhận", meta["recipient_name"])
            st.metric("Ngân hàng nhận", meta["bank"])

        # ── SHAP waterfall ───────────────────────────────────────
        st.markdown("##### 🔍 SHAP — Vì sao AI quyết định như vậy?")
        st.caption("Đỏ = đẩy về phía Fraud · Xanh = kéo về phía Hợp lệ")
        try:
            import shap
            explainer, shap_vals, base_val = get_shap_values(model, x_row)
            sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
            contribs = pd.DataFrame({
                "feature": [feature_label(c) for c in feat_cols],
                "value":   x_row.iloc[0].values,
                "shap":    sv,
            })
            contribs["abs"] = contribs["shap"].abs()
            contribs = contribs.sort_values("abs", ascending=False).head(10).iloc[::-1]
            bar_colors = [PALETTE["rose_dark"] if v > 0 else PALETTE["mint_dark"]
                          for v in contribs["shap"]]
            labels = [f"{f} = {v:.3g}" for f, v in zip(contribs["feature"], contribs["value"])]
            fig_w = go.Figure(go.Bar(
                x=contribs["shap"], y=labels, orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in contribs["shap"]], textposition="outside",
            ))
            fig_w.add_vline(x=0, line=dict(color=PALETTE["ink"], width=1))
            fig_w.update_layout(
                xaxis_title="Mức đóng góp tới điểm Fraud (SHAP value)", yaxis_title="",
                title=f"Base = {base_val:+.3f} → Final logit ≈ {base_val + sv.sum():+.3f}",
            )
            plotly_pastel_layout(fig_w, height=420)
            st.plotly_chart(fig_w, use_container_width=True)
            top_factors = list(zip(contribs["feature"][::-1], contribs["shap"][::-1]))
        except ImportError:
            st.error("❌ Thiếu thư viện `shap`. Chạy: `pip install shap`")
            top_factors = []
        except Exception as e:
            st.error(f"❌ Lỗi tính SHAP: {e}")
            top_factors = []

        # ── Khuyến nghị ──────────────────────────────────────────
        st.markdown("##### 📋 Khuyến nghị hành động")
        st.markdown(recommend_action(proba, top_factors))

        # ── Khung chat Gemini (lấy data realtime) ────────────────
        st.divider()
        chat_context = {
            "Khách hàng":              f"{name_user} ({user_id}, {age} tuổi, {job})",
            "Số tiền giao dịch (VNĐ)": f"{meta['amount']:,.0f}",
            "Người nhận":              f"{meta['recipient_name']} - STK {meta['recipient_acc']}",
            "Ngân hàng nhận":          meta["bank"],
            "Thời gian":               meta["timestamp"].strftime("%d/%m/%Y %H:%M"),
            "Thiết bị":                meta["device"],
            "Số dư trước GD":          f"{meta['balance_before']:,.0f} VNĐ",
            "Điểm rủi ro AI (0–1)":    f"{proba:.4f}",
            "Xác suất Fraud (%)":      f"{proba*100:.2f}",
            "Quyết định hệ thống":     ("APPROVED" if proba < 0.3 else
                                         "REVIEW" if proba < 0.65 else "BLOCKED"),
            "Top yếu tố ảnh hưởng":    [f"{f}: {v:+.3f}" for f, v in top_factors[:6]],
            "Đặc trưng giao dịch":     [f"{feature_label(k)}: {v}" for k, v in feats.items()],
        }
        render_chat_panel(
            context=chat_context,
            role="Phát hiện gian lận giao dịch chuyển tiền realtime (Tầng 1 — LightGBM)",
            key_prefix="t1_chat",
            suggested_questions=[
                "Vì sao giao dịch này bị đánh dấu rủi ro?",
                ("Giải thích dễ hiểu cho khách hàng",
                 "Tôi nên giải thích thế nào với khách hàng?"),
                "Khách hàng cần làm gì tiếp theo?",
            ],
            title="💬 Phân tích của chuyên gia AI — Giải thích cho khách hàng",
        )
