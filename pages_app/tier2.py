"""🧠 Tầng 2 — Sinh trắc học hành vi (Siamese Network + MLP)."""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from theme import inject_pastel_css, matplotlib_pastel, plotly_pastel_layout, PALETTE
from lib_personas import PERSONAS, FEATURE_COLS, FEATURE_VI, generate_sessions, generate_pairs
from lib_siamese import build_siamese_model
from lib_gemini import render_chat_panel

inject_pastel_css()
matplotlib_pastel()

st.markdown("<div class='hero-tag'>🧠 TẦNG 2 · SIAMESE NETWORK</div>", unsafe_allow_html=True)
st.title("Sinh trắc học hành vi")
st.caption(
    "Câu hỏi cốt lõi: *User đang thao tác có giống «chính họ» trong quá khứ không?* — "
    "ứng dụng cảm biến điện thoại (áp lực ngón, gyro, touch area) để phát hiện kẻ mạo danh "
    "dù biết mật khẩu/OTP."
)

tab_data, tab_train, tab_demo = st.tabs([
    "📊 Data Mô phỏng Hành vi",
    "🧬 Siamese Network — training Model MLP",
    "🎯 Demo Live hành vi",
])

# ════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA MÔ PHỎNG
# ════════════════════════════════════════════════════════════════════
with tab_data:
    st.info(
        "📌 **Vì sao phải dùng dữ liệu mô phỏng?**  \n"
        "Tại Việt Nam **chưa có ngân hàng nào triển khai sinh trắc học hành vi** "
        "ở quy mô đủ lớn để công bố dataset, đồng thời dữ liệu cảm biến là "
        "**dữ liệu cá nhân nhạy cảm** không thể chia sẻ. Chúng tôi xây bộ giả lập "
        "20 persona (mỗi người 1 «chữ ký gõ phím» riêng) **bám sát thực tế đo "
        "đạc trên thiết bị Android/iOS** — đủ để chứng minh tính khả thi của "
        "Siamese Network."
    )

    st.markdown("#### 📊 Tạo Bộ Dữ Liệu Giả Lập")
    st.caption("Sinh dữ liệu hành vi cho **20 user** với **12 features** mỗi người. Mỗi user có nhiều session với noise thực tế.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_sessions = st.slider("Số session mỗi user", 20, 50, 25, key="t2_n_sess")
    with c2:
        noise_seed = st.number_input("Random seed", 0, 9999, 42, key="t2_seed")
    with c3:
        show_persona = st.checkbox("Hiện bảng persona", value=True, key="t2_show")

    if show_persona:
        persona_df = pd.DataFrame([
            {"User ID": p["id"], "Tên": p["name"], "Đặc điểm": p["desc"]}
            for p in PERSONAS
        ])
        st.dataframe(persona_df, use_container_width=True, hide_index=True)

    if st.button("🚀 Tạo Dữ Liệu", type="primary", key="t2_gen"):
        with st.spinner("Đang tạo dữ liệu..."):
            dfs = [generate_sessions(p, n_sessions, seed=int(noise_seed)) for p in PERSONAS]
            full_df = pd.concat(dfs, ignore_index=True)
            st.session_state["raw_df"] = full_df
        st.success(f"✅ Đã tạo **{len(full_df)} sessions** cho 20 users!")

    if "raw_df" in st.session_state:
        full_df = st.session_state["raw_df"]
        st.markdown("##### Xem trước dữ liệu")
        st.dataframe(full_df.head(30), use_container_width=True, hide_index=True)

        ca, cb, cc = st.columns(3)
        ca.metric("Tổng sessions", len(full_df))
        cb.metric("Số users", full_df["user_id"].nunique())
        cc.metric("Số features", len(FEATURE_COLS))

        st.markdown("##### Phân bố `avg_pressure` theo từng user")
        fig, ax = plt.subplots(figsize=(13, 4.2))
        from theme import PALETTE_LIST
        for i, p in enumerate(PERSONAS):
            uid_df = full_df[full_df["user_id"] == p["id"]]
            ax.scatter(
                [p["id"]] * len(uid_df), uid_df["avg_pressure"],
                alpha=0.55, s=22, color=PALETTE_LIST[i % len(PALETTE_LIST)], label=p["id"]
            )
        ax.set_xlabel("User ID"); ax.set_ylabel("avg_pressure")
        ax.set_title("Mỗi chấm = 1 session — mỗi user có «chữ ký» riêng")
        plt.xticks(rotation=45, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        csv_raw = full_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Tải xuống Raw Data CSV", csv_raw,
                           file_name="ato_behavioral_data.csv", mime="text/csv")

    st.divider()
    st.markdown("#### 🔗 Tạo Bộ Dữ Liệu Cặp (Pair Dataset)")
    st.caption("Ghép các session thành cặp để train Siamese Network — cùng người (label=1) hoặc khác người (label=0).")

    cp1, cp2 = st.columns(2)
    n_same = cp1.slider("Số cặp cùng người", 200, 1000, 500, step=50, key="t2_same")
    n_diff = cp2.slider("Số cặp khác người", 200, 1000, 500, step=50, key="t2_diff")

    if st.button("🔗 Tạo Pair Dataset", key="t2_pairs"):
        if "raw_df" not in st.session_state:
            st.warning("⚠️ Hãy tạo Raw Data trước.")
        else:
            with st.spinner("Đang tạo cặp..."):
                pair_df = generate_pairs(st.session_state["raw_df"], n_same, n_diff)
                st.session_state["pair_df"] = pair_df
            st.success(f"✅ Đã tạo **{len(pair_df)} cặp**.")

    if "pair_df" in st.session_state:
        pair_df = st.session_state["pair_df"]
        st.dataframe(pair_df.head(20), use_container_width=True, hide_index=True)
        cd1, cd2 = st.columns(2)
        cd1.metric("Cặp cùng người", int((pair_df["label"] == 1).sum()))
        cd2.metric("Cặp khác người", int((pair_df["label"] == 0).sum()))


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — SIAMESE TRAINING
# ════════════════════════════════════════════════════════════════════
with tab_train:
    with st.expander("📘 Siamese Network + MLP — kiến trúc & nguyên lý", expanded=False):
        st.markdown("""
**Siamese Network** (Mạng song sinh) là kiến trúc **2 nhánh chia sẻ chung trọng số** —
nó học cách *so sánh* hai đầu vào thay vì phân loại.

```
Session A (12 features) ─┐
                          ├─► MLP shared (64→32→16) ─► Embedding A ─┐
Session B (12 features) ─┘                                            ├─► Khoảng cách Euclid ─► Sigmoid ─► Score
                                                  ─► Embedding B ─────┘
```

**MLP** (Multi-Layer Perceptron) backbone **3 lớp** chuyển 12 đặc trưng cảm biến
thô thành **vector embedding 16 chiều** đại diện cho «chữ ký hành vi».

**Vì sao Siamese phù hợp với behavioral biometrics?**
- ✅ Ngân hàng có **hàng triệu user** — không thể train mỗi user 1 model
- ✅ User mới đăng ký chỉ cần **vài session** là dùng được (one-shot learning)
- ✅ Bảo mật: không lưu mật khẩu sinh học gốc, chỉ lưu **embedding 16 chiều**
        """)

    if "pair_df" not in st.session_state:
        st.info("ℹ️ Vui lòng tạo Pair Dataset ở tab **Data Mô phỏng** trước.")
    else:
        pair_df = st.session_state["pair_df"]

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            epochs = st.slider("Số epochs", 5, 100, 30, key="t2_epochs")
        with col_t2:
            batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1, key="t2_bs")
        with col_t3:
            test_split = st.slider("Test split %", 10, 40, 20, key="t2_split")

        if st.button("🏋️ Train Siamese Network", type="primary", key="t2_train_btn"):
            import tensorflow as tf
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            with st.spinner("Đang xây dựng và train mô hình..."):
                feat_A = [f"A_{f}" for f in FEATURE_COLS]
                feat_B = [f"B_{f}" for f in FEATURE_COLS]
                X_A = pair_df[feat_A].values.astype(np.float32)
                X_B = pair_df[feat_B].values.astype(np.float32)
                y = pair_df["label"].values.astype(np.float32)

                scaler = StandardScaler()
                scaler.fit(np.vstack([X_A, X_B]))
                X_A_s = scaler.transform(X_A)
                X_B_s = scaler.transform(X_B)

                XA_tr, XA_te, XB_tr, XB_te, y_tr, y_te = train_test_split(
                    X_A_s, X_B_s, y, test_size=test_split/100, random_state=42, stratify=y
                )

                model, mlp_backbone = build_siamese_model()
                st.session_state.update({
                    "scaler": scaler, "mlp_backbone": mlp_backbone, "model": model,
                })

                hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
                pbar = st.progress(0); status = st.empty()

                class SCB(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        for k in hist: hist[k].append(logs.get(k, 0))
                        pbar.progress(int((epoch + 1) / epochs * 100))
                        status.text(
                            f"Epoch {epoch+1}/{epochs} | loss={logs.get('loss',0):.4f} | "
                            f"val_acc={logs.get('val_accuracy',0):.4f}"
                        )

                model.fit(
                    [XA_tr, XB_tr], y_tr,
                    validation_data=([XA_te, XB_te], y_te),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[SCB()], verbose=0,
                )

                st.session_state.update({
                    "XA_te": XA_te, "XB_te": XB_te, "y_te": y_te, "history": hist,
                })

            st.success("✅ Huấn luyện hoàn tất!")

        # ── Kết quả ────────────────────────────────────────────
        if "history" in st.session_state:
            from sklearn.metrics import confusion_matrix, roc_auc_score
            model = st.session_state["model"]
            XA_te = st.session_state["XA_te"]; XB_te = st.session_state["XB_te"]
            y_te = st.session_state["y_te"]
            h = st.session_state["history"]

            st.markdown("##### 📈 Đường Cong Học Tập")
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
            a1.plot(h["loss"], color=PALETTE["rose_dark"], label="Train")
            a1.plot(h["val_loss"], color=PALETTE["lavender_dark"], linestyle="--", label="Val")
            a1.set_title("Loss"); a1.set_xlabel("Epoch"); a1.legend()
            a2.plot(h["accuracy"], color=PALETTE["mint_dark"], label="Train")
            a2.plot(h["val_accuracy"], color=PALETTE["peach_dark"], linestyle="--", label="Val")
            a2.set_title("Accuracy"); a2.set_xlabel("Epoch"); a2.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

            preds = model.predict([XA_te, XB_te], verbose=0).flatten()
            pred_lbls = (preds >= 0.5).astype(int)
            auc = roc_auc_score(y_te, preds)

            mc = st.columns(3)
            mc[0].metric("AUC-ROC", f"{auc:.4f}")
            mc[1].metric("Accuracy", f"{np.mean(pred_lbls == y_te):.4f}")
            mc[2].metric("Test samples", len(y_te))

            cf1, cf2 = st.columns(2)
            with cf1:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_te, pred_lbls)
                fig_cm, ax = plt.subplots(figsize=(4.5, 3.6))
                sns.heatmap(cm, annot=True, fmt="d",
                            cmap=sns.light_palette(PALETTE["lavender_dark"], as_cmap=True),
                            ax=ax, cbar=False,
                            xticklabels=["Khác", "Cùng"], yticklabels=["Khác", "Cùng"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                plt.tight_layout(); st.pyplot(fig_cm); plt.close()
            with cf2:
                st.markdown("**Phân bố Score dự đoán**")
                fig_s, ax = plt.subplots(figsize=(5.5, 3.6))
                ax.hist(preds[y_te == 1], bins=30, alpha=0.7, label="Cùng người",
                        color=PALETTE["mint_dark"])
                ax.hist(preds[y_te == 0], bins=30, alpha=0.7, label="Khác người",
                        color=PALETTE["rose_dark"])
                ax.axvline(0.5, color=PALETTE["ink"], linestyle="--")
                ax.set_xlabel("Score"); ax.legend()
                plt.tight_layout(); st.pyplot(fig_s); plt.close()


# ════════════════════════════════════════════════════════════════════
#  TAB 3 — DEMO LIVE
# ════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("#### 🎯 Demo Live — Kiểm tra một session mới")
    st.caption(
        "Chọn user gốc (reference) và kịch bản test. Mô hình sẽ chấm điểm tương tự "
        "giữa session mới và «chữ ký gõ phím» của user gốc."
    )

    if "model" not in st.session_state:
        st.info("ℹ️ Vui lòng train mô hình ở tab **Siamese Network — training Model MLP**.")
        st.stop()
    if "raw_df" not in st.session_state:
        st.warning("⚠️ Cần có Raw Data từ tab **Data Mô phỏng**.")
        st.stop()

    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    raw_df = st.session_state["raw_df"]

    cd1, cd2 = st.columns(2)
    with cd1:
        ref_user_id = st.selectbox(
            "Chọn User Gốc (Reference)",
            options=[p["id"] for p in PERSONAS],
            format_func=lambda x: f"{x} — {next(p['name'] for p in PERSONAS if p['id']==x)}",
            key="t2_ref",
        )
    with cd2:
        scenario = st.radio(
            "Kịch bản test", ["✅ Cùng người (nhiễu nhẹ)", "❌ Kẻ mạo danh (user khác)"],
            horizontal=True, key="t2_scen",
        )

    if st.button("🔍 Kiểm tra ngay", key="t2_check", type="primary"):
        ref_sessions = raw_df[raw_df["user_id"] == ref_user_id]
        ref_vec = ref_sessions[FEATURE_COLS].mean().values.astype(np.float32)

        if "Cùng người" in scenario:
            rng = np.random.default_rng(77)
            new_vec = (ref_vec + rng.normal(0, np.abs(ref_vec) * 0.07)).astype(np.float32)
            true_label = "Cùng người"; imposter_name = None
        else:
            others = [p for p in PERSONAS if p["id"] != ref_user_id]
            imp = np.random.choice(others)
            new_vec = raw_df[raw_df["user_id"] == imp["id"]][FEATURE_COLS].mean().values.astype(np.float32)
            true_label = f"Kẻ mạo danh ({imp['name']})"; imposter_name = imp['name']

        ref_s = scaler.transform(ref_vec.reshape(1, -1))
        new_s = scaler.transform(new_vec.reshape(1, -1))
        score = float(model.predict([ref_s, new_s], verbose=0).flatten()[0])

        mlp_b = st.session_state["mlp_backbone"]
        emb_ref = mlp_b.predict(ref_s, verbose=0).flatten()
        emb_new = mlp_b.predict(new_s, verbose=0).flatten()
        dist = float(np.sqrt(((emb_ref - emb_new) ** 2).sum()))

        st.session_state["t2_demo_result"] = dict(
            ref_user_id=ref_user_id, ref_vec=ref_vec, new_vec=new_vec,
            score=score, dist=dist, emb_ref=emb_ref, emb_new=emb_new,
            true_label=true_label, scenario=scenario, imposter=imposter_name,
        )

    if "t2_demo_result" in st.session_state:
        r = st.session_state["t2_demo_result"]

        st.markdown("##### 🤖 Kết quả AI")
        gauge_color = (PALETTE["mint_dark"] if r["score"] >= 0.65 else
                       PALETTE["peach_dark"] if r["score"] >= 0.35 else
                       PALETTE["rose_dark"])
        # Score đối với Siamese: gần 1 = cùng người = SAFE
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r["score"] * 100,
            number={"suffix": "%", "font": {"size": 38, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 35],  "color": "#FCE4EC"},
                    {"range": [35, 65], "color": "#FFF4DE"},
                    {"range": [65, 100],"color": "#E7F7EF"},
                ],
                "bgcolor": "#FFFAFC", "borderwidth": 2, "bordercolor": "#F4D7E1",
            },
            title={"text": "Similarity Score (gần 1 = cùng người)", "font": {"size": 13}},
        ))
        plotly_pastel_layout(gauge, height=260)

        cc1, cc2 = st.columns([1, 1.2])
        with cc1:
            st.plotly_chart(gauge, use_container_width=True)
            st.metric("Khoảng cách Euclid", f"{r['dist']:.4f}", help="Càng gần 0 = càng giống nhau")
            st.caption(f"Ground truth: **{r['true_label']}**")
            if r["score"] >= 0.65:
                st.markdown("<div class='badge badge-ok'>✅ HỢP LỆ — Đúng người chủ tài khoản</div>",
                            unsafe_allow_html=True)
            elif r["score"] >= 0.35:
                st.markdown("<div class='badge badge-warn'>⚠️ NGHI NGỜ — Yêu cầu xác thực bổ sung</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge badge-block'>🚫 ATO — Phát hiện kẻ mạo danh!</div>",
                            unsafe_allow_html=True)

        # ── SHAP cho Siamese (KernelExplainer trên hàm wrapper) ──────
        with cc2:
            st.markdown("**🔍 SHAP — Mỗi đặc trưng đóng góp bao nhiêu vào quyết định?**")
            st.caption("Đỏ = đẩy về phía «kẻ mạo danh» · Xanh = đẩy về phía «cùng người»")

            with st.spinner("Đang tính SHAP (Kernel Explainer ~5-10 giây)…"):
                try:
                    import shap
                    raw_users = raw_df.groupby("user_id")[list(FEATURE_COLS)].mean()
                    background_unscaled = raw_users.values.astype(np.float32)
                    ref_scaled_arr = scaler.transform(r["ref_vec"].reshape(1, -1)).astype(np.float32)

                    def predict_score(new_vecs_unscaled: np.ndarray) -> np.ndarray:
                        """Cố định reference, chỉ thay đổi 'session mới' → score."""
                        if new_vecs_unscaled.ndim == 1:
                            new_vecs_unscaled = new_vecs_unscaled.reshape(1, -1)
                        new_scaled = scaler.transform(new_vecs_unscaled).astype(np.float32)
                        n = new_scaled.shape[0]
                        ref_rep = np.tile(ref_scaled_arr, (n, 1)).astype(np.float32)
                        out = model.predict([ref_rep, new_scaled], verbose=0).flatten()
                        return out

                    explainer_k = shap.KernelExplainer(predict_score, background_unscaled)
                    shap_vals = explainer_k.shap_values(
                        r["new_vec"].reshape(1, -1).astype(np.float32),
                        nsamples=80, silent=True,
                    )
                    sv = np.asarray(shap_vals).flatten()
                    base_score = float(np.asarray(explainer_k.expected_value).flatten()[0])

                    # Vì SHAP value cao = score cao = "cùng người", ta đảo dấu để
                    # "đỏ = đẩy về phía kẻ mạo danh" (trực quan với khách hàng)
                    contrib = -sv
                    df_contrib = pd.DataFrame({
                        "feature": [FEATURE_VI[f] for f in FEATURE_COLS],
                        "shap":    contrib,
                        "value":   r["new_vec"],
                    })
                    df_contrib["abs"] = df_contrib["shap"].abs()
                    df_contrib = df_contrib.sort_values("abs", ascending=True).tail(10)
                    bar_colors = [
                        PALETTE["rose_dark"] if v > 0 else PALETTE["mint_dark"]
                        for v in df_contrib["shap"]
                    ]
                    labels = [
                        f"{f} = {v:.3g}" for f, v in zip(df_contrib["feature"], df_contrib["value"])
                    ]
                    fig_w = go.Figure(go.Bar(
                        x=df_contrib["shap"], y=labels,
                        orientation="h", marker_color=bar_colors,
                        text=[f"{v:+.3f}" for v in df_contrib["shap"]],
                        textposition="outside",
                    ))
                    fig_w.add_vline(x=0, line=dict(color=PALETTE["ink"], width=1))
                    fig_w.update_layout(
                        xaxis_title="Mức đẩy về phía 'kẻ mạo danh' ←  | →  'cùng người'",
                        title=f"Base score = {base_score:.3f} → Final score = {r['score']:.3f}",
                    )
                    plotly_pastel_layout(fig_w, height=420)
                    st.plotly_chart(fig_w, use_container_width=True)
                    st.session_state["t2_shap_top"] = [
                        (f, float(v)) for f, v in zip(df_contrib["feature"][::-1], df_contrib["shap"][::-1])
                    ]
                except ImportError:
                    st.error("❌ Thiếu `shap`. Chạy: `pip install shap`")
                    st.session_state["t2_shap_top"] = []
                except Exception as e:
                    st.error(f"❌ Lỗi tính SHAP: {e}")
                    st.session_state["t2_shap_top"] = []

        # ── Bổ trợ: % lệch của từng đặc trưng (intuitive cho khách hàng) ──
        st.markdown("##### 📐 Mức lệch tuyệt đối từng đặc trưng so với «chữ ký» của user gốc")
        st.caption("Bổ trợ trực quan cho SHAP — kể cả người không chuyên cũng hiểu ngay.")
        ref_vec = r["ref_vec"]; new_vec = r["new_vec"]
        delta_pct = (new_vec - ref_vec) / (np.abs(ref_vec) + 1e-9) * 100
        order = np.argsort(np.abs(delta_pct))[::-1]
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
        fig_d.update_layout(xaxis_title="% lệch so với chữ ký gõ của user gốc")
        plotly_pastel_layout(fig_d, height=380)
        st.plotly_chart(fig_d, use_container_width=True)

        # ── Embedding compare (16D) ─────────────────────────
        st.markdown("##### 🎼 So sánh «chữ ký» 16 chiều của 2 session")
        fig_e = go.Figure()
        x_idx = np.arange(16)
        fig_e.add_trace(go.Bar(
            x=x_idx - 0.2, y=r["emb_ref"], width=0.4,
            name="Reference (user gốc)", marker_color=PALETTE["lavender_dark"],
        ))
        fig_e.add_trace(go.Bar(
            x=x_idx + 0.2, y=r["emb_new"], width=0.4,
            name="Session mới", marker_color=PALETTE["rose_dark"],
        ))
        fig_e.update_layout(xaxis_title="Dimension", yaxis_title="Giá trị embedding",
                            barmode="overlay")
        plotly_pastel_layout(fig_e, height=300)
        st.plotly_chart(fig_e, use_container_width=True)

        with st.expander("💡 Cách giải thích Siamese Network bằng SHAP", expanded=False):
            st.markdown("""
**Vấn đề:** Siamese có 2 đầu vào song song chia sẻ trọng số → SHAP gốc trên
*toàn bộ cặp* khó diễn giải.

**Giải pháp đã triển khai:**
1. **Cố định reference** = «chữ ký» trung bình của user gốc.
2. **Wrap model** thành hàm 1 đầu vào: `f(new_vec) = score(ref, new_vec)`.
3. Dùng **`shap.KernelExplainer`** với background = trung bình của 20 user.
4. **Đảo dấu SHAP** để biểu đồ trực quan với khách hàng:
   - Đỏ = đặc trưng đẩy quyết định về phía **«kẻ mạo danh»**
   - Xanh = đặc trưng giữ quyết định ở phía **«cùng người»**

**Bổ trợ:** thêm 2 biểu đồ trực quan hơn cho người không chuyên:
- **% lệch** từng đặc trưng so với chữ ký gốc
- **So sánh embedding 16D** giữa 2 session
            """)

        # ── Khung chat Gemini ───────────────────────────────
        st.divider()
        chat_context = {
            "User gốc (reference)":      r["ref_user_id"],
            "Kịch bản test":             r["scenario"],
            "Ground truth":              r["true_label"],
            "Similarity Score (0–1)":    f"{r['score']:.4f}",
            "Khoảng cách Euclid":        f"{r['dist']:.4f}",
            "Quyết định hệ thống":       ("HỢP LỆ" if r["score"] >= 0.65 else
                                          "NGHI NGỜ" if r["score"] >= 0.35 else "ATO BLOCK"),
            "Đặc trưng lệch nhiều nhất": [
                f"{FEATURE_VI[FEATURE_COLS[i]]}: {(r['new_vec'][i]-r['ref_vec'][i])/(abs(r['ref_vec'][i])+1e-9)*100:+.1f}%"
                for i in np.argsort(np.abs((r['new_vec']-r['ref_vec'])/(np.abs(r['ref_vec'])+1e-9)))[-5:][::-1]
            ],
            "Top SHAP đẩy về phía kẻ mạo danh": [
                f"{f}: {v:+.4f}" for f, v in st.session_state.get("t2_shap_top", [])[:5]
            ],
        }
        render_chat_panel(
            context=chat_context,
            role="Sinh trắc học hành vi (Tầng 2 — Siamese Network) phát hiện ATO",
            key_prefix="t2_chat",
            suggested_questions=[
                "Vì sao session này bị nghi là kẻ mạo danh?",
                "Khách hàng có cần đổi mật khẩu không?",
                "Nhân viên cần xác thực thêm gì?",
            ],
            title="💬 Phân tích của AI (Gemini) — Giải thích hành vi cho khách hàng",
        )
