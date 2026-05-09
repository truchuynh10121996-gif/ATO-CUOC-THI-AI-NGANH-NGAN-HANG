"""💳 Tầng 1 — Phát hiện giao dịch Fraud ATO bằng LightGBM."""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from theme import inject_pastel_css, matplotlib_pastel, plotly_pastel_layout, PALETTE, PALETTE_LIST
from lib_lightgbm import (
    DEFAULT_FRAUD_FEATURES, TARGET_COL, FEATURE_VI, feature_label,
    split_features, train_lightgbm, evaluate, get_shap_values,
    PRESETS, recommend_action, synthetic_dataset,
)
from lib_gemini import render_chat_panel

inject_pastel_css()
matplotlib_pastel()

st.markdown("<div class='hero-tag'>💳 TẦNG 1 · LIGHTGBM</div>", unsafe_allow_html=True)
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

    up = st.file_uploader("Tải lên file CSV", type=["csv"], key="t1_csv")
    use_synth = st.checkbox(
        "Hoặc dùng **dữ liệu giả lập** (5.000 giao dịch, ~8% fraud) để demo nhanh",
        value=(up is None),
    )

    df = None
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.success(f"✅ Đã đọc {len(df):,} dòng từ `{up.name}`.")
        except Exception as e:
            st.error(f"❌ Không đọc được CSV: {e}")
    elif use_synth:
        df = synthetic_dataset(n=5000, seed=42)
        st.info("ℹ️ Đang dùng dữ liệu giả lập. Upload CSV để dùng dữ liệu thật.")

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

            with st.spinner("⏳ Đang huấn luyện…"):
                X, y, feat_cols = split_features(df, TARGET_COL)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
                )
                model, history = train_lightgbm(
                    X_tr, y_tr, X_va, y_va,
                    n_estimators=n_est, learning_rate=float(lr), num_leaves=n_leaves,
                )
                metrics = evaluate(model, X_te, y_te)
                st.session_state["t1_model"] = model
                st.session_state["t1_features"] = feat_cols
                st.session_state["t1_X_te"] = X_te
                st.session_state["t1_y_te"] = y_te
                st.session_state["t1_metrics"] = metrics
                st.session_state["t1_history"] = history
            st.success(f"✅ Huấn luyện xong! Best iter = {model.best_iteration_}")

    # ── Hiển thị kết quả nếu đã train ─────────────────────────────
    if "t1_metrics" in st.session_state:
        m = st.session_state["t1_metrics"]
        model = st.session_state["t1_model"]
        feat_cols = st.session_state["t1_features"]
        X_te = st.session_state["t1_X_te"]
        y_te = st.session_state["t1_y_te"]

        st.markdown("#### 📈 Kết quả đánh giá trên tập test")
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
#  TAB 2 — DEMO REALTIME
# ════════════════════════════════════════════════════════════════════
with tab_demo:
    if "t1_model" not in st.session_state:
        st.warning("⚠️ Vui lòng huấn luyện model ở tab **Huấn luyện** trước.")
        st.stop()

    model = st.session_state["t1_model"]
    feat_cols = st.session_state["t1_features"]

    st.markdown("#### 🏦 Mô phỏng một giao dịch chuyển tiền")

    # ── Preset buttons ────────────────────────────────────────────
    p1, p2, p3 = st.columns(3)
    if p1.button(PRESETS["normal"]["label"], key="preset_normal", use_container_width=True):
        st.session_state["t1_preset"] = "normal"
    if p2.button(PRESETS["suspicious"]["label"], key="preset_suspicious", use_container_width=True):
        st.session_state["t1_preset"] = "suspicious"
    if p3.button(PRESETS["fraud"]["label"], key="preset_fraud", use_container_width=True):
        st.session_state["t1_preset"] = "fraud"

    preset_key = st.session_state.get("t1_preset", "normal")
    preset = PRESETS[preset_key]

    left, right = st.columns([1, 1.2], gap="large")

    # ─── CỘT TRÁI: Form giao dịch ────────────────────────────────
    with left:
        st.markdown("##### 💳 Giao dịch chuyển tiền")
        ui = preset["ui"]
        st.markdown(
            f"""
<div class='pastel-card'>
<table style='width:100%;font-size:14px'>
<tr><td>👤 <b>Tài khoản nguồn</b></td><td style='text-align:right'>{ui['src']}</td></tr>
<tr><td>🎯 <b>Tài khoản đích</b></td><td style='text-align:right'>{ui['dst']}</td></tr>
<tr><td>💰 <b>Số tiền</b></td><td style='text-align:right;color:{PALETTE['rose_dark']};font-weight:700;font-size:17px'>{ui['amount']:,} VNĐ</td></tr>
<tr><td>📲 <b>Kênh</b></td><td style='text-align:right'>{ui['channel']}</td></tr>
<tr><td>🕒 <b>Thời gian</b></td><td style='text-align:right'>{ui['time']}</td></tr>
<tr><td>📍 <b>Vị trí</b></td><td style='text-align:right'>{ui['location']}</td></tr>
<tr><td>📱 <b>Thiết bị</b></td><td style='text-align:right'>{ui['device']}</td></tr>
</table>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Các đặc trưng đã trích xuất (feature engineering):")
        feat_show = pd.DataFrame([
            {"Đặc trưng": feature_label(k), "Giá trị": f"{v:,.4f}" if isinstance(v, float) else str(v)}
            for k, v in preset["features"].items()
        ])
        st.dataframe(feat_show, hide_index=True, use_container_width=True, height=320)

    # ─── CỘT PHẢI: Kết quả AI ────────────────────────────────────
    with right:
        st.markdown("##### 🤖 Kết quả AI")

        # Build feature vector matching trained model
        x_row = pd.DataFrame([[
            preset["features"].get(c, 0) for c in feat_cols
        ]], columns=feat_cols)

        proba = float(model.predict_proba(x_row)[0, 1])

        # ── Gauge chart ──────────────────────────────────────────
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
        st.plotly_chart(gauge, use_container_width=True)

        # ── Badge ────────────────────────────────────────────────
        if proba < 0.30:
            st.markdown("<div class='badge badge-ok'>✅ APPROVED — Cho qua tự động</div>",
                        unsafe_allow_html=True)
        elif proba < 0.65:
            st.markdown("<div class='badge badge-warn'>⚠️ REVIEW — Yêu cầu xác thực bổ sung</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge badge-block'>🚫 BLOCKED — Tạm khoá giao dịch</div>",
                        unsafe_allow_html=True)

    # ── SHAP waterfall (full width dưới 2 cột) ───────────────────
    st.markdown("##### 🔍 SHAP — Vì sao AI quyết định như vậy?")
    st.caption(
        "Mỗi thanh = mức ảnh hưởng của 1 đặc trưng tới quyết định cuối. "
        "Đỏ = đẩy về phía Fraud, xanh = kéo về phía Hợp lệ."
    )

    try:
        import shap
        explainer, shap_vals, base_val = get_shap_values(model, x_row)
        # shap_vals shape: (1, n_features)
        sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

        # ── Vẽ waterfall pastel bằng plotly (vì shap.plots không kiểm soát màu)
        contribs = pd.DataFrame({
            "feature": [feature_label(c) for c in feat_cols],
            "value":   x_row.iloc[0].values,
            "shap":    sv,
        })
        contribs["abs"] = contribs["shap"].abs()
        contribs = contribs.sort_values("abs", ascending=False).head(10)
        contribs = contribs.iloc[::-1]  # đảo cho waterfall đẹp

        bar_colors = [
            PALETTE["rose_dark"] if v > 0 else PALETTE["mint_dark"]
            for v in contribs["shap"]
        ]
        labels = [
            f"{f} = {v:.3g}" for f, v in zip(contribs["feature"], contribs["value"])
        ]

        fig_w = go.Figure(go.Bar(
            x=contribs["shap"], y=labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in contribs["shap"]],
            textposition="outside",
        ))
        fig_w.add_vline(x=0, line=dict(color=PALETTE["ink"], width=1))
        fig_w.update_layout(
            xaxis_title="Mức đóng góp tới điểm Fraud (SHAP value)",
            yaxis_title="",
            title=f"Base = {base_val:+.3f} → Final logit ≈ {base_val + sv.sum():+.3f}",
        )
        plotly_pastel_layout(fig_w, height=420)
        st.plotly_chart(fig_w, use_container_width=True)

        top_factors = list(zip(contribs["feature"][::-1], contribs["shap"][::-1]))
    except ImportError:
        st.error("❌ Thiếu thư viện `shap`. Chạy: `pip install shap streamlit-shap`")
        top_factors = []
    except Exception as e:
        st.error(f"❌ Lỗi tính SHAP: {e}")
        top_factors = []

    # ── Khuyến nghị hành động ────────────────────────────────────
    st.markdown("##### 📋 Khuyến nghị hành động")
    st.markdown(recommend_action(proba, top_factors))

    # ── Khung chat Gemini ───────────────────────────────────────
    st.divider()
    chat_context = {
        "Loại preset đang xét":   preset["label"],
        "Số tiền giao dịch (VNĐ)": f"{preset['ui']['amount']:,}",
        "Tài khoản đích":         preset["ui"]["dst"],
        "Thời gian":              preset["ui"]["time"],
        "Thiết bị":               preset["ui"]["device"],
        "Điểm rủi ro AI (0–1)":   f"{proba:.4f}",
        "Xác suất Fraud (%)":     f"{proba*100:.2f}",
        "Quyết định hệ thống":    ("APPROVED" if proba < 0.3 else "REVIEW" if proba < 0.65 else "BLOCKED"),
        "Top yếu tố ảnh hưởng":   [f"{f}: {v:+.3f}" for f, v in top_factors[:6]],
    }
    render_chat_panel(
        context=chat_context,
        role="Phát hiện gian lận giao dịch chuyển tiền (Tầng 1 — LightGBM)",
        key_prefix="t1_chat",
        suggested_questions=[
            "Vì sao giao dịch này bị đánh dấu rủi ro?",
            "Tôi nên giải thích thế nào với khách hàng?",
            "Khách hàng cần làm gì tiếp theo?",
        ],
        title="💬 Phân tích của AI (Gemini) — Giải thích cho khách hàng",
    )
