"""LightGBM utilities — Tầng 1 phát hiện giao dịch Fraud ATO."""
import numpy as np
import pandas as pd

# Bộ feature gợi ý cho realtime demo (dataset gốc của bài toán).
DEFAULT_FRAUD_FEATURES = [
    "time_gap_prev_min", "velocity_1h", "velocity_2ph", "velocity_24h",
    "cumulative_amount_1h", "amount", "amount_log",
    "amount_vs_avg_user", "amount_just_below_10M",
    "balance_drain_ratio", "is_new_recipient", "consecutive_new_recipients",
    "is_new_device", "hour_of_day", "is_night_hours",
]
TARGET_COL = "is_fraud"

FEATURE_VI = {
    "time_gap_prev_min":          "Khoảng cách giao dịch trước (phút)",
    "velocity_1h":                "Số giao dịch trong 1 giờ qua",
    "velocity_2ph":               "Số giao dịch trong 2 giờ qua",
    "velocity_24h":               "Số giao dịch trong 24 giờ qua",
    "cumulative_amount_1h":       "Tổng tiền chuyển trong 1 giờ qua",
    "amount":                     "Số tiền giao dịch",
    "amount_log":                 "Log số tiền giao dịch",
    "amount_vs_avg_user":         "Tỉ lệ so với mức trung bình của user",
    "amount_just_below_10M":      "Số tiền sát ngưỡng 10 triệu (né hạn mức)",
    "balance_drain_ratio":        "Tỉ lệ rút cạn số dư (% chuyển/tổng)",
    "is_new_recipient":           "Người nhận mới (chưa từng chuyển)",
    "consecutive_new_recipients": "Số người nhận mới liên tiếp",
    "is_new_device":              "Thiết bị mới (chưa từng đăng nhập)",
    "hour_of_day":                "Giờ trong ngày (0-23)",
    "is_night_hours":             "Giao dịch ban đêm (22h-5h)",
}


def feature_label(col: str) -> str:
    return FEATURE_VI.get(col, col)


def split_features(df: pd.DataFrame, target: str = TARGET_COL):
    """Tách X/y, bỏ cột target và bất kì cột không phải số."""
    if target not in df.columns:
        raise ValueError(f"Thiếu cột target '{target}' trong dữ liệu.")
    y = df[target].astype(int).values
    feature_cols = [
        c for c in df.columns
        if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].astype(float)
    return X, y, feature_cols


def train_lightgbm(X_train, y_train, X_val, y_val,
                   n_estimators: int = 300,
                   learning_rate: float = 0.05,
                   num_leaves: int = 31,
                   max_depth: int = -1,
                   class_balance: bool = True):
    """Huấn luyện LightGBM với early stopping.

    Trả về (model, evals_result_dict).
    """
    import lightgbm as lgb

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    spw = (neg / max(pos, 1.0)) if class_balance and pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        scale_pos_weight=spw,
        objective="binary",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.evals_result_


def find_optimal_threshold(model, X_val, y_val) -> float:
    """Tìm ngưỡng tối ưu trên tập validation theo F1-score.

    Vì sao cần?  Với dữ liệu mất cân bằng (Fraud ~1-8%), proba của model
    thường lệch về 0 → ngưỡng mặc định 0.5 sẽ KHÔNG có dự đoán dương →
    Precision = Recall = F1 = 0 (dù AUC vẫn cao).
    """
    from sklearn.metrics import precision_recall_curve
    proba = model.predict_proba(X_val)[:, 1]
    precs, recs, thrs = precision_recall_curve(y_val, proba)
    # f1 cho mỗi threshold; precision_recall_curve trả về thrs có len = n-1
    f1s = 2 * precs * recs / (precs + recs + 1e-9)
    if len(thrs) == 0:
        return 0.5
    best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    return float(thrs[best_idx])


def evaluate(model, X_test, y_test, threshold: float = 0.5):
    """Tính các metric chuẩn cho phân loại nhị phân."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        roc_curve, precision_recall_curve,
    )
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy":   accuracy_score(y_test, pred),
        "precision":  precision_score(y_test, pred, zero_division=0),
        "recall":     recall_score(y_test, pred, zero_division=0),
        "f1":         f1_score(y_test, pred, zero_division=0),
        "auc_roc":    roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan"),
        "auc_pr":     average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan"),
        "cm":         confusion_matrix(y_test, pred),
        "fpr_tpr":    roc_curve(y_test, proba) if len(np.unique(y_test)) > 1 else (None, None, None),
        "pr_curve":   precision_recall_curve(y_test, proba) if len(np.unique(y_test)) > 1 else (None, None, None),
        "proba":      proba,
        "pred":       pred,
    }


def get_shap_values(model, X_sample):
    """Tính SHAP cho LightGBM (TreeExplainer rất nhanh)."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    # Một số phiên bản trả list (binary) hoặc array
    if isinstance(sv, list):
        sv = sv[1]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.asarray(expected).ravel()[-1])
    return explainer, sv, float(expected)


# ─────────────────────────────────────────────
#  Preset transactions cho Tab Demo Realtime
# ─────────────────────────────────────────────
PRESETS = {
    "normal": {
        "label": "🟢 Giao dịch bình thường",
        "features": {
            "time_gap_prev_min": 720, "velocity_1h": 1, "velocity_2ph": 1, "velocity_24h": 3,
            "cumulative_amount_1h": 500_000, "amount": 500_000, "amount_log": np.log1p(500_000),
            "amount_vs_avg_user": 1.05, "amount_just_below_10M": 0,
            "balance_drain_ratio": 0.05, "is_new_recipient": 0, "consecutive_new_recipients": 0,
            "is_new_device": 0, "hour_of_day": 14, "is_night_hours": 0,
        },
        "ui": {
            "src": "9704****1234", "dst": "VCB - 1029384756", "amount": 500_000,
            "channel": "Mobile App", "device": "iPhone 14 (đã tin cậy)",
            "location": "Hà Nội — IP 113.161.x.x", "time": "14:32 hôm nay",
        },
    },
    "suspicious": {
        "label": "🟡 Đáng ngờ",
        "features": {
            "time_gap_prev_min": 7, "velocity_1h": 4, "velocity_2ph": 5, "velocity_24h": 9,
            "cumulative_amount_1h": 12_500_000, "amount": 8_900_000, "amount_log": np.log1p(8_900_000),
            "amount_vs_avg_user": 4.2, "amount_just_below_10M": 1,
            "balance_drain_ratio": 0.42, "is_new_recipient": 1, "consecutive_new_recipients": 2,
            "is_new_device": 0, "hour_of_day": 23, "is_night_hours": 1,
        },
        "ui": {
            "src": "9704****1234", "dst": "TCB - 6677889900 (mới)", "amount": 8_900_000,
            "channel": "Mobile App", "device": "iPhone 14",
            "location": "Hải Phòng — IP 27.65.x.x", "time": "23:18 hôm nay",
        },
    },
    "fraud": {
        "label": "🔴 Fraud rõ rệt",
        "features": {
            "time_gap_prev_min": 2, "velocity_1h": 8, "velocity_2ph": 11, "velocity_24h": 15,
            "cumulative_amount_1h": 78_000_000, "amount": 42_500_000, "amount_log": np.log1p(42_500_000),
            "amount_vs_avg_user": 18.5, "amount_just_below_10M": 0,
            "balance_drain_ratio": 0.93, "is_new_recipient": 1, "consecutive_new_recipients": 5,
            "is_new_device": 1, "hour_of_day": 3, "is_night_hours": 1,
        },
        "ui": {
            "src": "9704****1234", "dst": "MBB - 5544332211 (mới)", "amount": 42_500_000,
            "channel": "Mobile App", "device": "Samsung Galaxy A52 (THIẾT BỊ MỚI)",
            "location": "TP.HCM — IP 14.241.x.x (ĐỔI VÙNG)", "time": "03:07 sáng nay",
        },
    },
}


def recommend_action(score: float, top_factors: list[tuple[str, float]]) -> str:
    """Sinh khuyến nghị hành động dựa trên score + SHAP."""
    if score < 0.30:
        verdict = "✅ **APPROVED** — Cho phép giao dịch tự động."
        bullets = [
            "Hành vi giao dịch phù hợp với lịch sử của khách hàng.",
            "Thiết bị, kênh, và thời gian đều quen thuộc.",
            "Không cần xác thực bổ sung.",
        ]
    elif score < 0.65:
        verdict = "⚠️ **REVIEW** — Cần xác thực 2 yếu tố trước khi cho qua."
        bullets = [
            "Yêu cầu OTP qua SMS hoặc Smart-OTP.",
            "Có thể yêu cầu xác thực sinh trắc học (FaceID/TouchID).",
            "Lý do đáng ngờ chính: " + ", ".join([f for f, _ in top_factors[:3]]),
        ]
    else:
        verdict = "🚫 **BLOCKED** — Tạm khoá giao dịch, gọi tổng đài."
        bullets = [
            "Khoá tạm thời tài khoản và gửi thông báo cho khách hàng.",
            "Chuyển sang đội ngũ phòng chống gian lận xác minh trực tiếp.",
            "Yêu cầu xác thực lại qua video call (eKYC + rPPG).",
            "Yếu tố rủi ro chính: " + ", ".join([f for f, _ in top_factors[:3]]),
        ]
    return verdict + "\n\n" + "\n".join(f"- {b}" for b in bullets)


# ─────────────────────────────────────────────
#  REALTIME FEATURE ENGINEERING — dùng cho Tab Demo
# ─────────────────────────────────────────────
VN_BANKS = [
    "VCB — Vietcombank",  "BIDV — BIDV",            "CTG — Vietinbank",
    "AGR — Agribank",     "TCB — Techcombank",      "MB — MBBank",
    "ACB — ACB",          "VPB — VPBank",           "TPB — TPBank",
    "STB — Sacombank",    "SHB — SHB",              "HDB — HDBank",
    "OCB — OCB",          "MSB — MSB",              "EIB — Eximbank",
    "VIB — VIB",          "SCB — SCB",              "ABB — ABBank",
    "NAB — Nam A Bank",   "PGB — PGBank",           "PVCB — PVcomBank",
    "BVB — BaoVietBank",  "SeABank",                "Kienlongbank",
]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Tìm cột đầu tiên khớp tên trong list, không phân biệt hoa thường."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def parse_history_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Tìm cột timestamp (hoặc ghép date+time) → trả về df đã sort theo thời gian
    và tên cột timestamp (sau khi normalize tên)."""
    ts_col = find_col(df, [
        "timestamp", "datetime", "date_time", "transaction_time",
        "txn_time", "time_stamp", "created_at", "trans_time",
    ])
    if ts_col is not None:
        out = df.copy()
        out["_ts"] = pd.to_datetime(out[ts_col], errors="coerce")
        out = out.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
        return out, "_ts"

    # Fallback: ghép date + time
    date_col = find_col(df, ["date", "transaction_date", "txn_date", "day"])
    time_col = find_col(df, ["time", "txn_hour", "hour"])
    if date_col is not None:
        out = df.copy()
        if time_col is not None:
            combined = out[date_col].astype(str) + " " + out[time_col].astype(str)
        else:
            combined = out[date_col].astype(str)
        out["_ts"] = pd.to_datetime(combined, errors="coerce")
        out = out.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
        return out, "_ts"

    raise ValueError(
        "Không tìm thấy cột thời gian trong CSV. Cần có ít nhất một trong: "
        "`timestamp`, `datetime`, `date_time`, `transaction_time`, `created_at`, "
        "hoặc cặp `date` + `time`."
    )


def compute_realtime_features(
    history_df: pd.DataFrame,
    new_ts: pd.Timestamp,
    new_amount: float,
    new_recipient: str,
    new_device: str,
    balance_before: float,
) -> dict:
    """Tính 15 feature cho 1 giao dịch mới dựa trên lịch sử ~80 ngày.

    Args:
        history_df       : DataFrame lịch sử giao dịch của user
        new_ts           : timestamp của giao dịch mới (pd.Timestamp)
        new_amount       : số tiền giao dịch mới (VND)
        new_recipient    : ID/STK người nhận mới
        new_device       : device_id đang dùng
        balance_before   : số dư TRƯỚC giao dịch mới (lấy từ balance_after dòng cuối)
    """
    h, ts_col = parse_history_timestamps(history_df)
    rec_col = find_col(h, ["recipient_account", "recipient_id", "recipient",
                            "to_account", "destination_account", "dst_account"])
    dev_col = find_col(h, ["device_id", "device", "device_used", "main_device"])
    amt_col = find_col(h, ["amount", "transaction_amount", "txn_amount"])

    if amt_col is None:
        raise ValueError("Không tìm thấy cột `amount` trong CSV.")

    # ── 1. time_gap_prev_min
    if len(h) > 0:
        last_ts = h[ts_col].iloc[-1]
        time_gap_prev_min = max((new_ts - last_ts).total_seconds() / 60.0, 0.0)
    else:
        time_gap_prev_min = 0.0

    # ── 2-4. velocity_*
    one_hour_ago  = new_ts - pd.Timedelta(hours=1)
    two_min_ago   = new_ts - pd.Timedelta(minutes=2)
    one_day_ago   = new_ts - pd.Timedelta(hours=24)
    velocity_1h   = int((h[ts_col] >= one_hour_ago).sum())
    velocity_2ph  = int((h[ts_col] >= two_min_ago).sum())
    velocity_24h  = int((h[ts_col] >= one_day_ago).sum())

    # ── 5. cumulative_amount_1h
    cumulative_amount_1h = float(h.loc[h[ts_col] >= one_hour_ago, amt_col].sum())

    # ── 6, 7. amount, amount_log
    amount = float(new_amount)
    amount_log = float(np.log1p(amount))

    # ── 8. amount_vs_avg_user (so với 30 ngày trước)
    thirty_days_ago = new_ts - pd.Timedelta(days=30)
    recent_30d = h.loc[h[ts_col] >= thirty_days_ago, amt_col]
    avg_30d = float(recent_30d.mean()) if len(recent_30d) > 0 else float(h[amt_col].mean() or 1.0)
    amount_vs_avg_user = amount / max(avg_30d, 1.0)

    # ── 9. amount_just_below_10M
    amount_just_below_10M = int(8_000_000 <= amount < 10_000_000)

    # ── 10. balance_drain_ratio
    balance_drain_ratio = amount / max(float(balance_before), 1.0)

    # ── 11. is_new_recipient
    if rec_col is not None and new_recipient:
        seen_recipients = set(h[rec_col].astype(str).str.strip())
        is_new_recipient = int(str(new_recipient).strip() not in seen_recipients)
    else:
        is_new_recipient = 1

    # ── 12. consecutive_new_recipients
    if rec_col is not None and new_recipient:
        consec = 1 if is_new_recipient else 0
        # Đếm ngược các giao dịch gần nhất có cùng người nhận và chưa từng xuất hiện trước đó
        if is_new_recipient:
            for i in range(len(h) - 1, -1, -1):
                if str(h[rec_col].iloc[i]).strip() == str(new_recipient).strip():
                    consec += 1
                else:
                    break
        consecutive_new_recipients = consec
    else:
        consecutive_new_recipients = 0

    # ── 13. is_new_device
    if dev_col is not None and new_device:
        seen_devices = set(h[dev_col].astype(str).str.strip())
        is_new_device = int(str(new_device).strip() not in seen_devices)
    else:
        is_new_device = 0

    # ── 14, 15. hour features
    hour_of_day = int(new_ts.hour)
    is_night_hours = int(0 <= hour_of_day <= 5)

    return {
        "time_gap_prev_min":          round(time_gap_prev_min, 6),
        "velocity_1h":                 velocity_1h,
        "velocity_2ph":                velocity_2ph,
        "velocity_24h":                velocity_24h,
        "cumulative_amount_1h":        round(cumulative_amount_1h, 2),
        "amount":                      round(amount, 2),
        "amount_log":                  round(amount_log, 6),
        "amount_vs_avg_user":          round(amount_vs_avg_user, 6),
        "amount_just_below_10M":       amount_just_below_10M,
        "balance_drain_ratio":         round(balance_drain_ratio, 6),
        "is_new_recipient":            is_new_recipient,
        "consecutive_new_recipients":  consecutive_new_recipients,
        "is_new_device":               is_new_device,
        "hour_of_day":                 hour_of_day,
        "is_night_hours":              is_night_hours,
    }


def synthetic_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Sinh dataset giả lập (dùng khi user chưa upload CSV) cho phép demo nhanh."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * 0.08)
    n_legit = n - n_fraud
    rows = []
    for _ in range(n_legit):
        amount = float(rng.lognormal(13.5, 0.8))
        rows.append({
            "time_gap_prev_min":          float(rng.exponential(360)),
            "velocity_1h":                int(rng.poisson(0.6)),
            "velocity_2ph":                int(rng.poisson(1.0)),
            "velocity_24h":               int(rng.poisson(3.0)),
            "cumulative_amount_1h":       float(amount * rng.uniform(1, 1.5)),
            "amount":                      amount,
            "amount_log":                  float(np.log1p(amount)),
            "amount_vs_avg_user":          float(rng.normal(1.0, 0.4)),
            "amount_just_below_10M":       int(amount > 9_000_000 and amount < 10_000_000),
            "balance_drain_ratio":         float(np.clip(rng.beta(1.5, 6), 0, 1)),
            "is_new_recipient":            int(rng.random() < 0.18),
            "consecutive_new_recipients":  int(rng.poisson(0.3)),
            "is_new_device":               int(rng.random() < 0.05),
            "hour_of_day":                 int(rng.integers(7, 23)),
            "is_night_hours":              0,
            "is_fraud":                    0,
        })
    for _ in range(n_fraud):
        amount = float(rng.lognormal(15.5, 1.0))
        hour = int(rng.choice([1, 2, 3, 4, 23, 0]))
        rows.append({
            "time_gap_prev_min":          float(rng.exponential(8)),
            "velocity_1h":                int(rng.poisson(4)) + 1,
            "velocity_2ph":                int(rng.poisson(6)) + 1,
            "velocity_24h":               int(rng.poisson(10)) + 1,
            "cumulative_amount_1h":       float(amount * rng.uniform(1.5, 4)),
            "amount":                      amount,
            "amount_log":                  float(np.log1p(amount)),
            "amount_vs_avg_user":          float(rng.normal(8, 3)),
            "amount_just_below_10M":       int(rng.random() < 0.35),
            "balance_drain_ratio":         float(np.clip(rng.beta(5, 1.5), 0, 1)),
            "is_new_recipient":            int(rng.random() < 0.85),
            "consecutive_new_recipients":  int(rng.poisson(3)) + 1,
            "is_new_device":               int(rng.random() < 0.6),
            "hour_of_day":                 hour,
            "is_night_hours":              int(hour <= 5 or hour >= 22),
            "is_fraud":                    1,
        })
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
