"""20 personas + sinh session/pair cho Tầng 2 (Siamese)."""
import numpy as np
import pandas as pd

PERSONAS = [
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

FEATURE_VI = {
    "avg_pressure":       "Áp lực ngón TB",
    "std_pressure":       "Độ lệch áp lực",
    "avg_touch_area":     "Diện tích chạm TB",
    "std_touch_area":     "Độ lệch diện tích",
    "avg_touch_duration": "Thời lượng chạm TB",
    "std_touch_duration": "Độ lệch thời lượng",
    "avg_inter_gap":      "Khoảng nghỉ TB",
    "std_inter_gap":      "Độ lệch khoảng nghỉ",
    "avg_gyro_x":         "Gyro X TB",
    "std_gyro_x":         "Độ lệch Gyro X",
    "avg_gyro_y":         "Gyro Y TB",
    "std_gyro_y":         "Độ lệch Gyro Y",
}


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


def generate_pairs(df: pd.DataFrame, n_same: int = 500, n_diff: int = 500, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique()
    pairs = []
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
        row.update({"user_A": s1["user_id"], "user_B": s2["user_id"],
                    "label": 1, "label_text": "Cùng người"})
        pairs.append(row)
    for _ in range(n_diff):
        uid_a, uid_b = rng.choice(users, 2, replace=False)
        s1 = df[df["user_id"] == uid_a].iloc[rng.integers(len(df[df["user_id"] == uid_a]))]
        s2 = df[df["user_id"] == uid_b].iloc[rng.integers(len(df[df["user_id"] == uid_b]))]
        row = {}
        for f in FEATURE_COLS:
            row[f"A_{f}"] = s1[f]
            row[f"B_{f}"] = s2[f]
        row.update({"user_A": uid_a, "user_B": uid_b,
                    "label": 0, "label_text": "Khác người"})
        pairs.append(row)
    return pd.DataFrame(pairs).sample(frac=1, random_state=seed).reset_index(drop=True)
