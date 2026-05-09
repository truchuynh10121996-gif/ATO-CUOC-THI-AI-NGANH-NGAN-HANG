"""rPPG nâng cao — POS + Multi-ROI + Multi-window HR Stability (Tầng 3)."""
import os
import tempfile
import numpy as np


def save_video(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def extract_multi_roi(video_path: str, max_frames: int = 300):
    """Trích xuất RGB trung bình từ 3 ROI: trán, má trái, má phải."""
    import cv2
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_proc = min(max_frames, total)

    rois = {k: {"r": [], "g": [], "b": []}
            for k in ("forehead", "left_cheek", "right_cheek")}
    last_face = None
    n_det = 0

    for _ in range(n_proc):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(detected) > 0:
            last_face = max(detected, key=lambda f: f[2] * f[3])
            n_det += 1

        Hf, Wf = frame.shape[:2]

        def patch(y1r, y2r, x1r, x2r):
            y1 = max(0, int(y1r)); y2 = min(Hf, int(y2r))
            x1 = max(0, int(x1r)); x2 = min(Wf, int(x2r))
            p = frame[y1:y2, x1:x2]
            if p.size == 0:
                return None
            return (float(np.mean(p[:, :, 2])),
                    float(np.mean(p[:, :, 1])),
                    float(np.mean(p[:, :, 0])))

        if last_face is not None:
            x, y, w, h = last_face
            region_vals = {
                "forehead":    patch(y+h*.08, y+h*.35, x+w*.25, x+w*.75),
                "left_cheek":  patch(y+h*.40, y+h*.70, x+w*.10, x+w*.42),
                "right_cheek": patch(y+h*.40, y+h*.70, x+w*.58, x+w*.90),
            }
        else:
            region_vals = {k: None for k in rois}

        for k, val in region_vals.items():
            if val is None:
                pr = rois[k]["r"][-1] if rois[k]["r"] else 128.0
                pg = rois[k]["g"][-1] if rois[k]["g"] else 128.0
                pb = rois[k]["b"][-1] if rois[k]["b"] else 128.0
                rois[k]["r"].append(pr); rois[k]["g"].append(pg); rois[k]["b"].append(pb)
            else:
                rois[k]["r"].append(val[0]); rois[k]["g"].append(val[1]); rois[k]["b"].append(val[2])

    cap.release()
    return rois, fps, n_det, len(rois["forehead"]["r"])


def pos_signal(r_arr, g_arr, b_arr) -> np.ndarray:
    """POS — Wang et al. 2017: kết hợp 3 kênh RGB → SNR cao hơn GREEN đơn thuần."""
    r = np.asarray(r_arr, dtype=np.float64)
    g = np.asarray(g_arr, dtype=np.float64)
    b = np.asarray(b_arr, dtype=np.float64)
    rn = r / (r.mean() + 1e-12)
    gn = g / (g.mean() + 1e-12)
    bn = b / (b.mean() + 1e-12)
    s1 = rn - gn
    s2 = rn + gn - 2.0 * bn
    alpha = (s1.std() + 1e-12) / (s2.std() + 1e-12)
    return s1 + alpha * s2


def combine_rois(rois: dict) -> np.ndarray:
    sigs = []
    for k in ("forehead", "left_cheek", "right_cheek"):
        s = pos_signal(rois[k]["r"], rois[k]["g"], rois[k]["b"])
        sigs.append(s / (s.std() + 1e-12))
    return np.mean(sigs, axis=0)


def bandpass(signal: np.ndarray, fps: float, low: float = 0.75, high: float = 3.0) -> np.ndarray:
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


def multiwindow_hr(signal: np.ndarray, fps: float, window_sec: float = 4.0, step_sec: float = 1.0):
    from scipy.signal import welch
    win = int(window_sec * fps)
    step = max(1, int(step_sec * fps))
    hr_list, t_list = [], []
    for start in range(0, len(signal) - win, step):
        seg = signal[start:start + win]
        freqs, psd = welch(seg, fps, nperseg=min(len(seg), 128))
        band = (freqs >= 0.75) & (freqs <= 3.0)
        if not np.any(band):
            continue
        pf = float(freqs[band][np.argmax(psd[band])])
        hr_list.append(pf * 60.0)
        t_list.append((start + win / 2.0) / fps)
    return np.array(hr_list, dtype=np.float64), np.array(t_list, dtype=np.float64)


def quality(signal: np.ndarray, fps: float, hr_arr: np.ndarray):
    import math
    from scipy.signal import welch, detrend
    sig = detrend(signal.astype(np.float64))
    freqs, psd = welch(sig, fps, nperseg=min(len(sig), 256))
    band = (freqs >= 0.75) & (freqs <= 3.0)
    if not np.any(band):
        return 0.0, 0.0, 0.5, 0.0, freqs, psd

    psd_hr = psd[band]
    prominence = float(psd_hr.max()) / (float(psd_hr.mean()) + 1e-12)
    q_prom = float(np.clip(math.log(max(prominence, 1.0)) / math.log(12), 0.0, 1.0))

    if len(hr_arr) >= 3:
        hr_std = float(hr_arr.std())
        q_stab = float(np.clip(1.0 - (hr_std - 5.0) / 25.0, 0.0, 1.0))
    else:
        q_stab = 0.5
    q_combined = 0.5 * q_prom + 0.5 * q_stab
    hr_bpm = float(freqs[band][np.argmax(psd_hr)]) * 60.0
    return hr_bpm, q_prom, q_stab, q_combined, freqs, psd


def safe_unlink(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass
