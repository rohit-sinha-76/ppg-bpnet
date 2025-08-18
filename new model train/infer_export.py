import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
import pywt


def bandpass_filter(x, fs, low=0.5, high=5.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def wavelet_denoise(x, wavelet="db4", level=5):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.std(coeffs[-1])
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[: len(x)]


def build_channels(x):
    dx = np.gradient(x)
    ddx = np.gradient(dx)
    return np.stack([x, dx, ddx], axis=-1)


def create_sequences(x3, win=200, step=50):
    X = []
    for start in range(0, max(0, len(x3) - win + 1), step):
        seg = x3[start : start + win]
        if seg.shape[0] == win:
            X.append(seg)
    return np.asarray(X)


def classify_bp(sbp, dbp):
    if sbp < 90 or dbp < 60:
        return "Hypotension"
    elif sbp < 120 and dbp < 80:
        return "Normal"
    elif (120 <= sbp < 130) and dbp < 80:
        return "Elevated"
    elif (130 <= sbp < 140) or (80 <= dbp < 90):
        return "Hypertension Stage 1"
    elif (140 <= sbp < 180) or (90 <= dbp < 120):
        return "Hypertension Stage 2"
    else:
        return "Hypertensive Crisis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained Keras .h5 model")
    ap.add_argument("--csv", required=True, help="Path to CSV containing a PPG column")
    ap.add_argument("--fs", type=int, default=125)
    ap.add_argument("--export_savedmodel", help="Directory to export SavedModel (optional)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ppg_col = None
    for cand in ["ppg", "PPG", "signal", df.columns[0]]:
        if cand in df.columns:
            ppg_col = cand
            break
    if ppg_col is None:
        raise ValueError("Could not find PPG column in CSV")

    x = df[ppg_col].astype(float).values
    x = x - np.mean(x)
    x = bandpass_filter(x, args.fs)
    x = wavelet_denoise(x)

    x3 = build_channels(x)
    X = create_sequences(x3, win=200, step=50)
    if len(X) == 0:
        raise ValueError("Signal too short after preprocessing for 200-length windows")

    model = tf.keras.models.load_model(args.model)
    preds = model.predict(X, verbose=0)

    sbp = float(np.mean(preds[:, 0]))
    dbp = float(np.mean(preds[:, 1]))
    category = classify_bp(sbp, dbp)

    result = {"systolic": sbp, "diastolic": dbp, "bp_category": category}
    print(json.dumps(result, indent=2))

    if args.export_savedmodel:
        export_dir = Path(args.export_savedmodel)
        export_dir.mkdir(parents=True, exist_ok=True)
        tf.saved_model.save(model, str(export_dir))
        print(f"Exported SavedModel to: {export_dir}")


if __name__ == "__main__":
    main()
