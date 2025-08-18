import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import pywt
from sklearn.model_selection import train_test_split

BP_BINS = [0, 90, 120, 130, 140, 180, 1000]
BP_LABELS = ["Hypo", "Normal", "Elevated", "Stage1", "Stage2", "Crisis"]


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


def smooth(x, k=5):
    if k <= 1:
        return x
    return signal.medfilt(x, kernel_size=k if k % 2 else k + 1)


def build_channels(x):
    dx = np.gradient(x)
    ddx = np.gradient(dx)
    return np.stack([x, dx, ddx], axis=-1)


def create_sequences(x3, y=None, win=200, step=50):
    X, Y = [], []
    for start in range(0, max(0, len(x3) - win + 1), step):
        seg = x3[start : start + win]
        if seg.shape[0] != win:
            continue
        X.append(seg)
        if y is not None:
            Y.append(np.median(y[start : start + win], axis=0))
    X = np.asarray(X)
    Y = np.asarray(Y) if y is not None else None
    return X, Y


def categorize_bp(y):
    # y shape (N,2) -> categories from SBP/DBP
    sbp = y[:, 0]
    dbp = y[:, 1]
    # category by worst of sbp/dbp
    sbp_cat = np.digitize(sbp, BP_BINS, right=False) - 1
    dbp_cat = np.digitize(dbp, BP_BINS, right=False) - 1
    cat = np.maximum(sbp_cat, dbp_cat)
    return cat


def preprocess_csv(csv_path, fs, labeled=False, win=200, step=50):
    df = pd.read_csv(csv_path)
    # Get PPG column
    ppg_col = None
    for cand in ["ppg", "PPG", "signal", df.columns[0]]:
        if cand in df.columns:
            ppg_col = cand
            break
    if ppg_col is None:
        raise ValueError("Could not find PPG column in CSV")

    ppg = df[ppg_col].astype(float).values
    ppg = ppg - np.mean(ppg)
    ppg = bandpass_filter(ppg, fs)
    ppg = wavelet_denoise(ppg)
    ppg = smooth(ppg, k=5)

    x3 = build_channels(ppg)

    y = None
    if labeled:
        if not {"sbp", "dbp"}.issubset(set(map(str.lower, df.columns))):
            # try case-insensitive mapping
            cols_lower = {c.lower(): c for c in df.columns}
            if "sbp" in cols_lower and "dbp" in cols_lower:
                sbp = df[cols_lower["sbp"]].astype(float).values
                dbp = df[cols_lower["dbp"]].astype(float).values
            else:
                raise ValueError("Labeled mode requires sbp and dbp columns")
        else:
            # exact names present
            sbp = df[[c for c in df.columns if c.lower() == "sbp"][0]].astype(float).values
            dbp = df[[c for c in df.columns if c.lower() == "dbp"][0]].astype(float).values
        y = np.vstack([sbp, dbp]).T

    X, Y = create_sequences(x3, y, win=win, step=step)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--fs", type=int, default=125, help="Sampling rate")
    ap.add_argument("--out", default="data", help="Output directory")
    ap.add_argument("--labeled", action="store_true", help="CSV contains sbp,dbp columns")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y = preprocess_csv(args.csv, fs=args.fs, labeled=args.labeled)

    indices = np.arange(len(X))
    splits = {"train": [], "val": [], "test": []}

    if Y is not None:
        cats = categorize_bp(Y)
        X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = train_test_split(
            X, Y, indices, test_size=args.test_size, random_state=args.seed, stratify=cats
        )
        # val split from train portion
        cats_tmp = categorize_bp(y_tmp)
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_tmp, y_tmp, idx_tmp, test_size=0.5, random_state=args.seed, stratify=cats_tmp
        )
        np.savez_compressed(out_dir / "dataset.npz", X=X, y=Y)
        splits = {
            "train": idx_train.tolist(),
            "val": idx_val.tolist(),
            "test": idx_test.tolist(),
        }
    else:
        # Unlabeled: simple split indices
        X_train, X_tmp, idx_train, idx_tmp = train_test_split(
            X, indices, test_size=args.test_size, random_state=args.seed
        )
        X_val, X_test, idx_val, idx_test = train_test_split(
            X_tmp, idx_tmp, test_size=0.5, random_state=args.seed
        )
        np.savez_compressed(out_dir / "dataset.npz", X=X)
        splits = {
            "train": idx_train.tolist(),
            "val": idx_val.tolist(),
            "test": idx_test.tolist(),
        }

    meta = {
        "fs": args.fs,
        "win": 200,
        "step": 50,
        "labeled": bool(Y is not None),
        "splits": splits,
        "source_csv": os.path.abspath(args.csv),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved dataset to {out_dir}")


if __name__ == "__main__":
    main()
