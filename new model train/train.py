import argparse
import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import build_cnn_bilstm


def load_dataset(data_dir: Path):
    npz = np.load(data_dir / "dataset.npz")
    X = npz["X"]
    y = npz["y"] if "y" in npz.files else None
    with open(data_dir / "meta.json", "r") as f:
        meta = json.load(f)
    splits = meta.get("splits", {})
    return X, y, meta, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Directory containing dataset.npz and meta.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="best_enhanced_cnn_bilstm_model.h5", help="Output model filename")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data)
    X, y, meta, splits = load_dataset(data_dir)

    if y is None:
        raise ValueError("This dataset is unlabeled. Training requires y (SBP, DBP). Rerun preprocessing with --labeled.")

    idx_train = np.array(splits["train"], dtype=int)
    idx_val = np.array(splits["val"], dtype=int)

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]

    input_len = X.shape[1]
    channels = X.shape[2]

    model = build_cnn_bilstm(input_len=input_len, channels=channels)
    model.summary()

    ckpt_path = Path(__file__).resolve().parent / args.out

    callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(monitor="val_mae", mode="min", patience=args.patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5, patience=max(3, args.patience // 3), verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    # Save final model as well
    final_path = Path(__file__).resolve().parent / ("final_" + args.out)
    model.save(final_path)

    # Save training history
    hist_path = Path(__file__).resolve().parent / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history.history, f, indent=2)

    print(f"Best model (by val_mae) saved to: {ckpt_path}")
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
