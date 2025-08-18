# New Model Train

This directory contains scripts to preprocess PPG data, train a CNN-BiLSTM model for SBP/DBP regression, run inference, and export the trained model.

## Expected Input Format
Provide a CSV with at least these columns:
- `ppg`: raw PPG signal samples (one column, long series)
- `sbp`: systolic BP values aligned to windows (optional for training)
- `dbp`: diastolic BP values aligned to windows (optional for training)

If you only have `ppg`, you can run preprocessing and inference (using a trained model). For training, `sbp` and `dbp` are required.

## Preprocessing
Preprocessing implements the 5-phase pipeline:
1) Signal processing (detrend, bandpass 0.5-5 Hz, smoothing, outlier clipping)
2) 3-channel features (PPG, 1st derivative, 2nd derivative)
3) Sequence creation: length=200, overlap=75% (step=50). Label per window via median of SBP/DBP inside window if labels are provided.
4) Data augmentation: applied to training split (scale, noise, jitter) with 3x factor.
5) Stratified split by BP category (Normal, Elevated, Stage 1, Stage 2, Crisis) if labels provided; otherwise simple random split.

Outputs saved as `.npz` with arrays:
- `X`: shape (N, 200, 3)
- `y`: shape (N, 2) with SBP, DBP (if available)
- `meta.json`: sampling rate and split indices

## Scripts
- `preprocess.py`: run preprocessing
- `model.py`: CNN-BiLSTM model definition (Keras)
- `train.py`: train and save `best_enhanced_cnn_bilstm_model.h5`
- `infer_export.py`: run inference on CSV and optional export to SavedModel

## Quickstart
```bash
# Preprocess
python "preprocess.py" --csv "../static/samples/sample_ppg.csv" --fs 125 --out data

# If you have labeled CSV with columns ppg,sbp,dbp
python "preprocess.py" --csv "path/to/labeled.csv" --fs 125 --out data --labeled

# Train (requires labeled dataset processed above)
python "train.py" --data data --epochs 50 --batch 32

# Inference on a CSV using trained model
python "infer_export.py" --model "best_enhanced_cnn_bilstm_model.h5" --csv "../static/samples/sample_ppg.csv" --fs 125
```

The Flask app (`app.py`) points to this trained model path by default:
`new model train\best_enhanced_cnn_bilstm_model.h5`
