import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_bilstm(input_len=200, channels=3, dropout=0.3):
    """
    CNN-BiLSTM for BP regression.
    Input: (None, input_len, channels)
    Output: 2 (SBP, DBP)
    """
    inputs = layers.Input(shape=(input_len, channels), name="ppg_seq")

    # 1D CNN feature extractor
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    # BiLSTM temporal aggregator
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(dropout)(x)

    # Dense heads
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(2, name="bp_out")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_bilstm_bp")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae", "mse"])
    return model
