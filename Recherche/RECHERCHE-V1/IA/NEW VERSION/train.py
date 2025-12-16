import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks,  initializers, regularizers

from data import get_autoencoder_dataset_train, save_schema




def build_autoencoder(input_dim: int, latent_dim: int = None, final_activation="sigmoid"):
    if latent_dim is None:
        latent_dim = max(4, input_dim // 8)

    he_init = initializers.HeNormal()

    # ----- Encoder -----
    encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")

    x = layers.Dense(input_dim, kernel_initializer=he_init)(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Dense(input_dim // 2, kernel_initializer=he_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Dense(input_dim // 4, kernel_initializer=he_init)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.15)(x)

    latent = layers.Dense(
        latent_dim,
        activation=None,
        kernel_regularizer=regularizers.l2(1e-5),
        name="latent_vector"
    )(x)

    encoder = models.Model(encoder_input, latent, name="encoder")

    # ----- Decoder -----
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")

    y = layers.Dense(input_dim // 4, kernel_initializer=he_init)(decoder_input)
    y = layers.LeakyReLU(alpha=0.1)(y)

    y = layers.Dense(input_dim // 2, kernel_initializer=he_init)(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.1)(y)

    y = layers.Dense(input_dim, kernel_initializer=he_init)(y)
    y = layers.LeakyReLU(alpha=0.1)(y)

    # Sortie finale
    decoder_output = layers.Dense(input_dim, activation=final_activation)(y)

    decoder = models.Model(decoder_input, decoder_output, name="decoder")

    # ----- Autoencoder -----
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = models.Model(encoder_input, autoencoder_output, name="autoencoder")

    return autoencoder, encoder, decoder



def compute_reconstruction_errors(autoencoder, X):
    reconstructed = autoencoder.predict(X, verbose=0)
    errors = np.mean((X - reconstructed) ** 2, axis=1)
    return errors


def train_autoencoder(
    csv_path="part_1.csv",
    test_size=0.2,
    random_state=42,
    batch_size=256,
    epochs=50,
    model_dir="models",
    schema_path="schema.json"
):
    os.makedirs(model_dir, exist_ok=True)

    # 1) load/clean/scale TRAIN
    df_clean, X, scaler = get_autoencoder_dataset_train(csv_path)
    print(f"[INFO] Train dataset shape: {X.shape}")

    # 2) save schema + scaler
    save_schema(df_clean.columns, schema_path=schema_path)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("[INFO] Saved schema.json and scaler.pkl")

    # 3) split
    X_train, X_val = train_test_split(
        X, test_size=test_size, random_state=random_state, shuffle=True
    )
    input_dim = X_train.shape[1]

    # 4) build
    autoencoder, encoder, decoder = build_autoencoder(input_dim)
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    autoencoder.summary()

    # 5) callbacks
    cb_early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    cb_ckpt = callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "autoencoder_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # 6) train
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[cb_early, cb_reduce_lr, cb_ckpt],
        verbose=1
    )

    # 7) Sauvegarde du model
    autoencoder.save(os.path.join(model_dir, "autoencoder_final.keras"))
    encoder.save(os.path.join(model_dir, "encoder_final.keras"))
    decoder.save(os.path.join(model_dir, "decoder_final.keras"))

    # 8) learn threshold ON TRAIN errors and save it
    train_errors = compute_reconstruction_errors(autoencoder, X_train)
    threshold = np.percentile(train_errors, 95)  # top 5% train errors = anomalies
    np.save(os.path.join(model_dir, "threshold.npy"), threshold)
    print(f"[INFO] Threshold learned on train (p95): {threshold:.6f}")

    print("[INFO] Training complete.")
    return autoencoder, encoder, decoder, history


if __name__ == "__main__":
    train_autoencoder("part_19.csv")
