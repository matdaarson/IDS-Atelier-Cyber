import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import AdamW
import os


def train_model(model, train_ds, val_ds, model_name):
    # --- Dossiers de logs et modèles ---
    os.makedirs("Models", exist_ok=True)
    os.makedirs("CSV", exist_ok=True)
    log_dir = f"logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)



    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="mse",
        metrics=["mae"]
    )


    # --- Callbacks utiles ---
    checkpoint = ModelCheckpoint(
        filepath=f"Models/{model_name}_best.keras",
        monitor="val_loss",             # on surveille la reconstruction (pas accuracy)
        save_best_only=True,
        verbose=1
    )

    #early_stop = EarlyStopping(
   #     monitor="val_loss",             # arrêt si plus d'amélioration
     #   patience=10,
    #    restore_best_weights=True
   # )

    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    csv_logger = CSVLogger(f"CSV/{model_name}_history.csv")

    # --- Entraînement ---
    with tf.device('/GPU:0'):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=2,
            callbacks=[checkpoint, tensorboard_cb, csv_logger],
            verbose=1
        )

    # --- Sauvegarde finale du modèle ---
    model.save(f'Models/{model_name}_final.keras')

    # --- Récupération de la meilleure erreur ---
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = min(history.history['val_mae'])

    print(f"🏁 {model_name} — Meilleure erreur validation (MSE) : {best_val_loss:.6f}")
    print(f"📉 {model_name} — Meilleure erreur MAE validation : {best_val_mae:.6f}")

    return best_val_loss
