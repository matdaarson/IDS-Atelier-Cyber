import tensorflow as tf
import time
import os

# Activation automatique du mixed-precision
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed precision activée")
except Exception:
    print("ℹ️ Mixed precision non disponible sur cette version de TF")


# -------------------------------------------------------------------------
# 🔥 One-Cycle Learning Rate Scheduler
# -------------------------------------------------------------------------
class OneCycleLR(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, total_steps, pct_start=0.3, final_div_factor=1e3):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor

    def on_train_begin(self, logs=None):
        self.step = 0
        lr = float(self.model.optimizer.learning_rate.numpy())
        self.initial_lr = lr

    def on_batch_end(self, batch, logs=None):
        self.step += 1
        progress = self.step / self.total_steps

        if progress < self.pct_start:
            # phase montée
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (progress / self.pct_start)
        else:
            # phase descente jusqu'à final lr
            final_lr = self.max_lr / self.final_div_factor
            lr = final_lr + (self.max_lr - final_lr) * (1 - (progress - self.pct_start) / (1 - self.pct_start))

        self.model.optimizer.learning_rate.assign(lr)


# -------------------------------------------------------------------------
# 🔥 ProgressBar propre
# -------------------------------------------------------------------------
class ProgressBar(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"\r🟦 batch {batch}  loss={logs.get('loss'):.5f}  val_loss={logs.get('val_loss')}", end="")


# -------------------------------------------------------------------------
# 🔥 Fonction d'entraînement principale
# -------------------------------------------------------------------------
def train_model(
    model,
    train_ds,
    val_ds,
    model_name="model",
    epochs=25,
    max_lr=1e-3,
    patience=5,
    save_dir="checkpoints_autoenc",
):

    os.makedirs(save_dir, exist_ok=True)

    # Récupération safe du learning rate initial
    lr_attr = (
        model.optimizer.learning_rate
        if hasattr(model.optimizer, "learning_rate")
        else model.optimizer.lr
    )
    base_lr = float(lr_attr.numpy())

    # Nombre total d'étapes
    total_steps = (len(train_ds)) * epochs

    callbacks = [
        ProgressBar(),
        OneCycleLR(max_lr=max_lr, total_steps=total_steps),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{save_dir}/{model_name}.keras",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False
        ),
    ]

    print(f"\n🚀 Début entraînement: {model_name}")
    print(f"📌 Learning rate initial: {base_lr}")
    print(f"📌 OneCycle max_lr: {max_lr}")

    t0 = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )

    train_time = time.time() - t0

    best_val = min(history.history["val_loss"])

    print(f"\n✅ Fin entraînement {model_name}")
    print(f"⏱️ Temps total: {train_time:.1f}s")
    print(f"📉 Meilleure val_loss: {best_val:.5f}")

    return best_val
