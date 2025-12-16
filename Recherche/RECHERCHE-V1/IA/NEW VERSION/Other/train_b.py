import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "part_1.csv"
MISSING_THRESHOLD = 0.30

BATCH_SIZE = 64
EPOCHS = 200
VALID_SPLIT = 0.2
SEED = 42

MAX_CATEG_CARDINALITY = 50
MIN_DATETIME_PARSE_RATE = 0.70

ENCODER_DIMS = [512, 256, 128]
LATENT_DIM = 64
DROPOUT_RATE = 0.15
LR = 1e-3


# ============================================================
# UTILS
# ============================================================
def try_parse_datetime(col: pd.Series):
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    return parsed.notna().mean(), parsed


def split_columns(df: pd.DataFrame):
    # Détection datetime
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = []
    for c in obj_cols:
        rate, parsed = try_parse_datetime(df[c])
        if rate >= MIN_DATETIME_PARSE_RATE:
            df[c] = parsed.view("int64") / 1e9
            datetime_cols.append(c)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # catégorielles UNIQUEMENT
    cat_cols = []
    for c in obj_cols:
        if df[c].nunique(dropna=True) <= MAX_CATEG_CARDINALITY:
            cat_cols.append(c)

    return df, num_cols, cat_cols, [], datetime_cols   # text_cols = []


# ============================================================
# 1. LOAD CSV + CLEANING
# ============================================================
df = pd.read_csv(CSV_PATH)

keep_cols = df.columns[df.isnull().mean() < MISSING_THRESHOLD]
df = df[keep_cols].copy()

df, num_cols, cat_cols, text_cols, datetime_cols = split_columns(df)

print("Colonnes numériques :", num_cols)
print("Colonnes catégorielles :", cat_cols)
print("Colonnes datetime :", datetime_cols)

# Fill NA
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    df[c] = df[c].fillna("").astype(str)


# ============================================================
# 2. PREPROCESSING MODEL (Label Encoding)
# ============================================================
inputs = {}
encoded_parts = []

# -- Numériques normalisées
if num_cols:
    num_input = layers.Input(shape=(len(num_cols),), dtype=tf.float32, name="num_in")
    norm = layers.Normalization(name="num_norm")
    norm.adapt(df[num_cols].values.astype("float32"))
    num_encoded = norm(num_input)
    inputs["num_in"] = num_input
    encoded_parts.append(num_encoded)

# -- Label Encoding pour catégories
for c in cat_cols:

    inp = layers.Input(shape=(1,), dtype=tf.string, name=f"{c}_in")

    lookup = layers.StringLookup(output_mode="int", name=f"{c}_lookup")
    lookup.adapt(df[c].values)

    label = lookup(inp)

    # Normalisation : entiers convertis en float entre 0 et 1
    max_val = lookup.vocabulary_size()
    label = layers.Rescaling(1.0 / max_val, name=f"{c}_scaled")(label)

    inputs[f"{c}_in"] = inp
    encoded_parts.append(label)

# -- Concaténation
features = layers.Concatenate(name="all_features")(encoded_parts)

preprocess_model = models.Model(inputs=inputs, outputs=features, name="preprocess")
os.makedirs("models", exist_ok=True)
preprocess_model.save("models/preprocessing.keras")
preprocess_model.summary()


# ============================================================
# 3. TF.DATA DATASET
# ============================================================
def df_to_dict(df_batch):
    out = {}
    if num_cols:
        out["num_in"] = df_batch[num_cols].values.astype("float32")
    for c in cat_cols:
        out[f"{c}_in"] = df_batch[c].values.astype(str)
    return out


df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(df) * VALID_SPLIT)
df_val = df.iloc[:n_val]
df_train = df.iloc[n_val:]

def make_dataset(df_part, training=True):
    ds = tf.data.Dataset.from_tensor_slices(df_to_dict(df_part))
    if training:
        ds = ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (preprocess_model(x), preprocess_model(x)),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(df_train)
val_ds = make_dataset(df_val)


# ============================================================
# 4. AUTOENCODER
# ============================================================
for xb, _ in train_ds.take(1):
    input_dim = xb.shape[1]

def build_autoencoder(input_dim):
    inp = layers.Input(shape=(input_dim,), name="ae_in")
    x = inp

    for d in ENCODER_DIMS:
        x = layers.Dense(d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(DROPOUT_RATE)(x)

    latent = layers.Dense(LATENT_DIM, name="latent")(x)

    x = latent
    for d in reversed(ENCODER_DIMS):
        x = layers.Dense(d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(DROPOUT_RATE)(x)

    out = layers.Dense(input_dim, name="recon")(x)

    model = models.Model(inp, out)

    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-5)
    except:
        opt = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


autoencoder = build_autoencoder(input_dim)
autoencoder.summary()


# ============================================================
# 5. CALLBACKS & TRAIN
# ============================================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/autoencoder_best.keras", save_best_only=True, monitor="val_loss"),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
]

history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

autoencoder.save("models/autoencoder_final.keras")
print("✔ Entraînement terminé.")
