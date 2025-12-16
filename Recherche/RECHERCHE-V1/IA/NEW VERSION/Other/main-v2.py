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

# heuristiques colonnes
MAX_CATEG_CARDINALITY = 50   # <= 50 valeurs uniques => catégorielle
MAX_TOKENS_TEXT = 10000      # vocab TF-IDF max
MIN_DATETIME_PARSE_RATE = 0.70

# architecture
ENCODER_DIMS = [512, 256, 128]
LATENT_DIM = 64
DROPOUT_RATE = 0.15
LR = 1e-3


# ============================================================
# UTILS
# ============================================================
def try_parse_datetime(col: pd.Series):
    """Essaie de parser une colonne object en datetime.
       Retourne (success_rate, parsed_series)."""
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    success_rate = parsed.notna().mean()
    return success_rate, parsed


def split_columns(df: pd.DataFrame):
    """Sépare numérique / catégoriel / texte avec heuristiques."""
    # 1) tenter de convertir des colonnes object en datetime si possible
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    datetime_cols = []
    for c in obj_cols:
        rate, parsed = try_parse_datetime(df[c])
        if rate >= MIN_DATETIME_PARSE_RATE:
            df[c] = parsed.view("int64") / 1e9  # seconds
            datetime_cols.append(c)

    # maj colonnes object après conversion datetime
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 2) split object -> categorical vs text
    cat_cols, text_cols = [], []
    for c in obj_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= MAX_CATEG_CARDINALITY:
            cat_cols.append(c)
        else:
            text_cols.append(c)

    return df, num_cols, cat_cols, text_cols, datetime_cols


# ============================================================
# 1. LOAD + DROP HIGH-MISSING COLS
# ============================================================
df = pd.read_csv(CSV_PATH)

# drop cols > 30% missing
keep_cols = df.columns[df.isnull().mean() < MISSING_THRESHOLD]
df = df[keep_cols].copy()

# split columns
df, num_cols, cat_cols, text_cols, datetime_cols = split_columns(df)

print("Colonnes numériques :", num_cols)
print("Colonnes catégorielles :", cat_cols)
print("Colonnes texte :", text_cols)
print("Colonnes datetime converties :", datetime_cols)

# ============================================================
# 2. FILL NA (robuste)
# ============================================================
# num => median
for c in num_cols:
    med = df[c].median()
    df[c] = df[c].fillna(med)

# cat/text => ""
for c in cat_cols + text_cols:
    df[c] = df[c].fillna("").astype(str)

# ============================================================
# 3. BUILD PREPROCESSING MODEL
# ============================================================
inputs = {}
encoded_parts = []

# --- numériques normalisées
if num_cols:
    num_input = layers.Input(shape=(len(num_cols),), dtype=tf.float32, name="num_in")
    normalizer = layers.Normalization(name="num_norm")
    normalizer.adapt(df[num_cols].values.astype("float32"))
    num_encoded = normalizer(num_input)
    inputs["num_in"] = num_input
    encoded_parts.append(num_encoded)

# --- catégorielles one-hot
for c in cat_cols:
    inp = layers.Input(shape=(1,), dtype=tf.string, name=f"{c}_in")
    lookup = layers.StringLookup(output_mode="one_hot", name=f"{c}_lookup")
    lookup.adapt(df[c].values)
    onehot = lookup(inp)
    inputs[f"{c}_in"] = inp
    encoded_parts.append(onehot)

# --- texte TF-IDF
for c in text_cols:
    inp = layers.Input(shape=(1,), dtype=tf.string, name=f"{c}_in")
    vec = layers.TextVectorization(
        max_tokens=MAX_TOKENS_TEXT,
        output_mode="tf_idf",
        name=f"{c}_tfidf"
    )
    vec.adapt(df[c].values)
    tfidf = vec(inp)
    inputs[f"{c}_in"] = inp
    encoded_parts.append(tfidf)

# concat final
if len(encoded_parts) == 1:
    features = encoded_parts[0]
else:
    features = layers.Concatenate(name="all_features")(encoded_parts)

preprocess_model = models.Model(inputs=inputs, outputs=features, name="preprocess")
preprocess_model.save("models/preprocessing.keras")
print("Préprocessing sauvegardé dans models/preprocessing.keras")
preprocess_model.summary()


# ============================================================
# 4. TF.DATA DATASET (AUTOENCODER => X=Y)
# ============================================================
def df_to_dict(df_batch):
    out = {}
    if num_cols:
        out["num_in"] = df_batch[num_cols].values.astype("float32")
    for c in cat_cols:
        out[f"{c}_in"] = df_batch[c].values.astype(str)
    for c in text_cols:
        out[f"{c}_in"] = df_batch[c].values.astype(str)
    return out

# shuffle / split train-val
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(df) * VALID_SPLIT)
df_val = df.iloc[:n_val]
df_train = df.iloc[n_val:]

def make_dataset(df_part, training=True):
    ds = tf.data.Dataset.from_tensor_slices(df_to_dict(df_part))
    if training:
        ds = ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # map -> preprocess -> (x,x)
    ds = ds.map(lambda x: (preprocess_model(x), preprocess_model(x)),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(df_train, training=True)
val_ds = make_dataset(df_val, training=False)

# déterminer input_dim dynamiquement
for xb, _ in train_ds.take(1):
    input_dim = xb.shape[1]
print("Input dim preprocessé =", input_dim)


# ============================================================
# 5. AUTOENCODER OPTIMISÉ
# ============================================================
def build_autoencoder(input_dim):
    inp = layers.Input(shape=(input_dim,), name="ae_in")

    x = inp
    # Encoder
    for d in ENCODER_DIMS:
        x = layers.Dense(d, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(DROPOUT_RATE)(x)

    latent = layers.Dense(LATENT_DIM, activation=None, name="latent")(x)

    # Decoder (symétrique)
    x = latent
    for d in reversed(ENCODER_DIMS):
        x = layers.Dense(d, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(DROPOUT_RATE)(x)

    out = layers.Dense(input_dim, activation=None, name="recon")(x)

    ae = models.Model(inp, out, name="autoencoder")

    # Optimizer robuste
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-5)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=LR)

    ae.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return ae

autoencoder = build_autoencoder(input_dim)
autoencoder.summary()


# ============================================================
# 6. CALLBACKS PERFORMANTS
# ============================================================
os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/autoencoder_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# 7. TRAIN
# ============================================================
history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

autoencoder.save("models/autoencoder_final.keras")
print("\n✅ Autoencodeur entraîné et sauvegardé dans models/")
