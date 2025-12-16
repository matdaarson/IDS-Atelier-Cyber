import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

CSV_TEST_PATH = "part_22.csv"
MODEL_BEST_PATH = "models/autoencoder_best.keras"
MODEL_FINAL_PATH = "models/autoencoder_final.keras"

MISSING_THRESHOLD = 0.30
MAX_CATEG_CARDINALITY = 50
MAX_TOKENS_TEXT = 10000
MIN_DATETIME_PARSE_RATE = 0.70


# ============================================================
# UTILS
# ============================================================
def try_parse_datetime(col: pd.Series):
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    return parsed.notna().mean(), parsed


def split_columns(df: pd.DataFrame):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = []
    for c in obj_cols:
        rate, parsed = try_parse_datetime(df[c])
        if rate >= MIN_DATETIME_PARSE_RATE:
            df[c] = parsed.view("int64") / 1e9
            datetime_cols.append(c)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cat_cols, text_cols = [], []
    for c in obj_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= MAX_CATEG_CARDINALITY:
            cat_cols.append(c)
        else:
            text_cols.append(c)

    return df, num_cols, cat_cols, text_cols, datetime_cols


def df_to_dict(df_batch, num_cols, cat_cols, text_cols):
    out = {}
    if num_cols:
        out["num_in"] = df_batch[num_cols].values.astype("float32")
    for c in cat_cols:
        out[f"{c}_in"] = df_batch[c].values.astype(str)
    for c in text_cols:
        out[f"{c}_in"] = df_batch[c].values.astype(str)
    return out


# ============================================================
# 1. LOAD TEST CSV
# ============================================================
df = pd.read_csv(CSV_TEST_PATH)

keep_cols = df.columns[df.isnull().mean() < MISSING_THRESHOLD]
df = df[keep_cols].copy()

df, num_cols, cat_cols, text_cols, datetime_cols = split_columns(df)

# Fill NA
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

for c in cat_cols + text_cols:
    df[c] = df[c].fillna("").astype(str)


# ============================================================
# 2. REBUILD PREPROCESSING
# ============================================================
inputs = {}
encoded_parts = []

# numériques
if num_cols:
    num_input = layers.Input(shape=(len(num_cols),), dtype=tf.float32, name="num_in")
    norm = layers.Normalization()
    norm.adapt(df[num_cols].values.astype("float32"))
    encoded_parts.append(norm(num_input))
    inputs["num_in"] = num_input

# catégorielles
for c in cat_cols:
    inp = layers.Input(shape=(1,), dtype=tf.string, name=f"{c}_in")
    lookup = layers.StringLookup(output_mode="one_hot")
    lookup.adapt(df[c].values)
    encoded_parts.append(lookup(inp))
    inputs[f"{c}_in"] = inp

# texte
for c in text_cols:
    inp = layers.Input(shape=(1,), dtype=tf.string, name=f"{c}_in")
    vec = layers.TextVectorization(max_tokens=MAX_TOKENS_TEXT, output_mode="tf_idf")
    vec.adapt(df[c].values)
    encoded_parts.append(vec(inp))
    inputs[f"{c}_in"] = inp

features = layers.Concatenate()(encoded_parts)
preprocess_model = models.Model(inputs=inputs, outputs=features)

X = preprocess_model(df_to_dict(df, num_cols, cat_cols, text_cols)).numpy()

# ============================================================
# 3. LOAD AUTOENCODER
# ============================================================
if os.path.exists(MODEL_BEST_PATH):
    autoencoder = models.load_model(MODEL_BEST_PATH)
else:
    autoencoder = models.load_model(MODEL_FINAL_PATH)


# ============================================================
# 4. INFERENCE + ANOMALY DETECTION
# ============================================================
reco = autoencoder.predict(X, verbose=0)
mse = np.mean((X - reco) ** 2, axis=1)

# --- seuil dynamique ---
threshold = mse.mean() + 3 * mse.std()

# 1 = anomalie, 0 = normal
is_anomaly = (mse > threshold).astype(int)

print("\nThreshold utilisé :", threshold)
print("Nombre d'anomalies détectées :", is_anomaly.sum())


# ============================================================
# 5. SAVE RESULTS
# ============================================================
df_out = df.copy()
df_out["reconstruction_error"] = mse
df_out["anomaly"] = is_anomaly

df_out.to_csv("results_test.csv", index=False)

#print("\n✅ results_test.csv généré avec succès !")
#print("→ Contient reconstruction_error + anomaly = {0,1}")
