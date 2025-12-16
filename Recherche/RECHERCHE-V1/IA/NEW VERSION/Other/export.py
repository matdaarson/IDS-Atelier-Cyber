import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "part_22.csv"
EXPORT_DIR = "preprocessing_export"
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "part_1.csv"                 # CSV utilisé à l'entraînement
EXPORT_DIR = "preprocessing_export"

MISSING_THRESHOLD = 0.30
MAX_CATEG_CARDINALITY = 50
MAX_TOKENS_TEXT = 10000
MIN_DATETIME_PARSE_RATE = 0.70


# ============================================================
# UTILS
# ============================================================
def try_parse_datetime(col):
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    return parsed.notna().mean(), parsed


def split_columns(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = []

    # conversion datetime
    for c in obj_cols:
        rate, parsed = try_parse_datetime(df[c])
        if rate >= MIN_DATETIME_PARSE_RATE:
            df[c] = parsed.view("int64") / 1e9
            datetime_cols.append(c)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cat_cols, text_cols = [], []
    for c in obj_cols:
        if df[c].nunique(dropna=True) <= MAX_CATEG_CARDINALITY:
            cat_cols.append(c)
        else:
            text_cols.append(c)

    return df, num_cols, cat_cols, text_cols, datetime_cols


# ============================================================
# 1) load training data
# ============================================================
df = pd.read_csv(CSV_PATH)

keep = df.columns[df.isnull().mean() < MISSING_THRESHOLD]
df = df[keep].copy()

df, num_cols, cat_cols, text_cols, datetime_cols = split_columns(df)

# fill NA
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

for c in cat_cols + text_cols:
    df[c] = df[c].fillna("").astype(str)

print("\nColonnes utilisées :")
print("num     :", num_cols)
print("cat     :", cat_cols)
print("text    :", text_cols)
print("datetime:", datetime_cols)


# ============================================================
# 2) Create export directory
# ============================================================
os.makedirs(EXPORT_DIR, exist_ok=True)


# ============================================================
# 3) Export numeric normalization
# ============================================================
if num_cols:
    means = df[num_cols].mean().values
    stds = df[num_cols].std(ddof=0).values

    np.save(f"{EXPORT_DIR}/num_means.npy", means)
    np.save(f"{EXPORT_DIR}/num_stds.npy", stds)


# ============================================================
# 4) Export categorical vocabularies
# ============================================================
for c in cat_cols:
    vocab = sorted(df[c].unique().tolist())
    with open(f"{EXPORT_DIR}/{c}_lookup.txt", "w") as f:
        f.write("\n".join(vocab))


# ============================================================
# 5) Export TF-IDF vocab + IDF weights
# ============================================================
for c in text_cols:
    vec = layers.TextVectorization(
        max_tokens=MAX_TOKENS_TEXT,
        output_mode="tf_idf"
    )
    vec.adapt(df[c].values)

    vocab = vec.get_vocabulary()
    idf = vec.get_weights()[0]

    with open(f"{EXPORT_DIR}/{c}_tfidf_vocab.txt", "w") as f:
        f.write("\n".join(vocab))

    np.save(f"{EXPORT_DIR}/{c}_tfidf_idf.npy", idf)


# ============================================================
# 6) Save config file
# ============================================================
config = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "text_cols": text_cols,
    "datetime_cols": datetime_cols,
    "max_tokens_text": MAX_TOKENS_TEXT,
}

with open(f"{EXPORT_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=4)

print("\n✅ Préprocessing exporté dans preprocessing_export/")

MISSING_THRESHOLD = 0.30
MAX_CATEG_CARDINALITY = 50
MAX_TOKENS_TEXT = 10000
MIN_DATETIME_PARSE_RATE = 0.70


# ============================================================
# UTILS
# ============================================================
def try_parse_datetime(col):
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    return parsed.notna().mean(), parsed


def split_columns(df):
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
        if df[c].nunique(dropna=True) <= MAX_CATEG_CARDINALITY:
            cat_cols.append(c)
        else:
            text_cols.append(c)

    return df, num_cols, cat_cols, text_cols, datetime_cols


# ============================================================
# 1) LOAD & CLEAN TRAINING DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

# Drop empty columns
keep = df.columns[df.isnull().mean() < MISSING_THRESHOLD]
df = df[keep].copy()

df, num_cols, cat_cols, text_cols, datetime_cols = split_columns(df)

# Fill NA
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

for c in cat_cols + text_cols:
    df[c] = df[c].fillna("").astype(str)

print("Colonnes numériques :", num_cols)
print("Colonnes catégorielles :", cat_cols)
print("Colonnes texte :", text_cols)
print("Colonnes datetime :", datetime_cols)


# ============================================================
# 2) EXPORT DIRECTORY
# ============================================================
os.makedirs(EXPORT_DIR, exist_ok=True)


# ============================================================
# 3) EXPORT NUMERICAL NORMALIZATION
# ============================================================
if num_cols:
    means = df[num_cols].mean().values
    stds = df[num_cols].std(ddof=0).values

    np.save(os.path.join(EXPORT_DIR, "num_means.npy"), means)
    np.save(os.path.join(EXPORT_DIR, "num_stds.npy"), stds)


# ============================================================
# 4) EXPORT LOOKUP (CATÉGORIEL)
# ============================================================
for c in cat_cols:
    vocab = sorted(df[c].unique().tolist())
    with open(os.path.join(EXPORT_DIR, f"{c}_lookup.txt"), "w") as f:
        f.write("\n".join(vocab))


# ============================================================
# 5) EXPORT TEXT VECTOR VOCAB
# ============================================================
for c in text_cols:
    vec = layers.TextVectorization(max_tokens=MAX_TOKENS_TEXT, output_mode="tf_idf")
    vec.adapt(df[c].values)
    vocab = vec.get_vocabulary()

    with open(os.path.join(EXPORT_DIR, f"{c}_tfidf_vocab.txt"), "w") as f:
        f.write("\n".join(vocab))


# ============================================================
# 6) EXPORT PREPROCESSING CONFIG
# ============================================================
config = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "text_cols": text_cols,
    "datetime_cols": datetime_cols,
    "max_tokens_text": MAX_TOKENS_TEXT
}

with open(os.path.join(EXPORT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

print("\n✅ Préprocessing exporté dans preprocessing_export/")
