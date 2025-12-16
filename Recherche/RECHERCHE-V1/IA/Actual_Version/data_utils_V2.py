import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler


# Taille de l'embedding que tu utiliseras dans les modèles
EMBEDDING_DIM = 16  


def build_category_maps(df, str_cols):
    """
    Crée un dictionnaire {col: {value: index}} pour toutes les colonnes textuelles
    """
    maps = {}

    for col in str_cols:
        uniques = df[col].fillna("NA").astype(str).unique()
        maps[col] = {v: i+1 for i, v in enumerate(uniques)}  # 0 = padding
        print(f"🔑 {col}: {len(uniques)} catégories")

    return maps


def apply_category_maps(df, maps):
    """
    Transforme les colonnes texte en entiers selon les dictionnaires générés
    """
    encoded_blocks = []

    for col, mapping in maps.items():
        encoded = df[col].fillna("NA").astype(str).map(mapping).fillna(0).astype(int)
        encoded_blocks.append(encoded.values.reshape(-1, 1))

    if len(encoded_blocks) == 0:
        return None

    return np.concatenate(encoded_blocks, axis=1)


def get_datasets(batch_size=128, directory_path="data/divided"):

    # ----------------------------------------------------------
    # 1) Charger les CSV
    # ----------------------------------------------------------
    all_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".csv")
    ]

    dfs = [pd.read_csv(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"📊 Données chargées : {len(df)} lignes, {len(df.columns)} colonnes")

    # Détection colonnes numériques / textuelles
    num_cols = df.select_dtypes(include=[np.number]).columns
    str_cols = df.select_dtypes(exclude=[np.number]).columns

    print("📘 Colonnes numériques :", list(num_cols))
    print("📙 Colonnes textuelles :", list(str_cols))

    # ----------------------------------------------------------
    # 2) Encodage des colonnes textuelles → entiers
    # ----------------------------------------------------------
    category_maps = build_category_maps(df, str_cols)
    joblib.dump(category_maps, "category_maps.pkl")
    print("📁 category_maps.pkl sauvegardé.")

    X_cat = apply_category_maps(df, category_maps)

    # ----------------------------------------------------------
    # 3) Normalisation des colonnes numériques
    # ----------------------------------------------------------
    X_num = df[num_cols].astype(np.float32).values

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    joblib.dump(scaler, "scaler.pkl")
    print("📁 scaler.pkl sauvegardé.")

    # ----------------------------------------------------------
    # 4) Construction de X final
    # ----------------------------------------------------------
    if X_cat is not None:
        X = np.concatenate([X_num, X_cat], axis=1)
    else:
        X = X_num

    print(f"📐 Shape finale X = {X.shape}")

    # ----------------------------------------------------------
    # 5) Split Numpy 80 / 10 / 10
    # ----------------------------------------------------------
    n = len(X)
    idx = np.random.permutation(n)

    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = X[idx[:train_end]]
    X_val   = X[idx[train_end:val_end]]

    # ----------------------------------------------------------
    # 6) Création des datasets autoencodeur
    # ----------------------------------------------------------
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_train, X_train))
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_val, X_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("✅ Datasets prêts.")
    return train_ds, val_ds, (X.shape[1],), {}
