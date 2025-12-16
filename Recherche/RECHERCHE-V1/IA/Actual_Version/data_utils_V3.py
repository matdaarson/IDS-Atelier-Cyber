import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def select_columns(df, min_features=5):
    """
    Sélectionne automatiquement les colonnes à garder en fonction
    des valeurs nulles, en optimisant un score simple.
    
    - min_features : nombre minimum de colonnes à conserver
    Retourne:
        - df_filtered : DataFrame réduit
        - kept_columns : liste des colonnes conservées
        - best_threshold : seuil de nulls correspondant (~ max null des colonnes gardées)
    """
    null_ratio = df.isnull().mean()  # % de null par colonne
    sorted_ratios = null_ratio.sort_values()  # de la colonne la plus propre à la pire
    
    best_score = -np.inf
    best_threshold = 1.0
    best_keep_cols = df.columns
    
    # garder les k meilleures colonnes (k varie de min_features à tout)
    for k in range(min_features, len(sorted_ratios) + 1):
        keep = sorted_ratios.index[:k]
        max_null_among_keep = sorted_ratios.iloc[k-1]
        mean_null_among_keep = null_ratio[keep].mean()
        
        
        score = k * (1.0 - mean_null_among_keep)
        
        if score > best_score:
            best_score = score
            best_threshold = float(max_null_among_keep)
            best_keep_cols = keep
    
    print(f"[INFO] Seuil optimal estimé ≈ {best_threshold*100:.1f}% de valeurs nulles.")
    print(f"[INFO] Colonnes conservées ({len(best_keep_cols)}/{len(df.columns)}): {list(best_keep_cols)}")
    
    df_filtered = df[list(best_keep_cols)].copy()
    return df_filtered, list(best_keep_cols), best_threshold




def build_autoencoder_datasets(
    df,
    batch_size=128,
    test_size=0.2,
    min_features=5,
    random_state=42
):
    """
    - Sélectionne automatiquement les colonnes en fonction des nulls
    - Impute les valeurs nulles
    - Crée des tf.data.Dataset pour l'autoencodeur
    
    Retourne:
        - train_ds : dataset TF pour l'entraînement
        - val_ds   : dataset TF pour la validation
        - normalizer : couche de normalisation adaptée sur X_train
        - kept_columns : colonnes utilisées
        - best_threshold : seuil optimal trouvé
    """
    # 1) Sélection intelligente des colonnes
    df_filtered, kept_columns, best_threshold = select_columns(
        df, min_features=min_features
    )
    
    # 2) Imputation simple des valeurs nulles (médiane par colonne)
    df_filtered = df_filtered.copy()
    df_filtered = df_filtered.fillna(df_filtered.median(numeric_only=True))
    
    # 3) Conversion en float32 (nécessaire pour TF)
    X = df_filtered.astype("float32").values
    
    # 4) Split train / validation
    X_train, X_val = train_test_split(
        X, test_size=test_size, random_state=random_state
    )
    
    # 5) Couche de normalisation
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    
    # 6) Création des tf.data.Dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices(X_train)
        .shuffle(len(X_train))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    val_ds = (
        tf.data.Dataset.from_tensor_slices(X_val)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return train_ds, val_ds, normalizer, kept_columns, best_threshold
