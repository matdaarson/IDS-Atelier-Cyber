import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib

def get_datasets(directory_path='data/divided',
                 batch_size=128, save_encoders=True):
    """
    Prépare les données pour un autoencodeur :
    - Encodage automatique des colonnes alphanumériques
    - Remplacement des valeurs manquantes et infinies
    - Création de datasets TensorFlow (X, X)
    """

    all_dfs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, low_memory=False)
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("Aucun fichier CSV trouvé dans le dossier spécifié.")

    # Fusion de tous les CSV
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Remplacement des NaN et None
    full_df = full_df.fillna("missing")

    # Encodage des colonnes texte
    label_encoders = {}
    for col in full_df.columns:
        if full_df[col].dtype == 'object':
            le = LabelEncoder()
            full_df[col] = le.fit_transform(full_df[col].astype(str))
            label_encoders[col] = le

    # Remplacer les valeurs infinies et NaN résiduelles
    full_df = full_df.replace([np.inf, -np.inf], 0)
    full_df = full_df.fillna(0)
    print(full_df)
    # Vérification finale
    if full_df.isnull().values.any():
        print("🚨 Des valeurs manquantes subsistent :")
        print(full_df.isnull().sum()[full_df.isnull().sum() > 0])
        raise ValueError("Des valeurs None / NaN subsistent dans full_df.")
        
    # Conversion en tenseur
    X = full_df.values.astype(np.float32)
    features_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    # Création du dataset (X, X)
    dataset = tf.data.Dataset.from_tensor_slices((features_tensor, features_tensor))

    # Partition du dataset
    n = len(full_df)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    dataset = dataset.shuffle(buffer_size=n, seed=42)
    train_ds = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_ds = remaining.take(val_size)

    train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    input_shape = (full_df.shape[1],)

    # Sauvegarde optionnelle des encoders
    if save_encoders:
        joblib.dump(label_encoders, "label_encoders.pkl")

    print(f"Dataset autoencodeur prêt : {n} échantillons, input_dim={input_shape[0]}")
    return train_ds, val_ds, input_shape, label_encoders
