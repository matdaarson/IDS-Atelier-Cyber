import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# ------------------------------
# 1. Charger le CSV
# ------------------------------
df = pd.read_csv("part_1.csv")

# ------------------------------
# 2. Supprimer colonnes avec > 30% de valeurs manquantes
# ------------------------------
seuil = 0.3
df = df[df.columns[df.isnull().mean() < seuil]]

# ------------------------------
# 3. Séparer colonnes numériques / textuelles
# ------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
text_cols = df.select_dtypes(include=["object"]).columns.tolist()

# ------------------------------
# 4. Remplir les NA
# ------------------------------
df[numeric_cols] = df[numeric_cols].fillna(0)
df[text_cols] = df[text_cols].fillna("")

# ------------------------------
# 5. Tokeniser chaque colonne textuelle
# ------------------------------
vectorizers = {}
max_tokens = 20000
output_sequence_length = 50

for col in text_cols:
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_sequence_length
    )
    vectorizer.adapt(df[col].astype(str).values)
    vectorizers[col] = vectorizer

# Appliquer les vectorizers et stocker résultat
text_tensors = []
for col, vec in vectorizers.items():
    text_tensors.append(vec(df[col].astype(str).values))

# Fusionner toutes les colonnes textuelles tokenisées
if text_tensors:
    X_text = tf.concat(text_tensors, axis=1)
else:
    X_text = None

# ------------------------------
# 6. Convertir les colonnes numériques en tenseur
# ------------------------------
if numeric_cols:
    X_num = tf.convert_to_tensor(df[numeric_cols].values, dtype=tf.float32)
else:
    X_num = None

# ------------------------------
# 7. Fusionner texte + numérique dans un seul tenseur
# ------------------------------
if X_text is not None and X_num is not None:
    X_final = tf.concat([X_num, X_text], axis=1)
elif X_text is not None:
    X_final = X_text
else:
    X_final = X_num

print("Shape finale :", X_final.shape)

# ------------------------------
# 8. Construire le dataset pour autoencodeur (X = Y)
# ------------------------------
dataset = tf.data.Dataset.from_tensor_slices((X_final, X_final))
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

print("Dataset autoencodeur créé !")
