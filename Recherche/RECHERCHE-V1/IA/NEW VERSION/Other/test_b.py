import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models

CSV_TEST_PATH = "part_22.csv"
MODEL_BEST_PATH = "models/autoencoder_best.keras"
MODEL_FINAL_PATH = "models/autoencoder_final.keras"

# ============================================================
# 1. LOAD TEST CSV
# ============================================================
df = pd.read_csv(CSV_TEST_PATH)

# Remplissage simple
df = df.fillna("")

print("Colonnes du CSV test :", df.columns.tolist())

# ============================================================
# 2. LOAD PREPROCESSING MODEL TRAINED
# ============================================================
preprocess_model = models.load_model("models/preprocessing.keras")

# Récupération compatible Keras 3
expected_inputs = [inp.name.split(":")[0] for inp in preprocess_model.inputs]

print("Inputs attendus par le modèle :", expected_inputs)
# ============================================================
# 3. ALIGNEMENT AUTOMATIQUE DES COLONNES
# ============================================================
# Toute colonne manquante est ajoutée
for col in expected_inputs:
    if col not in df.columns:
        print(f"[INFO] Ajout colonne manquante : {col}")
        df[col] = ""

# Les colonnes doivent apparaître dans EXACTEMENT le même ordre
df = df[expected_inputs]

# ============================================================
# 4. CONVERSION EN DICT POUR LE PREPROCESSING
# ============================================================
def df_to_dict(df, expected_inputs):
    out = {}
    for col in expected_inputs:
        # Détermination auto numérique / string
        if df[col].dtype in [np.float64, np.int64]:
            out[col] = df[col].values.astype("float32")
        else:
            out[col] = df[col].astype(str).values.reshape(-1, 1)
    return out

X_dict = df_to_dict(df, expected_inputs)

# Passage dans le preprocessing
X = preprocess_model(X_dict).numpy()

print("Shape X préprocessé =", X.shape)

# ============================================================
# 5. LOAD AUTOENCODER
# ============================================================
if os.path.exists(MODEL_BEST_PATH):
    autoencoder = models.load_model(MODEL_BEST_PATH)
else:
    autoencoder = models.load_model(MODEL_FINAL_PATH)

# ============================================================
# 6. ANOMALY DETECTION
# ============================================================
reco = autoencoder.predict(X, verbose=0)
mse = np.mean((X - reco) ** 2, axis=1)

threshold = mse.mean() + 3 * mse.std()
df["reconstruction_error"] = mse
df["anomaly"] = (mse > threshold).astype(int)

df.to_csv("results_test.csv", index=False)

print("✔ results_test.csv généré.")
print("Seuil =", threshold)
print("Anomalies détectées =", df['anomaly'].sum())
1700