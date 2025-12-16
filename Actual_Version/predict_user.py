import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from data_utils import extract_features, FEATURE_NAMES

def predict_user(json_path, model_path="IA/ACTUAL_VERSION/V2/Models/V9.keras", scaler_path="IA/ACTUAL_VERSION/V2/scalers/scaler.pkl"):
    # Chargement du modèle et du scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Lecture des données utilisateur
    with open(json_path, "r") as f:
        data = json.load(f)

    features = extract_features(data, label=0)
    if not features:
        print("⚠️ Aucune donnée exploitable dans ce fichier.")
        return

    # Construction du DataFrame avec les noms de colonnes corrects
    df = pd.DataFrame(features, columns=FEATURE_NAMES)

    # Suppression de la colonne 'label' avant le scaling
    X = df.drop(columns=['label'])

    # Application du scaler de l'entraînement
    X_scaled = scaler.transform(X)

    # Prédiction
    y_pred = model.predict(X_scaled)
    classes = (y_pred >= 0.5).astype(int)
    user_class = int(np.mean(classes) >= 0.5)

    print(f"🧠 Prédictions individuelles : {classes.flatten()}")
    print(f"📊 Résultat du vote : {np.bincount(classes.flatten())}")
    print(f"👤 Utilisateur détecté : {'Utilisateur 2' if user_class == 1 else 'Utilisateur 1'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage : python predict_user.py chemin/vers/fichier.json")
    else:
        json_file = sys.argv[1]
        predict_user(json_file)