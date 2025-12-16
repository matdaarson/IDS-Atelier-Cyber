import numpy as np
import joblib
import tensorflow as tf

from data import get_autoencoder_dataset


def load_models(model_dir="models"):
    """
    Charge l'autoencodeur, l'encodeur et le scaler.
    """
    autoencoder = tf.keras.models.load_model(f"{model_dir}/autoencoder_final.keras")
    encoder = tf.keras.models.load_model(f"{model_dir}/encoder_final.keras")
    decoder = tf.keras.models.load_model(f"{model_dir}/decoder_final.keras")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")

    print("[INFO] Models and scaler successfully loaded.")
    return autoencoder, encoder, decoder, scaler


def test_autoencoder(csv_path="part_1.csv", model_dir="models"):
    """
    Utilise l'autoencodeur pour tester une prédiction
    et calcule l’erreur de reconstruction.
    """
    # 1) Charger dataset (même préprocessing que lors de l’entraînement)
    X, _ = get_autoencoder_dataset(csv_path)
    print(f"[INFO] Dataset loaded with shape {X.shape}")

    # 2) Charger les modèles
    autoencoder, encoder, decoder, scaler = load_models(model_dir)

    # 3) Sélection d’un échantillon au hasard
    idx = np.random.randint(0, X.shape[0])
    sample = X[idx:idx+1]  # shape (1, n_features)

    print(f"[INFO] Selected sample index: {idx}")

    # 4) Reconstruction
    reconstructed = autoencoder.predict(sample)

    # 5) Calcul erreur
    mse = np.mean(np.square(sample - reconstructed))
    mae = np.mean(np.abs(sample - reconstructed))

    # 6) Affichage resultats
    print("\n===== TEST RESULT =====")
    print("Original vector:")
    print(sample)

    print("\nReconstructed vector:")
    print(reconstructed)

    print("\nReconstruction error:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    return {
        "sample_index": idx,
        "mse": mse,
        "mae": mae,
        "original": sample,
        "reconstructed": reconstructed
    }


if __name__ == "__main__":
    test_autoencoder("part_69.csv")
