import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_curves(csv_path, model_name="Modèle"):
    """
    Affiche et sauvegarde les courbes d'apprentissage (loss et MAE)
    à partir du fichier CSV généré pendant l'entraînement.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Fichier introuvable : {csv_path}")
        return

    # Lecture du CSV
    history = pd.read_csv(csv_path)

    # Création du dossier plots
    os.makedirs("plots", exist_ok=True)

    # Vérification des colonnes disponibles
    print(f"📊 Colonnes disponibles dans {csv_path} : {list(history.columns)}")

    # ---------------------------------------------------
    # 🔹 Courbe de perte (MSE reconstruction)
    # ---------------------------------------------------
    if "loss" in history.columns and "val_loss" in history.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(history["loss"], label="Train Loss (MSE)")
        plt.plot(history["val_loss"], label="Validation Loss (MSE)")
        plt.title(f"Courbe de perte - {model_name}")
        plt.xlabel("Époques")
        plt.ylabel("Erreur de reconstruction (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_loss.png")
        plt.close()
        print(f"✅ Graphique sauvegardé : plots/{model_name}_loss.png")
    else:
        print("⚠️ Colonnes 'loss' et 'val_loss' non trouvées dans le CSV.")

    # ---------------------------------------------------
    # 🔹 Courbe d’erreur absolue moyenne (MAE)
    # ---------------------------------------------------
    if "mae" in history.columns and "val_mae" in history.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(history["mae"], label="Train MAE")
        plt.plot(history["val_mae"], label="Validation MAE")
        plt.title(f"Courbe d'erreur absolue moyenne - {model_name}")
        plt.xlabel("Époques")
        plt.ylabel("Erreur absolue moyenne (MAE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_mae.png")
        plt.close()
        print(f"✅ Graphique sauvegardé : plots/{model_name}_mae.png")
    else:
        print("⚠️ Colonnes 'mae' et 'val_mae' non trouvées dans le CSV.")

if __name__ == "__main__":
    # Exemple : adapter au modèle que tu veux visualiser
    plot_training_curves("CSV/build_v9_autoencoder_history.csv", model_name="V9_Autoencoder")
