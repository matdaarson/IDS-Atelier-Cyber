import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------------------
# 🧱 Utilitaire : conversion d’un tf.data.Dataset → NumPy
# ---------------------------------------------------------------------
def dataset_to_numpy(dataset, limit=None):
    """
    Convertit un tf.data.Dataset (de tuples (X, X)) en tableau NumPy.
    Optionnellement, limite le nombre d’échantillons pour éviter une surcharge mémoire.
    """
    X_list = []
    for i, (x, _) in enumerate(dataset):
        X_list.append(x.numpy())
        if limit and i >= limit:
            break
    X_all = np.concatenate(X_list, axis=0)
    print(f"✅ Dataset converti en NumPy : {X_all.shape}")
    return X_all


# ---------------------------------------------------------------------
# 🧩 Fonction principale : Clustering avec DBSCAN
# ---------------------------------------------------------------------
def cluster_with_dbscan(model_path,
                        dataset_path=None,
                        eps=0.7,
                        min_samples=10,
                        layer_name="dense",
                        X_input=None):
    """
    Applique DBSCAN sur les embeddings latents d’un autoencodeur.
    Peut recevoir :
    - un dataset CSV (dataset_path)
    - ou directement un tableau NumPy (X_input)
    """

    # --- Chargement du modèle ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modèle introuvable : {model_path}")
    print(f"🧠 Chargement du modèle : {model_path}")
    model = load_model(model_path)

    # --- Chargement des données ---
    if X_input is not None:
        print(f"✅ Données reçues directement depuis le Dataset TensorFlow : {X_input.shape}")
        X = X_input
        df = pd.DataFrame(X)
    elif dataset_path and os.path.exists(dataset_path):
        print(f"📂 Chargement du dataset CSV : {dataset_path}")
        df = pd.read_csv(dataset_path)
        df = df.fillna(0)
        X = df.values.astype(np.float32)
    else:
        raise ValueError("❌ Aucune donnée fournie : spécifie dataset_path ou X_input")

    # --- Extraction de la couche latente ---
    try:
        encoder = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        embeddings = []
        batch_size = 512
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            emb = encoder.predict(batch)
            embeddings.append(emb)
        embeddings = np.concatenate(embeddings, axis=0)
        print(f"✅ Espace latent extrait depuis la couche '{layer_name}' : {embeddings.shape}")
    except Exception as e:
        print(f"⚠️ Erreur : {e}")
        print("➡️ Utilisation des erreurs de reconstruction à la place.")
        X_pred = model.predict(X)
        embeddings = np.mean(np.square(X - X_pred), axis=1).reshape(-1, 1)

    # --- Standardisation ---
    X_scaled = StandardScaler().fit_transform(embeddings)

    # --- Clustering DBSCAN ---
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"📊 Clustering terminé : {n_clusters} clusters détectés.")
    print(f"🚨 Anomalies détectées : {(clusters == -1).sum()} sur {len(clusters)} points.")

    # --- Visualisation avec PCA ---
    if embeddings.shape[1] > 1:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(embeddings)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=latent_2d[:, 0],
            y=latent_2d[:, 1],
            hue=clusters,
            palette='tab10',
            s=60,
            alpha=0.8,
            edgecolor='k'
        )
        plt.title("📈 DBSCAN sur l'espace latent de l'autoencodeur")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/dbscan_clusters.png")
        plt.close()
        print("✅ Graphique sauvegardé dans plots/dbscan_clusters.png")

    # --- Sauvegarde des résultats ---
    os.makedirs("CSV", exist_ok=True)
    result_df = df.copy()
    result_df["cluster"] = clusters
    result_df.to_csv("CSV/dbscan_results.csv", index=False)
    print("✅ Résultats sauvegardés dans CSV/dbscan_results.csv")

    anomalies = result_df[result_df["cluster"] == -1]
    return result_df, anomalies
