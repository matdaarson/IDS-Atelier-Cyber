from data_utils import get_datasets
from models_v2 import build_lstm_autoencoder, build_cnn_lstm_autoencoder, build_transformer_autoencoder, build_bilstm_attention_autoencoder, build_v9_autoencoder
from train import train_model
import tensorflow as tf
import joblib




print("GPU dispo :", tf.config.list_physical_devices('GPU'))
print()
results = {}

train_ds, val_ds, input_shape, label_encoders = get_datasets(batch_size=128,directory_path='data/divided')

joblib.dump(label_encoders, "label_encoders.pkl")

models = {
    "build_lstm_autoencoder": build_lstm_autoencoder(input_shape[0]),
    "build_cnn_lstm_autoencoder": build_cnn_lstm_autoencoder(input_shape[0]),
    "build_transformer_autoencoder": build_transformer_autoencoder(input_shape[0]),
    "build_bilstm_attention_autoencoder": build_bilstm_attention_autoencoder(input_shape[0]),
    "build_v9_autoencoder": build_v9_autoencoder(input_shape[0])
}


for name, model in models.items():
    print(f"\n🚀 Entraînement du modèle: {name}")
    best_val_loss = train_model(model, train_ds, val_ds, model_name=name)
    param_count = model.count_params()
    model_size_mb = param_count * 4 / (1024 ** 2)

    results[name] = {
        "val_loss": best_val_loss,   # ✅ erreur de reconstruction minimale
        "params": param_count,
        "size_mb": model_size_mb
    }

print("\n📊 Résumé des performances des modèles :")
print("-" * 70)
print(f"{'Modèle':<20} | {'val_loss (%)':<15} | {'Params':<12} | {'Taille (MB)'}")
print("-" * 70)

for model_name, infos in results.items():
    print(f"{model_name:<20} | {infos['val_loss'] * 100:<15.4f} | {infos['params']:<12,} | {infos['size_mb']:.2f}")

print("-" * 70)

