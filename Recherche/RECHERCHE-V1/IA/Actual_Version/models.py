import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Reshape, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten,
    BatchNormalization, Activation, LayerNormalization, GlobalAveragePooling1D,
    MultiHeadAttention, Add, RepeatVector, TimeDistributed
)
from tensorflow.keras.regularizers import l2


# -------------------------------------------------------------------------
# 🔹 1. AUTOENCODEUR LSTM
# -------------------------------------------------------------------------
def build_lstm_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # Encodeur
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)
    latent = Dense(32, activation='relu')(x)

    # Décodeur
    x = RepeatVector(input_dim)(latent)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    model = Model(inputs, outputs, name="LSTM_Autoencoder")
    return model


# -------------------------------------------------------------------------
# 🔹 2. AUTOENCODEUR CNN + LSTM
# -------------------------------------------------------------------------
def build_cnn_lstm_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # Encodeur
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    latent = Dense(32, activation='relu')(x)

    # Décodeur
    x = RepeatVector(input_dim)(latent)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    model = Model(inputs, outputs, name="CNN_LSTM_Autoencoder")
    return model


# -------------------------------------------------------------------------
# 🔹 3. AUTOENCODEUR TRANSFORMER
# -------------------------------------------------------------------------
def build_transformer_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # Position embeddings simulés
    pos_embedding = Dense(64, activation='relu')(x)

    # Encodeur Transformer
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(pos_embedding, pos_embedding)
    attn = Add()([pos_embedding, attn])
    x = LayerNormalization()(attn)
    x = Dense(64, activation='relu')(x)
    latent = GlobalAveragePooling1D()(x)

    # Décodeur simple
    x = RepeatVector(input_dim)(latent)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    model = Model(inputs, outputs, name="Transformer_Autoencoder")
    return model


# -------------------------------------------------------------------------
# 🔹 4. AUTOENCODEUR BiLSTM + ATTENTION
# -------------------------------------------------------------------------
def build_bilstm_attention_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # Encodeur
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    attn = Dense(1, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attn])
    x = GlobalAveragePooling1D()(x)
    latent = Dense(32, activation='relu')(x)

    # Décodeur
    x = RepeatVector(input_dim)(latent)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    model = Model(inputs, outputs, name="BiLSTM_Attention_Autoencoder")
    return model


# -------------------------------------------------------------------------
# 🔹 5. AUTOENCODEUR CNN + LSTM v9 (plus profond)
# -------------------------------------------------------------------------
def build_v9_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # Encodeur
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.2))(x)
    latent = Dense(64, activation='relu')(x)

    # Décodeur
    x = RepeatVector(input_dim)(latent)
    x = LSTM(128, return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    model = Model(inputs, outputs, name="V9_Autoencoder")
    return model
