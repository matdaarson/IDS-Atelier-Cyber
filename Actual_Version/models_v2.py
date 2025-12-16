import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Reshape, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten,
    BatchNormalization, Activation, LayerNormalization, GlobalAveragePooling1D,
    MultiHeadAttention, Add, RepeatVector, TimeDistributed
)
from tensorflow.keras.regularizers import l2


# -------------------------------------------------------------------------
# 🔹 1. LSTM AUTOENCODER (v2)
# -------------------------------------------------------------------------
def build_lstm_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)
    latent = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)

    x = RepeatVector(input_dim)(latent)
    x = LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    return Model(inputs, outputs, name="LSTM_Autoencoder_v2")


# -------------------------------------------------------------------------
# 🔹 2. CNN + LSTM AUTOENCODER (v2)
# -------------------------------------------------------------------------
def build_cnn_lstm_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.3))(x)
    latent = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)

    x = RepeatVector(input_dim // 4)(latent)
    x = LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    return Model(inputs, outputs, name="CNN_LSTM_Autoencoder_v2")



def build_transformer_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # projection
    x = Dense(64)(x)

    # corrected positional encoding
    pos = tf.range(start=0, limit=input_dim, delta=1)
    pos = tf.expand_dims(pos, 0)  # (1, input_dim)
    pos = tf.tile(pos, [tf.shape(x)[0], 1])  # match batch dimension
    pos = tf.expand_dims(pos, -1)  # (batch, input_dim, 1)
    pos = Dense(64)(pos)
    pos = tf.math.sin(pos)

    x = Add()([x, pos])

    # Transformer encoder blocks
    for _ in range(2):
        attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)

        ffn = Dense(128, activation='relu')(x)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)

    latent = GlobalAveragePooling1D()(x)

    x = RepeatVector(input_dim)(latent)
    x = LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    return Model(inputs, outputs, name="Transformer_Autoencoder_v2")





# -------------------------------------------------------------------------
# 🔹 4. BiLSTM + ATTENTION AUTOENCODER (v2)
# -------------------------------------------------------------------------
def build_bilstm_attention_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    latent = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    latent = Dropout(0.3)(latent)

    x = RepeatVector(input_dim)(latent)
    x = LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    return Model(inputs, outputs, name="BiLSTM_Attention_Autoencoder_v2")


# -------------------------------------------------------------------------
# 🔹 5. CNN + LSTM PROFOND (v9 amélioré)
# -------------------------------------------------------------------------
def build_v9_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(512, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.3))(x)
    latent = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)

    x = RepeatVector(input_dim // 8)(latent)
    x = LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = TimeDistributed(Dense(1))(x)

    outputs = Flatten()(x)
    return Model(inputs, outputs, name="V9_Autoencoder_v2")
