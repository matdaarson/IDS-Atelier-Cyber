import math
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Model

# ===============================================================
# Utility layers & blocks
# ===============================================================

class MatchTimeDimLayer(L.Layer):
    """
    Align time dimensions (axis=1) by right-padding the shortest.
    Allows skip-connections even if pooling produced uneven lengths.
    """
    def call(self, inputs):
        x, skip = inputs
        x_len = tf.shape(x)[1]
        s_len = tf.shape(skip)[1]

        def same():
            return x, skip

        def pad_x():
            pad = s_len - x_len
            return tf.pad(x, [[0,0],[0,pad],[0,0]]), skip

        def pad_skip():
            pad = x_len - s_len
            return x, tf.pad(skip, [[0,0],[0,pad],[0,0]])

        return tf.case(
            [
                (tf.equal(x_len, s_len), same),
                (tf.less(x_len, s_len), pad_x)
            ],
            default=pad_skip,
            exclusive=True
        )


class SqueezeExcite1D(L.Layer):
    def __init__(self, ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        c = int(input_shape[-1])
        r = max(1, int(c * self.ratio))
        self.gap = L.GlobalAveragePooling1D()
        self.fc1 = L.Dense(r, activation="relu")
        self.fc2 = L.Dense(c, activation="sigmoid")
        self.reshape = L.Reshape((1, c))
        super().build(input_shape)

    def call(self, x):
        s = self.gap(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.reshape(s)
        return x * s


def residual_conv_block(x, filters, kernel=5, dilation=1, dropout=0.1):
    h = L.Conv1D(filters, kernel, padding="same",
                 dilation_rate=dilation, kernel_initializer="he_normal")(x)
    h = L.BatchNormalization()(h)
    h = L.Activation("gelu")(h)

    h = L.Conv1D(filters, 3, padding="same", kernel_initializer="he_normal")(h)
    h = L.BatchNormalization()(h)
    h = SqueezeExcite1D()(h)

    if x.shape[-1] != filters:
        x = L.Conv1D(filters, 1, padding="same")(x)

    h = L.Add()([x, h])

    if dropout:
        h = L.Dropout(dropout)(h)

    return h


def up_block(x, filters):
    x = L.UpSampling1D(2)(x)
    x = L.Conv1D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("gelu")(x)
    return x


def transformer_encoder(x, num_heads=4, key_dim=32, mlp_mult=2, dropout=0.1):
    h = L.LayerNormalization()(x)
    h = L.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(h, h)
    x = L.Add()([x, h])

    h = L.LayerNormalization()(x)
    h = L.Dense(key_dim * mlp_mult, activation="gelu")(h)
    h = L.Dropout(dropout)(h)
    h = L.Dense(x.shape[-1])(h)

    return L.Add()([x, h])




class CropOrPad1D(L.Layer):
    def __init__(self, target_len, **kwargs):
        super().__init__(**kwargs)
        self.target_len = target_len

    def call(self, x):
        seq_len = tf.shape(x)[1]

        # Always pad enough
        pad_len = tf.maximum(0, self.target_len - seq_len)
        x = tf.pad(x, [[0,0], [0, pad_len], [0,0]])

        # Always slice to target length
        return x[:, :self.target_len, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_len, input_shape[2])





# ===============================================================
# 1) LSTM Autoencoder
# ===============================================================

def build_lstm_autoencoder(input_dim, latent_dim=64):
    inp = L.Input(shape=(input_dim,))
    x = L.Reshape((input_dim,1))(inp)

    x = L.Bidirectional(L.LSTM(128, return_sequences=True, dropout=0.15))(x)
    x = L.Bidirectional(L.LSTM(96, return_sequences=True, dropout=0.15))(x)
    x = L.LayerNormalization()(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=False))(x)

    z = L.Dense(latent_dim, activation="gelu", name="latent")(x)

    d = L.RepeatVector(input_dim)(z)
    d = L.LSTM(96, return_sequences=True)(d)
    d = L.LSTM(128, return_sequences=True)(d)
    d = L.TimeDistributed(L.Dense(1))(d)

    return Model(inp, L.Flatten()(d), name="LSTM_v2")


# ===============================================================
# 2) CNN + LSTM Autoencoder
# ===============================================================

def build_cnn_lstm_autoencoder(input_dim, latent_dim=64):

    inp = L.Input(shape=(input_dim,))
    x = L.Reshape((input_dim,1))(inp)

    # ENCODEUR

    x = residual_conv_block(x, 128, 7, 1)       
    x = L.MaxPooling1D(2, padding="same")(x)     

    x = residual_conv_block(x, 192, 5, 2)        
    x = L.MaxPooling1D(2, padding="same")(x)     

    x = residual_conv_block(x, 256, 3, 4)       

    encoded_len = x.shape[1]     # longueur finale après downsampling

    # LSTM bottleneck
    x = L.Bidirectional(L.LSTM(128, return_sequences=False, dropout=0.15))(x)
    z = L.Dense(latent_dim, activation="gelu")(x)


    # DECODEUR
  
    d = L.RepeatVector(encoded_len)(z)
    d = L.LSTM(128, return_sequences=True)(d)

    d = up_block(d, 192)
    d = up_block(d, 128)

    d = L.Conv1D(64, 5, padding="same", activation="gelu")(d)

    # No Lambda
    d = CropOrPad1D(input_dim)(d)

    d = L.Conv1D(1, 1, padding="same")(d)
    out = L.Flatten()(d)

    return Model(inp, out, name="CNN_LSTM_AE_v2")


# ===============================================================
# 3) Transformer Autoencoder
# ===============================================================

def build_transformer_autoencoder(input_dim, d_model=128, depth=4, latent_dim=64):
    inp = L.Input(shape=(input_dim,))
    x = L.Reshape((input_dim,1))(inp)
    x = L.Conv1D(d_model,1,padding="same")(x)

    for _ in range(depth):
        x = transformer_encoder(x,4,d_model//4,2)

    x = L.LayerNormalization()(x)
    x = L.GlobalAveragePooling1D()(x)
    z = L.Dense(latent_dim,activation="gelu")(x)

    d = L.RepeatVector(input_dim)(z)
    d = L.LSTM(d_model//2,return_sequences=True)(d)
    d = L.TimeDistributed(L.Dense(64,activation="gelu"))(d)
    d = L.TimeDistributed(L.Dense(1))(d)

    return Model(inp, L.Flatten()(d), name="Transformer_AE_v2")


# ===============================================================
# 4) BiLSTM + Multi-Head Attention Autoencoder
# ===============================================================

def build_bilstm_attention_autoencoder(input_dim, latent_dim=64):
    inp = L.Input(shape=(input_dim,))
    x = L.Reshape((input_dim,1))(inp)

    x = L.Bidirectional(L.LSTM(128,return_sequences=True,dropout=0.15))(x)
    x = L.Bidirectional(L.LSTM(96,return_sequences=True,dropout=0.1))(x)

    q = L.LayerNormalization()(x)
    attn = L.MultiHeadAttention(4,32,dropout=0.1)(q,q)
    x = L.Add()([x,attn])

    x = L.GlobalAveragePooling1D()(x)
    z = L.Dense(latent_dim,activation="gelu")(x)

    d = L.RepeatVector(input_dim)(z)
    d = L.LSTM(96,return_sequences=True)(d)
    d = L.LSTM(128,return_sequences=True)(d)
    d = L.TimeDistributed(L.Dense(1))(d)

    return Model(inp, L.Flatten()(d), name="BiLSTM_Attn_AE_v2")


# ===============================================================
# 5) Deep CNN + LSTM (V9) — FIXED
# ===============================================================

def build_v9_autoencoder(input_dim, latent_dim=96):
    inp = L.Input(shape=(input_dim,))
    x0 = L.Reshape((input_dim,1))(inp)

    # Encoder
    e1 = residual_conv_block(x0,128,7,1)
    p1 = L.MaxPooling1D(2,padding="same")(e1)

    e2 = residual_conv_block(p1,256,5,2)
    p2 = L.MaxPooling1D(2,padding="same")(e2)

    e3 = residual_conv_block(p2,384,3,4)
    p3 = L.MaxPooling1D(2,padding="same")(e3)

    # Bottleneck
    b = L.Bidirectional(L.LSTM(192,return_sequences=False,dropout=0.15))(p3)
    z = L.Dense(latent_dim,activation="gelu")(b)

    # Start decoding at a **static** length → ceil(input_dim / 8)
    L3 = (input_dim+7)//8
    d = L.RepeatVector(L3)(z)
    d = L.LSTM(192,return_sequences=True)(d)

    # Up 1
    d = up_block(d,384)
    d,e3f = MatchTimeDimLayer()([d,e3])
    d = L.Concatenate()([d,e3f])
    d = residual_conv_block(d,384,3)

    # Up 2
    d = up_block(d,256)
    d,e2f = MatchTimeDimLayer()([d,e2])
    d = L.Concatenate()([d,e2f])
    d = residual_conv_block(d,256,5)

    # Up 3
    d = up_block(d,128)
    d,e1f = MatchTimeDimLayer()([d,e1])
    d = L.Concatenate()([d,e1f])
    d = residual_conv_block(d,128,7)

    d = L.Conv1D(1,1,padding="same")(d)
    d = CropOrPad1D(input_dim)(d)

    return Model(inp, L.Flatten()(d), name="V9_AE_v2_FIXED")


# ===============================================================
# Optimizer & compile
# ===============================================================

def compile_autoencoder(
    model,
    lr=3e-4,
    weight_decay=1e-4,
    loss="huber",
    clipnorm=1.0
):
    try:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            clipnorm=clipnorm
        )
    except:
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=clipnorm
        )

    if loss == "huber":
        loss_fn = tf.keras.losses.Huber()
    elif loss == "mae":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ]
    )

    return model
