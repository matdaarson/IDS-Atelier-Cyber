import pandas as pd
import numpy as np
import tensorflow as tf
import json
import argparse
import os


# -----------------------------------------------------
# BUILD MODEL
# -----------------------------------------------------
def build_model(num_input_dim, cat_cardinalities, embed_dim=8, hidden_dims=[128, 64]):
    
    inputs = []
    encoded = []

    # --- Numeric branch ---
    if num_input_dim > 0:
        num_inp = tf.keras.Input(shape=(num_input_dim,), name="numeric_input")
        norm_layer = tf.keras.layers.Normalization()
        norm = norm_layer(num_inp)
        inputs.append(num_inp)
        encoded.append(norm)

    # --- Categorical embeddings ---
    for i, card in enumerate(cat_cardinalities):
        inp = tf.keras.Input(shape=(1,), name=f"cat_{i}")
        emb = tf.keras.layers.Embedding(input_dim=card, output_dim=embed_dim)(inp)
        emb = tf.keras.layers.Flatten()(emb)
        inputs.append(inp)
        encoded.append(emb)

    # Concatenate numeric + embeddings
    x = tf.keras.layers.Concatenate()(encoded)

    # Encoder
    for h in hidden_dims:
        x = tf.keras.layers.Dense(h, activation="relu")(x)

    latent = tf.keras.layers.Dense(32, activation="relu", name="latent")(x)

    # Decoder
    x = latent
    for h in reversed(hidden_dims):
        x = tf.keras.layers.Dense(h, activation="relu")(x)

    # Output numeric
    outputs = []
    if num_input_dim > 0:
        out_num = tf.keras.layers.Dense(num_input_dim, name="rec_num")(x)
        outputs.append(out_num)

    # Output categorical (softmax)
    for i, card in enumerate(cat_cardinalities):
        out = tf.keras.layers.Dense(card, activation="softmax", name=f"rec_cat_{i}")(x)
        outputs.append(out)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss=["mse"] + ["sparse_categorical_crossentropy"] * len(cat_cardinalities)
    )

    return model


# -----------------------------------------------------
# TRAIN SCRIPT
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", default="ae_embed_artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # -----------------------------------------------------
    # DROP too empty columns
    # -----------------------------------------------------
    missing_ratio = df.isna().mean()
    keep_cols = missing_ratio[missing_ratio < 0.7].index.tolist()
    df = df[keep_cols]

    # -----------------------------------------------------
    # SPLIT numeric / categorical
    # -----------------------------------------------------
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # -----------------------------------------------------
    # CLEAN + MAP categorical to integer indices
    # -----------------------------------------------------
    cat_maps = {}

    for c in categorical_cols:
        df[c] = df[c].astype(str).replace(["", " ", "nan", "None", "NaN"], "UNK")

        uniques = df[c].unique().tolist()

        # force UNK
        if "UNK" not in uniques:
            uniques.append("UNK")

        # mapping
        mapping = {v: i for i, v in enumerate(uniques)}
        cat_maps[c] = mapping

        # encode with fallback to UNK index 1
        df[c] = df[c].map(lambda v: mapping.get(v, 1))

    # Compute cardinalities MIN 2 (fix NAN from softmax(1))
    cat_cardinalities = [max(2, df[c].max() + 1) for c in categorical_cols]

    # -----------------------------------------------------
    # BUILD MODEL INPUTS
    # -----------------------------------------------------
    X_num = df[numeric_cols].fillna(df[numeric_cols].median()).values.astype("float32") \
            if numeric_cols else None

    X_cat = [df[c].values.astype("int32") for c in categorical_cols]

    # Output identical to input
    inputs = []
    outputs = []
    if X_num is not None:
        inputs.append(X_num)
        outputs.append(X_num)
    for x in X_cat:
        inputs.append(x)
        outputs.append(x)

    # -----------------------------------------------------
    # BUILD + TRAIN MODEL
    # -----------------------------------------------------
    model = build_model(
        num_input_dim=X_num.shape[1] if X_num is not None else 0,
        cat_cardinalities=cat_cardinalities,
    )

    model.fit(
        inputs, outputs,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        shuffle=True
    )

    # -----------------------------------------------------
    # SAVE ARTIFACTS
    # -----------------------------------------------------
    model.save(os.path.join(args.output_dir, "ae_model.keras"))

    json.dump({
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cat_maps": cat_maps,
        "keep_cols": keep_cols
    }, open(os.path.join(args.output_dir, "meta.json"), "w"), indent=2)

    print("✔ Training completed — Saved to", args.output_dir)


if __name__ == "__main__":
    main()
