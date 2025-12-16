import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--artifacts_dir", default="ae_embed_artifacts")
    parser.add_argument("--out", default="test_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Load metadata
    meta = json.load(open(os.path.join(args.artifacts_dir, "meta.json")))
    numeric_cols = meta["numeric_cols"]
    categorical_cols = meta["categorical_cols"]
    cat_maps = meta["cat_maps"]
    keep_cols = meta["keep_cols"]

    # Align test columns
    for col in keep_cols:
        if col not in df.columns:
            df[col] = "UNK"

    df = df[keep_cols]

    # ---------------------
    # Process numeric
    # ---------------------
    X_num = df[numeric_cols].fillna(df[numeric_cols].median()).values.astype("float32") \
            if numeric_cols else None

    # ---------------------
    # Process categorical
    # ---------------------
    X_cat = []
    for c in categorical_cols:
        df[c] = df[c].astype(str).replace(["", " ", "nan", "None", "NaN"], "UNK")
        mapping = cat_maps[c]
        df[c] = df[c].map(lambda v: mapping.get(v, 1))  # 1 = UNK
        X_cat.append(df[c].values.astype("int32"))

    # Build model inputs
    inputs = []
    if X_num is not None:
        inputs.append(X_num)
    inputs.extend(X_cat)

    # Load model
    model = tf.keras.models.load_model(os.path.join(args.artifacts_dir, "ae_model.keras"))

    preds = model.predict(inputs, verbose=0)

    # Reconstruction error (numeric only)
    if X_num is not None:
        rec_num = preds[0]
        mse = np.mean((X_num - rec_num) ** 2, axis=1)
        df["reconstruction_error"] = mse
    else:
        df["reconstruction_error"] = 0.0

    df.to_csv(os.path.join(args.artifacts_dir, args.out), index=False)
    print("✔ Results saved to:", args.out)


if __name__ == "__main__":
    main()
