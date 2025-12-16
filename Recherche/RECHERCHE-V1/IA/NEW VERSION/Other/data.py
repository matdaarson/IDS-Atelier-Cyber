import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Chargement du CSV
df = pd.read_csv("part_1.csv")

# -----------------------------
# 1) Suppression colonnes >30% NaN
# -----------------------------
threshold = 0.50
cols_to_drop = [col for col in df.columns if df[col].isna().mean() > threshold]
df_clean = df.drop(columns=cols_to_drop)

print("Colonnes supprimées (>30% NaN) :", cols_to_drop)
print("Colonnes restantes :", df_clean.columns.tolist())

# -----------------------------
# 2) Séparation colonnes numériques / catégorielles
# -----------------------------
num_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df_clean.select_dtypes(include=["object"]).columns

# -----------------------------
# 3) Pipeline de vectorisation
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Application du préprocessing
vectorized_data = preprocessor.fit_transform(df_clean)

# -----------------------------
# 4) Impression du résultat
# -----------------------------
print("\nShape du résultat vectorisé :", vectorized_data.shape)
print("\nAperçu (5 premières lignes) :\n")
print(vectorized_data[:5].toarray() if hasattr(vectorized_data, "toarray") else vectorized_data[:5])



