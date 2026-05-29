import numpy as np
import pandas as pd

def pipeline_pca_ridge(df):

    # eliminar duplicados y nulos
    df = df.drop_duplicates().dropna()

    # clasificacion de riesgo
    condiciones = [
        (df["glucosa"] >= 140) | (df["presion_arterial"] >= 140),
        (df["glucosa"] >= 100) | (df["presion_arterial"] >= 120)
    ]

    valores = ["alto", "medio"]

    df["grupo_riesgo"] = np.select(
        condiciones,
        valores,
        default="bajo"
    )

    # ordenar
    df = df.sort_values("edad").reset_index(drop=True)

    return df
