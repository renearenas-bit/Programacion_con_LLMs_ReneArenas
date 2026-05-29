import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segmentar_rutas(X, random_state=42, **kwargs):

    X = np.array(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mejor_k = 2
    mejor_score = -1
    mejores_labels = None

    for k in range(2, 9):

        modelo = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_state
        )

        labels = modelo.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            mejores_labels = labels

    df = pd.DataFrame(X)
    df.columns = [f"feature_{i}" for i in range(X.shape[1])]

    df["cluster"] = mejores_labels

    resumen = df.groupby("cluster").mean()

    return {
        "mejor_k": int(mejor_k),
        "mejor_score": float(mejor_score),
        "etiquetas": mejores_labels,
        "resumen": resumen
    }
