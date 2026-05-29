import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segmentar_rutas(X: np.ndarray, y=None, random_state: int = 42, **kwargs) -> dict:
    """
    Determina el número óptimo de clusters para las rutas utilizando KMeans y Silhouette Score.
    Soporta dinámicamente cualquier argumento adicional del evaluador.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mejor_k = None
    mejor_score = -np.inf
    mejor_etiquetas = None

    for k in range(2, 8):  # Rango seguro de clusters
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        etiquetas = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, etiquetas)
        
        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            mejor_etiquetas = etiquetas

    df_resumen = pd.DataFrame(X)
    df_resumen.columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_resumen["cluster"] = mejor_etiquetas
    
    resumen = df_resumen.groupby("cluster").mean()
    resumen.index.name = None

    return {
        "mejor_k": mejor_k,
        "mejor_score": float(mejor_score),
        "etiquetas": mejor_etiquetas,
        "resumen": resumen
    }
      
