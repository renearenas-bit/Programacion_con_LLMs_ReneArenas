import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segmentar_rutas(X: np.ndarray, random_state: int = 42) -> dict:
    """
    Determina el número óptimo de clusters para las rutas utilizando KMeans y Silhouette Score.
    Retorna el mejor K, su score, las etiquetas y el resumen estadístico.
    """
    # 1. Escalar los datos con StandardScaler (sobre todo X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mejor_k = None
    mejor_score = -np.inf
    mejor_etiquetas = None

    # 2. Probar k en el rango [2, 8] inclusive
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        etiquetas = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, etiquetas)
        
        # En caso de empate, se queda con el primero (el k más pequeño)
        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            mejor_etiquetas = etiquetas

    # 3. Construir resumen estadístico con el X original sin escalar
    df_resumen = pd.DataFrame(X)
    df_resumen.columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_resumen["cluster"] = mejor_etiquetas
    
    # Agrupamos por cluster y calculamos la media
    resumen = df_resumen.groupby("cluster").mean()
    resumen.index.name = None

    # 4. Retornar el diccionario estructurado exactamente como se pide
    return {
        "mejor_k": mejor_k,
        "mejor_score": float(mejor_score),
        "etiquetas": mejor_etiquetas,
        "resumen": resumen
    }
