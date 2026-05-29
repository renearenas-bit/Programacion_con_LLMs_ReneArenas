import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def pipeline_pca_ridge(X, **kwargs):
    """
    Recibe la matriz X del caso de uso, aplica StandardScaler y KMeans,
    y retorna el DataFrame con la estructura exacta exigida.
    """
    X_arr = np.array(X)
    df_base = pd.DataFrame(X_arr)
    
    # Escalado y entrenamiento estándar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    
    # Nombres de columnas según especificación de la rúbrica
    df_base.columns = [f"feature_{i}" for i in range(df_base.shape[1])]
    df_base["cluster"] = labels
    df_base.index.name = None
    
    return df_base
