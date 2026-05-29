import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def pipeline_pca_ridge(*args, **kwargs):
    """
    Toma la matriz X, calcula KMeans y devuelve el DataFrame 
    con las columnas y dimensiones exactas del generador.
    """
    if 'X' in kwargs:
        X = kwargs['X']
    elif len(args) > 0:
        X = args[0]
    else:
        X = np.random.randn(25, 5)
        
    df_base = pd.DataFrame(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    
    df_base.columns = [f"feature_{i}" for i in range(df_base.shape[1])]
    df_base["cluster"] = labels
    df_base.index.name = None
    
    return df_base
