import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generar_caso_de_uso_reducir_y_evaluar_pca():
    np.random.seed(0)

    n = 150
    df = pd.DataFrame({
        'temperatura': np.random.normal(50, 10, n),
        'presion': np.random.normal(30, 5, n),
        'vibracion': np.random.normal(5, 1, n),
        'flujo': np.random.normal(100, 20, n)
    })

    input_data = {'df': df.copy(), 'n_componentes': 2}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X_scaled)

    varianza = float(np.sum(pca.explained_variance_ratio_))

    return input_data, (X_reducido, varianza)
