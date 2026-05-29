import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def pipeline_pca_ridge(
    df=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    alpha=1.0,
    **kwargs
):

    # Caso cuando llega un DataFrame
    if df is not None:

        df_num = df.select_dtypes(include=[np.number])

        X = df_num.iloc[:, :-1]
        y = df_num.iloc[:, -1]

        X_train = X
        X_test = X
        y_train = y
        y_test = y

    # Imputación
    imputer = SimpleImputer(strategy="mean")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Escalado
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Modelo
    modelo = Ridge(alpha=alpha)

    modelo.fit(X_train_pca, y_train)

    predicciones = modelo.predict(X_test_pca)

    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    r2 = r2_score(y_test, predicciones)

    return {
        "n_componentes": int(pca.n_components_),
        "rmse": float(rmse),
        "r2": float(r2),
        "predicciones": predicciones
    }
