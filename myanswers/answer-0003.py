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
    alpha: float = 1.0,
    **kwargs
) -> dict:
    """
    Pipeline robusto PCA + Ridge.
    Compatible con DataFrames que contengan columnas categóricas.
    """

    # Caso donde llega un DataFrame completo
    if df is not None and isinstance(df, pd.DataFrame):

        # Tomar solo columnas numéricas
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.shape[1] > 1:
            X = df_numeric.iloc[:, :-1].values
            y = df_numeric.iloc[:, -1].values

            X_train = X
            X_test = X

            y_train = y
            y_test = y

    # Si llegan DataFrames separados
    if X_train is not None and isinstance(X_train, pd.DataFrame):
        X_train = X_train.select_dtypes(include=[np.number]).values

    if X_test is not None and isinstance(X_test, pd.DataFrame):
        X_test = X_test.select_dtypes(include=[np.number]).values

    # Conversión segura a numpy float
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)

    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)

    # Imputación
    imputer = SimpleImputer(strategy='mean')

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Escalado
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # PCA
    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Modelo Ridge
    model = Ridge(alpha=alpha)

    model.fit(X_train_pca, y_train)

    predicciones = model.predict(X_test_pca)

    # Métricas
    rmse = float(
        np.sqrt(mean_squared_error(y_test, predicciones))
    )

    r2 = float(
        r2_score(y_test, predicciones)
    )

    return {
        "n_componentes": int(pca.n_components_),
        "rmse": rmse,
        "r2": r2,
        "predicciones": predicciones
    }
