import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def pipeline_pca_ridge(
    X_train,
    y_train,
    X_test,
    y_test,
    alpha=1.0
):

    imputer = SimpleImputer(strategy='mean')

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.95)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    modelo = Ridge(alpha=alpha)

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predicciones)
    )

    r2 = r2_score(y_test, predicciones)

    return {
        "n_componentes": int(pca.n_components_),
        "rmse": float(rmse),
        "r2": float(r2),
        "predicciones": predicciones
    }
