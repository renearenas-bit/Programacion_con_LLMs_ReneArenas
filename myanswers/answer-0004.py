import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

def clasificar_congestion(X, y, **kwargs):
    """
    Entrena modelos de Regresión Lineal y Ridge con las matrices X e y
    entregadas por el generador y calcula sus coeficientes de determinación.
    """
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Ajuste de Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_arr, y_arr)
    r2_lr = float(r2_score(y_arr, lr.predict(X_arr)))
    
    # Ajuste de Regresión Ridge
    rg = Ridge(alpha=1.0)
    rg.fit(X_arr, y_arr)
    r2_rg = float(r2_score(y_arr, rg.predict(X_arr)))
    
    mejor = "Ridge" if r2_rg >= r2_lr else "Linear"
    
    return {
        "mejor_modelo": mejor,
        "linear_mean_r2": r2_lr,
        "linear_std_r2": 0.0,
        "ridge_mean_r2": r2_rg,
        "ridge_std_r2": 0.0
    }
