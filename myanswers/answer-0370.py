import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def evaluar_umbrales_decision(**kwargs):
    """
    Solución exacta para el caso de uso 0370.
    Entrena una regresión logística, extrae las probabilidades de la clase positiva
    y calcula el F1-Score para 9 umbrales distribuidos uniformemente entre 0.1 y 0.9.
    """
    # 1. Extracción directa y segura de los parámetros
    df = kwargs.get('df')
    target_col = kwargs.get('target_col')

    # Fallback de contingencia si cambian las estructuras de entrada de kwargs
    if df is None:
        for val in kwargs.values():
            if isinstance(val, pd.DataFrame):
                df = val
                break
    if target_col is None:
        target_col = "target"

    # 2. Separar características (X) y variable objetivo (y) usando las columnas originales
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # 3. Réplica idéntica del entrenamiento del modelo del compañero
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(X, y)

    # 4. Extraer probabilidades de la clase positiva (1) sobre todo el dataset
    y_proba = modelo.predict_proba(X)[:, 1]

    # 5. Configurar exactamente los mismos 9 umbrales uniformes
    umbrales = np.linspace(0.1, 0.9, 9)
    f1_scores = []

    # 6. Evaluar el F1-Score en bucle emulando su lógica y manejo de excepciones
    for umbral in umbrales:
        y_pred = (y_proba >= umbral).astype(int)
        
        # Evitar errores/warnings si las predicciones se sesgan a una única clase
        if len(np.unique(y_pred)) > 1:
            f1 = f1_score(y, y_pred)
        else:
            f1 = 0.0
            
        f1_scores.append(f1)

    # 7. Retornar el array NumPy resultante
    return np.array(f1_scores, dtype=np.float64)
