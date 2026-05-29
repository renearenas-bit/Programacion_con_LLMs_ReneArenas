import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def ranking_modelos_cv(**kwargs):
    """
    Solución exacta para el caso de uso 0228.
    Evalúa y ranquea modelos de clasificación usando Cross-Validation
    con pipelines de imputación y escalamiento integrados.
    """
    # 1. Extracción directa y segura usando las llaves mapeadas por el generador
    X = kwargs.get('X')
    y = kwargs.get('y')
    cv = kwargs.get('cv', 5)

    # Fallback defensivo por si cambiaran las estructuras de entrada
    if X is None or y is None:
        for val in kwargs.values():
            if isinstance(val, (pd.DataFrame, np.ndarray)):
                arr = np.array(val)
                if arr.ndim == 2: X = val
                elif arr.ndim == 1: y = val

    # 2. Replicar exactamente la estructura de Pipeline de preprocesamiento de la pregunta
    pre = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    # 3. Modelos con los hiperparámetros idénticos (¡Ojo con n_estimators=150!)
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42, n_estimators=150),
        "SVC": SVC(probability=True, random_state=42)
    }

    rows = []
    
    # 4. Evaluación cruzada secuencial por modelo
    for nombre, mdl in modelos.items():
        pipe = Pipeline([("pre", pre), ("mdl", mdl)])
        
        # Ejecución de cross_validate con las etiquetas de métricas idénticas
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={"acc": "accuracy", "f1": "f1", "auc": "roc_auc"},
            n_jobs=-1
        )
        
        # Extracción y cálculo de promedios flotantes
        acc_mean = float(np.mean(scores["test_acc"]))
        f1_mean = float(np.mean(scores["test_f1"]))
        auc_mean = float(np.mean(scores["test_auc"]))
        score_global = float(np.mean([acc_mean, f1_mean, auc_mean]))

        rows.append({
            "modelo": nombre,
            "acc_mean": acc_mean,
            "f1_mean": f1_mean,
            "auc_mean": auc_mean,
            "score_global": score_global
        })

    # 5. Retornar el DataFrame ordenado de forma descendente y con el índice limpio
    df_resultado = pd.DataFrame(rows).sort_values("score_global", ascending=False).reset_index(drop=True)
    
    return df_resultado
