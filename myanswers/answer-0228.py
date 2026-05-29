import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def ranking_modelos_cv(X=None, y=None, **kwargs):
    """
    Solución corregida para el caso de uso 0228.
    Soporta paso de argumentos posicionales (X, y) y por kwargs simultáneamente.
    """
    # 1. Si no llegaron por posición, los buscamos defensivamente en kwargs
    if X is None:
        X = kwargs.get('X')
    if y is None:
        y = kwargs.get('y')
        
    cv = kwargs.get('cv', 5)

    # 2. Replicar la estructura de Pipeline de preprocesamiento de la pregunta
    pre = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    # 3. Modelos con los hiperparámetros idénticos requeridos
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
