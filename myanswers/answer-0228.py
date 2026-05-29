import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def ranking_modelos_cv(X, y, cv=5, **kwargs):
    """
    Evalúa tres modelos de clasificación utilizando validación cruzada.
    Retorna un DataFrame ordenado por el score_global de forma descendente.
    """
    # 1. Definición del preprocesamiento de datos
    preprocesamiento = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler())
    ])

    # 2. Diccionario con los tres clasificadores base solicitados
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42, n_estimators=150),
        "SVC": SVC(probability=True, random_state=42)
    }

    resultados = []
    
    # 3. Iteración y evaluación mediante validación cruzada
    for nombre, modelo in modelos.items():
        pipe_completo = Pipeline([("pre", preprocesamiento), ("model", modelo)])
        
        scores = cross_validate(
            pipe_completo, X, y, cv=cv,
            scoring={"acc": "accuracy", "f1": "f1", "auc": "roc_auc"},
            n_jobs=-1
        )
        
        # Extracción y cálculo de medias métricas
        acc_mean = float(np.mean(scores["test_acc"]))
        f1_mean = float(np.mean(scores["test_f1"]))
        auc_mean = float(np.mean(scores["test_auc"]))
        score_global = float(np.mean([acc_mean, f1_mean, auc_mean]))

        resultados.append({
            "modelo": nombre,
            "acc_mean": acc_mean,
            "f1_mean": f1_mean,
            "auc_mean": auc_mean,
            "score_global": score_global
        })

    # 4. Construcción del DataFrame y ordenamiento jerárquico
    df_ranking = pd.DataFrame(resultados)
    df_ranking = df_ranking.sort_values("score_global", ascending=False).reset_index(drop=True)
    
    return df_ranking
