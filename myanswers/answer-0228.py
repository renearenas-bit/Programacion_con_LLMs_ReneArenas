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
    Solución definitiva para el caso de uso 0228.
    Argumentos estrictamente posicionales para evitar errores de inyección.
    """
    pre = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42, n_estimators=150),
        "SVC": SVC(probability=True, random_state=42)
    }

    rows = []
    for nombre, mdl in modelos.items():
        pipe = Pipeline([("pre", pre), ("mdl", mdl)])
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={"acc": "accuracy", "f1": "f1", "auc": "roc_auc"},
            n_jobs=-1
        )
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

    return pd.DataFrame(rows).sort_values("score_global", ascending=False).reset_index(drop=True)
