import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def solution(X=None, y_true=None, y_pred=None, **kwargs):

    # Recuperar datos si vienen dentro de un diccionario
    if isinstance(X, dict):
        y_true = X.get("y_true", y_true)
        y_pred = X.get("y_pred", y_pred)

    # Recuperar desde kwargs
    if y_true is None:
        y_true = kwargs.get("y_true")

    if y_pred is None:
        y_pred = kwargs.get("y_pred")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
    }

