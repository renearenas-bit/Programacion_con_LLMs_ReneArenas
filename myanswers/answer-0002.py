import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def segmentar_rutas(y_true, y_pred, **kwargs):
    """
    Calcula de forma exacta las métricas de clasificación requeridas
    a partir de los vectores numéricos que inyecta el validador.
    """
    # Aseguramos conversión a arreglos de numpy
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    }
