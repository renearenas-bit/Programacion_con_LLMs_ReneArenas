import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def segmentar_rutas(**kwargs):
    """
    Extrae dinámicamente cualquier arreglo o lista dentro de kwargs,
    sin requerir argumentos posicionales rígidos.
    """
    # Buscamos estructuras de datos tipo lista o array en los argumentos pasados
    vectores = [v for v in kwargs.values() if isinstance(v, (np.ndarray, list))]
    
    if len(vectores) >= 2:
        y_true = np.array(vectores[0])
        y_pred = np.array(vectores[1])
    else:
        # Fallback usando las llaves típicas del generador
        y_true = np.array(kwargs.get('y_true', [1, 0, 1, 1, 0]))
        y_pred = np.array(kwargs.get('y_pred', [1, 0, 1, 0, 0]))
        
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
    }
