import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

def clasificar_congestion(
    df=None,
    target_col=None,
    n_splits=5,
    **kwargs
):

    if df is None:
        return {
            "precision_media": 0.0,
            "recall_medio": 0.0,
            "f1_medio": 0.0,
            "roc_auc_medio": 0.0,
            "pesos_clase": {0: 1.0, 1: 1.0}
        }

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    clases = np.unique(y)

    pesos = compute_class_weight(
        class_weight='balanced',
        classes=clases,
        y=y
    )

    peso_map = dict(zip(clases.astype(int), pesos))

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    precisiones = []
    recalls = []
    f1s = []
    aucs = []

    for train_idx, test_idx in skf.split(X, y):

        X_tr = X[train_idx]
        X_te = X[test_idx]

        y_tr = y[train_idx]
        y_te = y[test_idx]

        imputer = SimpleImputer(strategy='mean')

        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        scaler = StandardScaler()

        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        pesos_muestra = np.where(
            y_tr == 1,
            peso_map[1],
            peso_map[0]
        )

        modelo = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )

        modelo.fit(
            X_tr,
            y_tr,
            sample_weight=pesos_muestra
        )

        y_pred = modelo.predict(X_te)

        y_prob = modelo.predict_proba(X_te)[:, 1]

        precisiones.append(
            precision_score(y_te, y_pred, zero_division=0)
        )

        recalls.append(
            recall_score(y_te, y_pred, zero_division=0)
        )

        f1s.append(
            f1_score(y_te, y_pred, zero_division=0)
        )

        aucs.append(
            roc_auc_score(y_te, y_prob)
        )

    return {
        "precision_media": float(np.mean(precisiones)),
        "recall_medio": float(np.mean(recalls)),
        "f1_medio": float(np.mean(f1s)),
        "roc_auc_medio": float(np.mean(aucs)),
        "pesos_clase": {
            int(k): float(v)
            for k, v in peso_map.items()
        }
    }
      
