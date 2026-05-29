import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.utils.class_weight import compute_class_weight

def clasificar_congestion(df, target_col, n_splits=5):

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    clases = np.unique(y)

    pesos = compute_class_weight(
        class_weight="balanced",
        classes=clases,
        y=y
    )

    pesos_dict = {
        int(clases[i]): float(pesos[i])
        for i in range(len(clases))
    }

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

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # imputacion
        imputer = SimpleImputer(strategy="mean")

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # escalado
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # pesos por muestra
        pesos_muestra = np.where(
            y_train == 1,
            pesos_dict[1],
            pesos_dict[0]
        )

        modelo = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )

        modelo.fit(
            X_train,
            y_train,
            sample_weight=pesos_muestra
        )

        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]

        precisiones.append(
            precision_score(y_test, pred, zero_division=0)
        )

        recalls.append(
            recall_score(y_test, pred, zero_division=0)
        )

        f1s.append(
            f1_score(y_test, pred, zero_division=0)
        )

        aucs.append(
            roc_auc_score(y_test, proba)
        )

    return {
        "precision_media": float(np.mean(precisiones)),
        "recall_medio": float(np.mean(recalls)),
        "f1_medio": float(np.mean(f1s)),
        "roc_auc_medio": float(np.mean(aucs)),
        "pesos_clase": pesos_dict
    }
