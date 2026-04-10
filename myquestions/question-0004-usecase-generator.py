import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def generar_caso_de_uso_evaluar_modelo_por_umbrales():
    np.random.seed(0)

    n = 200
    X = pd.DataFrame(np.random.randn(n, 3), columns=['f1', 'f2', 'f3'])
    y = pd.Series((X['f1'] + X['f2'] > 0).astype(int))

    umbrales = [0.3, 0.5, 0.7]

    input_data = {'X': X, 'y': y, 'umbrales': umbrales}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression().fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    scores = [f1_score(y_test, (probs >= u).astype(int)) for u in umbrales]

    return input_data, np.array(scores, dtype=float)  # 🔥 robusto
