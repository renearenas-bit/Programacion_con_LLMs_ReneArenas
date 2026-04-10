import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest

def generar_caso_de_uso_preparar_datos_credito():
    np.random.seed(0)

    n = 200
    df = pd.DataFrame({
        'ingresos': np.random.normal(3000, 1000, n),
        'deuda': np.random.normal(1000, 500, n),
        'edad': np.random.randint(18, 70, n),
        'tipo_empleo': np.random.choice(['fijo', 'temporal', 'independiente'], n),
        'aprobado': np.random.choice([0, 1], n)
    })

    input_data = {'df': df.copy(), 'target_col': 'aprobado'}

    X = df.drop(columns=['aprobado'])
    y = df['aprobado'].values

    transformer = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['ingresos', 'deuda', 'edad']),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(sparse_output=False))
        ]), ['tipo_empleo'])
    ])

    X_proc = transformer.fit_transform(X)

    iso = IsolationForest(random_state=42, contamination=0.05)
    mask = iso.fit_predict(X_proc) == 1

    return input_data, (X_proc[mask], y[mask])
