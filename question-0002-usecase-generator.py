X = df.drop(columns=['aprobado'])
    y = df['aprobado'].values

    transformer = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['ingresos', 'deuda', 'edad']),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(sparse=False))  # 🔥 CORREGIDO
        ]), ['tipo_empleo'])
    ])

    X_proc = transformer.fit_transform(X)

    iso = IsolationForest(random_state=42, contamination=0.05)
    mask = iso.fit_predict(X_proc) == 1

    return input_data, (X_proc[mask], y[mask])
