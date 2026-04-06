# Programacion_con_LLMs_ReneArenas
Fase1
# Preguntas - Programación con LLMs

## Pregunta 1: Generación de variables rezagadas (lags)

En el análisis de series temporales, es común crear variables rezagadas (lags) para capturar la dependencia de valores pasados sobre el presente.

### Tu Misión

Escribe una función llamada `crear_features_lag(df, columna_valor, n_lags)` que:

1. Reciba un DataFrame ordenado temporalmente.
2. Cree columnas `lag_1`, `lag_2`, ..., `lag_n`.
3. Cada columna debe usar shift(k).
4. Elimine filas con NaN generados.
5. No modifique el DataFrame original.

### Retorna:
Un DataFrame con columnas originales + lags.

---

## Pregunta 2: Evaluación con ROC-AUC

En clasificación binaria, el ROC-AUC permite evaluar el modelo en distintos umbrales.

### Tu Misión

Escribe una función llamada `evaluar_modelo_roc(df, target_col)` que:

1. Separe X e y.
2. Divida en train/test (80/20, random_state=42).
3. Entrene LogisticRegression.
4. Obtenga probabilidades.
5. Calcule ROC-AUC.

### Retorna:
Un float (AUC).

---

## Pregunta 3: Pipeline con selección de variables

Se desea automatizar selección de features y entrenamiento.

### Tu Misión

Escribe una función llamada `pipeline_seleccion_modelo(df, target_col, k_features)` que:

1. Separe X e y.
2. Cree Pipeline:
   - SelectKBest
   - Ridge
3. Divida datos (80/20).
4. Entrene modelo.
5. Calcule R².

### Retorna:
Un diccionario con:
- r2
- features_seleccionadas

---

## Pregunta 4: Detección de drift de datos

El data drift ocurre cuando cambian las distribuciones.

### Tu Misión

Escribe una función llamada `detectar_drift_datos(df1, df2)` que:

1. Calcule media y std por columna.
2. Calcule:

   drift = |media1 - media2| / (std1 + std2)

3. Maneje divisiones por cero.
4. No use loops.

### Retorna:
Un array de numpy con el drift por columna.
