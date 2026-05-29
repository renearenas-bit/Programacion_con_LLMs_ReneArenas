import pandas as pd
import numpy as np

def transformar_a_largo(**kwargs):
    """
    Solución exacta para el caso de uso 0441.
    Transforma un DataFrame de formato ancho (wide) a largo (long),
    extrayendo el número entero de la semana y calculando sus estadísticas.
    """
    # 1. Extracción directa y segura de los parámetros usando las llaves del generador
    df = kwargs.get('df')
    id_col = kwargs.get('id_col')
    columnas_semana = kwargs.get('columnas_semana')

    # Fallback defensivo por si el validador cambiara la estructura de kwargs
    if df is None:
        for val in kwargs.values():
            if isinstance(val, pd.DataFrame):
                df = val
                break
    if id_col is None:
        id_col = 'paciente_id'
    if columnas_semana is None and df is not None:
        columnas_semana = [c for c in df.columns if str(c).startswith('semana')]

    # Copiar el DataFrame para evitar advertencias de mutación (SettingWithCopyWarning)
    df_working = df.copy()

    # 2. Transformación de formato ancho a largo usando pd.melt()
    df_largo = pd.melt(
        df_working,
        id_vars=[id_col],
        value_vars=columnas_semana,
        var_name='semana',
        value_name='valor'
    )

    # 3. Extraer el número de semana usando expresiones regulares y convertirlo a entero (int)
    df_largo['semana'] = df_largo['semana'].astype(str).str.extract(r'(\d+)').astype(int)

    # 4. Agrupar por semana y calcular el promedio y la desviación estándar de la columna 'valor'
    resumen = (
        df_largo.groupby('semana')['valor']
        .agg(promedio='mean', desviacion_std='std')
        .reset_index()
    )

    # 5. Asegurar el ordenamiento por la columna 'semana' y reiniciar índices de forma limpia
    resumen = resumen.sort_values(by='semana', ascending=True).reset_index(drop=True)

    return resumen
