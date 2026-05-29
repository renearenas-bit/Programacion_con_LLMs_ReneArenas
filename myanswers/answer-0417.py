import pandas as pd
import numpy as np

def construir_panel_diario(df, fecha_col, grupo_col, valor_col, **kwargs):
    """
    Construye un panel diario completo rellenando fechas faltantes con 0.0
    para cada grupo de manera ordenada.
    """
    # Copia de trabajo para evitar mutaciones de datos externos
    df_trabajo = df.copy()
    
    # Convertir columna temporal de forma segura y remover registros nulos críticos
    df_trabajo[fecha_col] = pd.to_datetime(df_trabajo[fecha_col], errors="coerce")
    df_trabajo = df_trabajo.dropna(subset=[fecha_col, grupo_col, valor_col])

    if df_trabajo.empty:
        return pd.DataFrame(columns=[fecha_col, grupo_col, "valor_total"])

    # Agrupar y sumar los valores por fecha y grupo
    df_agregado = df_trabajo.groupby([fecha_col, grupo_col], as_index=False)[valor_col].sum()

    # Generar el rango completo de fechas sin vacíos (frecuencia diaria 'D')
    rango_fechas = pd.date_range(
        start=df_agregado[fecha_col].min(),
        end=df_agregado[fecha_col].max(),
        freq="D"
    )
    grupos_unicos = sorted(df_agregado[grupo_col].unique())

    # Crear el MultiIndex producto cartesiano (Todas las fechas x Todos los grupos)
    multi_indice = pd.MultiIndex.from_product(
        [rango_fechas, grupos_unicos], 
        names=[fecha_col, grupo_col]
    )

    # Reindexar el panel para aflorar los días faltantes y rellenar con 0.0
    panel_completo = (
        df_agregado.set_index([fecha_col, grupo_col])
        .reindex(multi_indice, fill_value=0.0)
        .reset_index()
    )

    # Formatear nombres de columnas, ordenamiento y tipos de datos finales
    panel_completo = panel_completo.rename(columns={valor_col: "valor_total"})
    panel_completo = panel_completo.sort_values([grupo_col, fecha_col]).reset_index(drop=True)
    panel_completo["valor_total"] = panel_completo["valor_total"].astype(float)
    
    return panel_completo

# Duplicación de firma explícita para mitigar el error de enlace del generador externo
def _resolver_construir_panel_diario(df, fecha_col, grupo_col, valor_col, **kwargs):
    return construir_panel_diario(df, fecha_col, grupo_col, valor_col, **kwargs)
