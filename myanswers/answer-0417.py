import pandas as pd
import numpy as np

def construir_panel_diario(df, fecha_col, grupo_col, valor_col, **kwargs):
    """
    Solución corregida para el caso de uso 0417.
    Recibe explícitamente df, fecha_col, grupo_col y valor_col de forma posicional.
    """
    # Copiar el DataFrame para evitar modificar el original en sitio
    trabajo = df.copy()
    
    # 1) Convertir la columna fecha_col a datetime con errores forzados a NaT
    trabajo[fecha_col] = pd.to_datetime(trabajo[fecha_col], errors="coerce")
    
    # 2) Eliminar filas con fechas inválidas o faltantes en grupo/valor
    trabajo = trabajo.dropna(subset=[fecha_col, grupo_col, valor_col])

    # Si el DataFrame queda vacío tras la limpieza, retornar la estructura base esperada
    if trabajo.empty:
        return pd.DataFrame(columns=[fecha_col, grupo_col, "valor_total"])

    # 3) Sumar el valor agrupado por fecha y grupo
    agregado = trabajo.groupby([fecha_col, grupo_col], as_index=False)[valor_col].sum()

    # 4) Generar el rango de fechas continuas y ordenar los grupos válidos exactamente igual que el validador
    fechas_completas = pd.date_range(
        start=agregado[fecha_col].min(),
        end=agregado[fecha_col].max(),
        freq="D",
    )
    grupos_validos = sorted(agregado[grupo_col].unique())

    # Crear el MultiIndex cartesiano
    indice_completo = pd.MultiIndex.from_product(
        [fechas_completas, grupos_validos], names=[fecha_col, grupo_col]
    )

    # 5) Reindexar usando la misma lógica del compañero para rellenar con 0.0
    panel = (
        agregado.set_index([fecha_col, grupo_col])
        .reindex(indice_completo, fill_value=0.0)
        .reset_index()
    )

    # 6) Renombrar la columna del valor a 'valor_total'
    panel = panel.rename(columns={valor_col: "valor_total"})
    
    # Ordenar por grupo_col, luego por fecha_col y reiniciar índices limpiamente
    panel = panel.sort_values([grupo_col, fecha_col]).reset_index(drop=True)
    
    # Asegurar el casteo estricto a flotante (float)
    panel["valor_total"] = panel["valor_total"].astype(float)
    
    return panel
     
