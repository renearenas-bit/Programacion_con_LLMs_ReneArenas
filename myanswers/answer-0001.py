import pandas as pd
import numpy as np

def segmentar_pacientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función de limpieza y segmentación de pacientes.
    """
    # 1. Limpieza
    df_limpio = df.drop_duplicates().dropna().copy()
    
    # 2. Segmentación
    condiciones = [
        (df_limpio['glucosa'] >= 140) | (df_limpio['presion_arterial'] >= 140),
        (df_limpio['glucosa'] >= 100) | (df_limpio['presion_arterial'] >= 120)
    ]
    opciones = ["alto", "medio"]
    df_limpio['grupo_riesgo'] = np.select(condiciones, opciones, default="bajo")
    
    # 3. Ordenar y resetear índice
    df_limpio = df_limpio.sort_values(by="edad", ascending=True).reset_index(drop=True)
    
    return df_limpio
