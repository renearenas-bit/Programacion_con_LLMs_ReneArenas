import pandas as pd
import numpy as np

def segmentar_pacientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función de limpieza y segmentación de pacientes (Fase 2 - Pregunta 0001).
    """
    # 1. Limpieza: eliminar duplicados y nulos en el orden exacto del generador
    df_limpio = df.drop_duplicates().dropna().copy()
    
    # 2. Segmentación lógica mediante condiciones vectorizadas
    condiciones = [
        (df_limpio['glucosa'] >= 140) | (df_limpio['presion_arterial'] >= 140), # alto
        (df_limpio['glucosa'] >= 100) | (df_limpio['presion_arterial'] >= 120) # medio
    ]
    opciones = ["alto", "medio"]
    
    df_limpio['grupo_riesgo'] = np.select(condiciones, opciones, default="bajo")
    
    # 3. Ordenar por edad ascendente y resetear índice
    df_limpio = df_limpio.sort_values(by="edad", ascending=True).reset_index(drop=True)
    
    return df_limpio
