import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_detectar_retrasos_por_ciudad():
    np.random.seed(0)
    random.seed(0)

    n = 100
    ciudades = ['Bogota', 'Medellin', 'Cali']

    df = pd.DataFrame({
        'ciudad': [random.choice(ciudades) for _ in range(n)],
        'fecha_envio': pd.date_range('2024-01-01', periods=n).astype(str),
        'tiempo_entrega': [round(random.uniform(2, 10), 2) for _ in range(n)]
    })

    df.loc[0, 'tiempo_entrega'] = 50.0

    input_data = {'df': df.copy()}

    df_res = df.copy()
    df_res['fecha_envio'] = pd.to_datetime(df_res['fecha_envio'])
    df_res = df_res.sort_values(['ciudad', 'fecha_envio'])

    stats = df_res.groupby('ciudad')['tiempo_entrega'].agg(['mean', 'std']).reset_index()
    stats.columns = ['ciudad', 'media_ciudad', 'std_ciudad']

    df_res = df_res.merge(stats, on='ciudad')

    df_res['retraso'] = df_res['tiempo_entrega'] > (
        df_res['media_ciudad'] + 2 * df_res['std_ciudad']
    )

    return input_data, df_res
