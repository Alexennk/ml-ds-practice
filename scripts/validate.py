import numpy as np
import pandas as pd


def transform_df_types(df, int_columns, float_columns=None, object_columns=None, int_type=np.int32):
    df[int_columns] = df[int_columns].astype(int_type)
    if float_columns is not None:
        df[float_columns] = df[float_columns].astype(np.float32)
    if object_columns is not None:
        df[object_columns] = df[object_columns].astype('category')
    return df
