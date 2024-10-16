import numpy as np
import pandas as pd


def transform_df_types(df, int_columns, float_columns=None, object_columns=None, int_type=np.int32):
    df[int_columns] = df[int_columns].astype(int_type)
    if float_columns is not None:
        df[float_columns] = df[float_columns].astype(np.float32)
    if object_columns is not None:
        df[object_columns] = df[object_columns].astype('category')
    return df


def train_add_lag_features(df, col, add_name='', on_columns=['date_block_num', 'shop_id', 'item_id'], operation='mean', lags=[1, 2, 3]):
    new_df = df[on_columns + [col]]
    new_df = new_df.groupby(on_columns).agg({col: operation}).reset_index()
    for lag in lags:
        tmp = new_df.copy()
        tmp.loc[:, 'date_block_num'] = tmp['date_block_num'] + lag
        tmp.columns = on_columns + [col + add_name + '_lag_' + str(lag)]
        df = df.merge(tmp, on=on_columns, how='left')

    for lag in lags:
        df['{}_lag_{}'.format(col + add_name, lag)] = df['{}_lag_{}'.format(col + add_name, lag)].fillna(0)
    return df


def test_add_lag_features(df, original_df, col, add_name='', on_columns=['shop_id', 'item_id'], operation='mean', lags=[1, 2, 3]):
    new_df = original_df[on_columns + ['date_block_num', col]]
    new_df = new_df.groupby(on_columns + ['date_block_num']).agg({col: operation}).reset_index()
    for lag in lags:
        tmp = new_df.copy()
        tmp = tmp[tmp['date_block_num'] == 34 - lag]
        tmp.drop('date_block_num', axis=1, inplace=True)
        tmp.columns = on_columns + [col + add_name + '_lag_' + str(lag)]
        df = df.merge(tmp, on=on_columns, how='left')

    for lag in lags:
        df['{}_lag_{}'.format(col + add_name, lag)] = df['{}_lag_{}'.format(col + add_name, lag)].fillna(0)
    return df