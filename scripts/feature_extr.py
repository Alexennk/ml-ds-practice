import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def transform_df_types(df, int_columns, float_columns=None, object_columns=None, int_type=np.int32):
    df[int_columns] = df[int_columns].astype(int_type)
    if float_columns is not None:
        df[float_columns] = df[float_columns].astype(np.float32)
    if object_columns is not None:
        df[object_columns] = df[object_columns].astype('category')
    return df


def train_add_lag_features(df, col, on=['item_id', 'shop_id', 'date_block_num'], lags=[1, 2, 3]):
    new_df = df[['item_id', 'shop_id', 'date_block_num', col]]
    for lag in lags:
        tmp = new_df.copy()
        tmp.loc[:, 'date_block_num'] = tmp['date_block_num'] + lag
        tmp.columns = ['item_id', 'shop_id', 'date_block_num', col + '_lag_' + str(lag)]
        df = df.merge(tmp, on=['item_id', 'shop_id', 'date_block_num'], how='left')

    for lag in lags:
        df['{}_lag_{}'.format(col, lag)] = df['{}_lag_{}'.format(col, lag)].fillna(0)
    return df


def test_add_lag_features(df, original_df, col, on=['item_id', 'shop_id'], lags=[1, 2, 3]):
    new_df = original_df[['item_id', 'shop_id', 'date_block_num', col]]
    for lag in lags:
        tmp = new_df.copy()
        tmp = tmp[tmp['date_block_num'] == 34 - lag]
        tmp.drop('date_block_num', axis=1, inplace=True)
        tmp.columns = ['item_id', 'shop_id', col + '_lag_' + str(lag)]
        df = df.merge(tmp, on=['item_id', 'shop_id'], how='left')

    for lag in lags:
        df['{}_lag_{}'.format(col, lag)] = df['{}_lag_{}'.format(col, lag)].fillna(0)
    return df


def threshold_sales(df, aggregated_df, column_name, on_test=False):
    column_aggregated = df.groupby([column_name])['item_cnt_month'].mean()
    column_aggregated = pd.DataFrame(column_aggregated).reset_index()

    def divide_sales(sales):
        low_threshold = column_aggregated['item_cnt_month'].quantile(0.25)
        high_threshold = column_aggregated['item_cnt_month'].quantile(0.75)

        if sales < low_threshold:
            return 0
        elif sales < high_threshold:
            return 1
        else:
            return 2
        
    column_aggregated[column_name + '_sales_level'] = column_aggregated['item_cnt_month'].apply(divide_sales)

    aggregated_df = aggregated_df.merge(column_aggregated, on=column_name, how='left')
    if on_test:
        aggregated_df.drop('item_cnt_month', axis=1, inplace=True)
    else:
        aggregated_df.drop('item_cnt_month_y', axis=1, inplace=True)
        aggregated_df.rename(columns={'item_cnt_month_x': 'item_cnt_month'}, inplace=True)

    return aggregated_df