import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_add_lag_features(df, lags=3):
    new_df = df[['item_id', 'shop_id', 'date_block_num', 'item_cnt_month']]
    for lag in range(1, lags + 1):
        new_df.loc[:, 'date_block_num'] = new_df['date_block_num'] + lag
        new_df.columns = ['item_id', 'shop_id', 'date_block_num', 'item_cnt_month_lag_' + str(lag)]
        df = df.merge(new_df, on=['item_id', 'shop_id', 'date_block_num'], how='left')

    df[['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']] = df[['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']].fillna(0)
    return df


def test_add_lag_features(df, original_df, lags=3):
    for lag in range(1, lags + 1):
        new_df = original_df[['item_id', 'shop_id', 'item_cnt_month', 'date_block_num']]
        new_df = new_df[new_df['date_block_num'] == 34 - lag]
        new_df.drop('date_block_num', axis=1, inplace=True)
        new_df.columns = ['item_id', 'shop_id', 'item_cnt_month_lag_' + str(lag)]
        df = df.merge(new_df, on=['item_id', 'shop_id'], how='left')

    df.fillna(0, inplace=True)
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