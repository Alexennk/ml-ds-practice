import numpy as np
import pandas as pd


def transform_df_types(df, int_columns, float_columns=None, object_columns=None):
    df[int_columns] = df[int_columns].astype(np.int32)
    if float_columns is not None:
        df[float_columns] = df[float_columns].astype(np.float32)
    if object_columns is not None:
        df[object_columns] = df[object_columns].astype('category')
    return df

def change_shop_attributes(df):

    df.loc[df['shop_id'] == 0, 'shop_id'] = 57
    df.loc[df['shop_id'] == 1, 'shop_id'] = 58
    df.loc[df['shop_id'] == 10, 'shop_id'] = 11

    df.loc[df['shop_name'] == '!Якутск Орджоникидзе, 56 фран', 'shop_name'] = 'Якутск Орджоникидзе, 56'
    df.loc[df['shop_name'] == '!Якутск ТЦ "Центральный" фран', 'shop_name'] = 'Якутск ТЦ "Центральный"'
    df.loc[df['shop_name'] == 'Жуковский ул. Чкалова 39м?', 'shop_name'] = 'Жуковский ул. Чкалова 39м²'

    return df

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

def threshold_sales(df, aggregated_df, column_name):
    column_aggregated = df.groupby([column_name])['item_cnt_day'].sum()
    column_aggregated = pd.DataFrame(column_aggregated).reset_index()

    def divide_sales(sales):
        low_threshold = column_aggregated['item_cnt_day'].quantile(0.25)
        high_threshold = column_aggregated['item_cnt_day'].quantile(0.75)

        if sales < low_threshold:
            return 0
        elif sales < high_threshold:
            return 1
        else:
            return 2
        
    column_aggregated[column_name + '_sales_level'] = column_aggregated['item_cnt_day'].apply(divide_sales)

    aggregated_df = aggregated_df.merge(column_aggregated, on=column_name, how='left')
    aggregated_df.drop('item_cnt_day', axis=1, inplace=True)

    return aggregated_df