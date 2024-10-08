import numpy as np
import pandas as pd


def transform_df_types(df, int_columns, float_columns=None, object_columns=None):
    df[int_columns] = df[int_columns].astype(np.int32)
    if float_columns is not None:
        df[float_columns] = df[float_columns].astype(np.float32)
    if object_columns is not None:
        df[object_columns] = df[object_columns].astype('category')
    return df
    

def del_negative(df, column_name):
    negatives = df[df[column_name] < 0]
    if len(negatives) > 0:
        df = df[df[column_name] >= 0]
    return df


def merge_df(df, items_df, categories_df, shops_df):
    df = df.merge(items_df, on='item_id', how='left')
    df = df.merge(categories_df, on='item_category_id', how='left')
    df = df.merge(shops_df, on='shop_id', how='left')
    return df


def add_month_year_columns(df):
    df['month'] = df['date_block_num'] % 12
    df['year'] = df['date_block_num'] // 12
    return df


def change_shop_attributes(df):

    df.loc[df['shop_id'] == 0, 'shop_id'] = 57
    df.loc[df['shop_id'] == 1, 'shop_id'] = 58
    df.loc[df['shop_id'] == 10, 'shop_id'] = 11

    df.loc[df['shop_name'] == '!Якутск Орджоникидзе, 56 фран', 'shop_name'] = 'Якутск Орджоникидзе, 56'
    df.loc[df['shop_name'] == '!Якутск ТЦ "Центральный" фран', 'shop_name'] = 'Якутск ТЦ "Центральный"'
    df.loc[df['shop_name'] == 'Жуковский ул. Чкалова 39м?', 'shop_name'] = 'Жуковский ул. Чкалова 39м²'

    return df