import numpy as np
import pandas as pd
import itertools


# I use this function in every single notebook, so it's easier to import it as a separate function
def transform_df_types(df, transform_date=False, float_type=np.float32):
    if transform_date:
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    float_columns = df.select_dtypes(include=np.number).columns.tolist()
    object_columns = df.select_dtypes(include=object).columns.tolist()
    df[float_columns] = df[float_columns].astype(float_type)
    df[object_columns] = df[object_columns].astype('category')
    return df


def create_full_train_dataset(train_df, train_aggregated, items_df, categories_df, shops_df):
    full_train_df = []
    cols = ['date_block_num', 'shop_id', 'item_id']
    for i in range(34):
        sales = train_df[train_df.date_block_num == i]
        full_train_df.append(np.array(list(itertools.product([i], shops_df.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
        
    full_train_df = pd.DataFrame(np.vstack(full_train_df), columns=cols)
    full_train_df.sort_values(cols, inplace=True)

    full_train_df = full_train_df.merge(train_aggregated, on=cols, how='left')

    full_train_df.fillna(0, inplace=True)
    full_train_df.drop('item_price', axis=1, inplace=True)
    full_train_df = transform_df_types(full_train_df, int_columns=['item_cnt_month'])
    full_train_df = ETLTransform.merge_df(full_train_df, items_df, categories_df, shops_df)

    return full_train_df
    

class ETLTransform:
    @staticmethod
    def del_negative(df, column_name):
        negatives = df[df[column_name] < 0]
        if len(negatives) > 0:
            df = df[df[column_name] >= 0]
        return df


    @staticmethod
    def merge_df(df, items_df, categories_df, shops_df):
        df = df.merge(items_df, on='item_id', how='left')
        df = df.merge(categories_df, on='item_category_id', how='left')
        df = df.merge(shops_df, on='shop_id', how='left')
        return df


    @staticmethod
    def add_month_year_columns(df):
        df['month'] = df['date_block_num'] % 12
        df['year'] = df['date_block_num'] // 12
        return df


    @staticmethod
    def change_shop_ids(df):
        df.loc[df['shop_id'] == 0, 'shop_id'] = 57
        df.loc[df['shop_id'] == 1, 'shop_id'] = 58
        df.loc[df['shop_id'] == 10, 'shop_id'] = 11

        return df
    

    @staticmethod
    def change_shop_names(df):
        df.loc[df['shop_name'] == '!Якутск Орджоникидзе, 56 фран', 'shop_name'] = 'Якутск Орджоникидзе, 56'
        df.loc[df['shop_name'] == '!Якутск ТЦ "Центральный" фран', 'shop_name'] = 'Якутск ТЦ "Центральный"'
        df.loc[df['shop_name'] == 'Жуковский ул. Чкалова 39м?', 'shop_name'] = 'Жуковский ул. Чкалова 39м²'

        return df