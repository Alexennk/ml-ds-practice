import numpy as np
import pandas as pd


def transform_df_types(df, int_columns, float_columns=None, object_columns=None, int_type=np.int32):
    df[int_columns] = df[int_columns].astype(int_type)
    if float_columns is not None:
        df[float_columns] = df[float_columns].astype(np.float32)
    if object_columns is not None:
        df[object_columns] = df[object_columns].astype('category')
    return df


class FeatureExtractionLayer:
    def __init__(self):
        pass
    

    def train_add_months_since_last_sale(self, aggregated_df, load_precalculated=False, precalculated_path='../data/aggregated_months.csv'):
        
        if load_precalculated:
            aggregated_sorted = pd.read_csv(precalculated_path)
        else:
            aggregated_sorted = aggregated_df.sort_values(by=['shop_id', 'item_id', 'date_block_num']).reset_index(drop=True)

            aggregated_sorted['months_since_last_sale'] = -1

            for row in aggregated_sorted.itertuples():
                if row.Index == 0: 
                    continue
                
                row_prev = aggregated_sorted.iloc[row.Index - 1]
                if row_prev['shop_id'] == row.shop_id and row_prev['item_id'] == row.item_id:
                    aggregated_sorted.at[row.Index, 'months_since_last_sale'] = row.date_block_num - row_prev['date_block_num']

            aggregated_sorted.to_csv(precalculated_path, index=False)
        
        int_columns = ['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'item_category_id', 'months_since_last_sale']
        float_columns = ['item_price', 'item_cnt_month']
        object_columns = ['item_name', 'item_category_name', 'shop_name']

        aggregated_train_df = transform_df_types(aggregated_sorted, int_columns, float_columns, object_columns)
            
        return aggregated_train_df
    

    def test_add_months_since_last_sale(self, aggregated_df, test_df, load_precalculated=False, precalculated_path='../data/test_months.csv'):
        
        if load_precalculated:
            test_df = pd.read_csv(precalculated_path)
        else:
            train_subdf = aggregated_df.copy()[['date_block_num', 'shop_id', 'item_id', 'months_since_last_sale']]
            train_subdf['is_test'] = 0

            test_subdf = test_df[['shop_id', 'item_id']]
            test_subdf['is_test'] = 1
            test_subdf['date_block_num'] = 34
            test_subdf['months_since_last_sale'] = -1

            united_subdf = pd.concat([train_subdf, test_subdf])

            united_subdf_sorted = united_subdf.sort_values(by=['shop_id', 'item_id', 'date_block_num']).reset_index(drop=True)

            for row in united_subdf_sorted.itertuples():
                if row.Index == 0 or row.is_test == 0: 
                    continue
                
                row_prev = united_subdf_sorted.iloc[row.Index - 1]
                if row_prev['shop_id'] == row.shop_id and row_prev['item_id'] == row.item_id:
                    united_subdf_sorted.at[row.Index, 'months_since_last_sale'] = row.date_block_num - row_prev['date_block_num']

            test_df['is_test'] = 1
            test_df = test_df.merge(united_subdf_sorted, on=['shop_id', 'item_id', 'is_test'], how='left')

            test_df.drop(columns=['is_test', 'date_block_num'], inplace=True)
            test_df = test_df.loc[test_df.groupby('ID')['months_since_last_sale'].idxmax()] # drop duplicates along the "ID" feature
            test_df.to_csv('../data/test_months.csv', index=False)

        int_columns = ['ID', 'shop_id', 'item_id', 'item_category_id', 'months_since_last_sale']
        object_columns = ['item_name', 'item_category_name', 'shop_name']

        test_df = transform_df_types(test_df, int_columns, object_columns=object_columns)

        return test_df


    def train_add_lag_features(self, df, col, add_name='', on_columns=['date_block_num', 'shop_id', 'item_id'], operation='mean', lags=[1, 2, 3]):
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


    def test_add_lag_features(self, df, original_df, col, add_name='', on_columns=['shop_id', 'item_id'], operation='mean', lags=[1, 2, 3]):
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


    def train_transform(self, df, aggregated_df):
        # 1. Add "months_since_last_sale" feature
        aggregated_df = self.train_add_months_since_last_sale(aggregated_df, load_precalculated=True)

        # 2. Calculate revenue
        df['revenue'] = df['item_price'] * df['item_cnt_day']
        train_aggregated = df.groupby(['date_block_num', 'shop_id']).agg({'revenue': 'sum'}).reset_index()
        aggregated_df = aggregated_df.merge(train_aggregated, on=['date_block_num', 'shop_id'], how='left')

        # 3. Add revenue lagged features
        aggregated_lagged = self.train_add_lag_features(aggregated_df, 'revenue', on_columns=['shop_id', 'date_block_num'], operation='mean', lags=[1, 2, 3, 6, 12])

        # 4. Clip "item_cnt_month" into [0, 20] range
        aggregated_lagged['item_cnt_month'] = aggregated_lagged['item_cnt_month'].clip(0, 20)

        # 5. Add "item_cnt_month" lagged features
        aggregated_lagged = self.train_add_lag_features(aggregated_lagged, 'item_cnt_month', on_columns=['item_id', 'shop_id', 'date_block_num'], lags=[1, 2, 3, 6, 12])

        # 6. Add average "item_cnt_month" lagged features by category
        aggregated_lagged = self.train_add_lag_features(aggregated_lagged, 'item_cnt_month', add_name='_cat_', on_columns=['item_category_id', 'date_block_num'], lags=[1, 2, 3])

        # 7. Add average "item_cnt_month" lagged features by category and shop
        aggregated_lagged = self.train_add_lag_features(aggregated_lagged, 'item_cnt_month', add_name='_cat_shop_', on_columns=['item_category_id', 'shop_id', 'date_block_num'], lags=[1, 2, 3])

        # 8. Add "number of days in the month" feature
        days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        aggregated_lagged['days'] = aggregated_lagged['month'].map(days)

        # 9. Add "item_price" lag features
        avg_item_price = aggregated_lagged.groupby(['item_id', 'date_block_num'])['item_price'].mean().reset_index()
        avg_item_price.columns = ['item_id', 'date_block_num', 'avg_item_price']
        aggregated_lagged = aggregated_lagged.merge(avg_item_price, on=['item_id', 'date_block_num'], how='left')
        aggregated_lagged = self.train_add_lag_features(aggregated_lagged, 'avg_item_price', on_columns=['item_id', 'date_block_num'], lags=[1, 2, 3, 4, 5, 6])

        return aggregated_lagged
    

    def test_transform(self, df, aggregated_df, test_df):
        # 1. Add "months_since_last_sale" feature
        test_df = self.test_add_months_since_last_sale(aggregated_df, test_df, load_precalculated=True)

        # 2. Calculate revenue
        df['revenue'] = df['item_price'] * df['item_cnt_day']
        train_aggregated = df.groupby(['date_block_num', 'shop_id']).agg({'revenue': 'sum'}).reset_index()
        aggregated_df = aggregated_df.merge(train_aggregated, on=['date_block_num', 'shop_id'], how='left')

        # 3. Add revenue lagged features
        test_lagged = self.test_add_lag_features(test_df, aggregated_df, col='revenue', on_columns=['shop_id'], lags=[1, 2, 3, 6, 12])

        # 3.5. Clip "item_cnt_month" into [0, 20] range
        aggregated_df['item_cnt_month'] = aggregated_df['item_cnt_month'].clip(0, 20)

        # 4. Add "item_cnt_month" lagged features
        test_lagged = self.test_add_lag_features(test_lagged, aggregated_df, col='item_cnt_month', on_columns=['item_id', 'shop_id'], lags=[1, 2, 3, 6, 12])

        # 5. Add average "item_cnt_month" lagged features by category
        test_lagged = self.test_add_lag_features(test_lagged, aggregated_df, col='item_cnt_month', add_name='_cat_', on_columns=['item_category_id'], lags=[1, 2, 3])

        # 6. Add average "item_cnt_month" lagged features by category and shop
        test_lagged = self.test_add_lag_features(test_lagged, aggregated_df, col='item_cnt_month', add_name='_cat_shop_', on_columns=['item_category_id', 'shop_id'], lags=[1, 2, 3])

        # 7. Add "month", "year" features
        test_lagged['month'] = 10
        test_lagged['year'] = 2

        # 8. Add "number of days in the month" feature
        days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        test_lagged['days'] = test_lagged['month'].map(days)

        # 9. Add "item_price" lag features
        avg_item_price = aggregated_df.groupby(['item_id', 'date_block_num'])['item_price'].mean().reset_index()
        avg_item_price.columns = ['item_id', 'date_block_num', 'avg_item_price']
        aggregated_df = aggregated_df.merge(avg_item_price, on=['item_id', 'date_block_num'], how='left')
        test_lagged = self.test_add_lag_features(test_lagged, aggregated_df, col='avg_item_price', on_columns=['item_id'], lags=[1, 2, 3, 4, 5, 6])

        return test_lagged
    

