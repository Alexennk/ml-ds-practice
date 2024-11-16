import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from scripts.etl import transform_df_types


class FeatureExtractionLayer:
    @staticmethod
    def add_zero_records(df, month_left=11, month_right=33, num_new_records=100000):
        item_price_median = df['item_price'].median()

        new_data = {
            'date_block_num': np.random.randint(month_left, month_right, num_new_records), # I use records with date_block_num >= 11 for training for submissions to decrease the number of zero lag features
            'shop_id': np.random.choice(df['shop_id'].unique().tolist(), num_new_records),
            'item_id': np.random.choice(df['item_id'].unique().tolist(), num_new_records),
            'item_cnt_month': 0,
            'item_price': item_price_median,
            'month': 0,
            'year': 0,
            'item_name': 'Sample Item',
            'item_category_id': np.random.choice(df['item_category_id'].unique().tolist(), num_new_records),
            'item_category_name': 'Sample Category',
            'shop_name': 'Sample Shop'
        }

        new_df = pd.DataFrame(new_data)

        new_df['month'] = new_df['date_block_num'] % 12
        new_df['year'] = new_df['date_block_num'] // 12

        df = pd.concat([df, new_df], ignore_index=True)

        return df

    @staticmethod
    def train_add_months_since_last_sale(aggregated_df, load_precalculated=False, precalculated_path='../data/aggregated_months.csv'):
        
        if load_precalculated:
            aggregated_sorted = pd.read_csv(precalculated_path)
        else:
            aggregated_sorted = aggregated_df.sort_values(by=['item_id', 'date_block_num']).reset_index(drop=True)

            aggregated_sorted['months_since_last_sale'] = -1

            for row in aggregated_sorted.itertuples():
                if row.Index == 0: 
                    continue
                
                row_prev = aggregated_sorted.iloc[row.Index - 1]
                if row_prev['item_id'] == row.item_id:
                    if row.date_block_num == row_prev['date_block_num']:
                        aggregated_sorted.at[row.Index, 'months_since_last_sale'] = row_prev['months_since_last_sale']
                    else:
                        aggregated_sorted.at[row.Index, 'months_since_last_sale'] = row.date_block_num - row_prev['date_block_num']

            aggregated_sorted.to_csv(precalculated_path, index=False)
        
        int_columns = ['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'item_category_id', 'months_since_last_sale']
        float_columns = ['item_price', 'item_cnt_month']
        object_columns = ['item_name', 'item_category_name', 'shop_name']

        aggregated_train_df = transform_df_types(aggregated_sorted, int_columns, float_columns, object_columns)

        aggregated_train_df['months_since_last_sale'] = aggregated_train_df['months_since_last_sale'].replace(-1, 0)
            
        return aggregated_train_df
    

    @staticmethod
    def test_add_months_since_last_sale(aggregated_df, test_df, load_precalculated=False, precalculated_path='../data/test_months.csv'):
        
        if load_precalculated:
            test_df = pd.read_csv(precalculated_path)
        else:
            train_subdf = aggregated_df.copy()[['date_block_num', 'item_id', 'months_since_last_sale']]
            train_subdf['is_test'] = 0

            test_subdf = test_df[['item_id']]
            test_subdf['is_test'] = 1
            test_subdf['date_block_num'] = 34
            test_subdf['months_since_last_sale'] = -1

            united_subdf = pd.concat([train_subdf, test_subdf])

            united_subdf_sorted = united_subdf.sort_values(by=['item_id', 'date_block_num']).reset_index(drop=True)

            for row in united_subdf_sorted.itertuples():
                if row.Index == 0 or row.is_test == 0: 
                    continue
                
                row_prev = united_subdf_sorted.iloc[row.Index - 1]
                if row_prev['item_id'] == row.item_id:
                    if row.date_block_num == row_prev['date_block_num']:
                        united_subdf_sorted.at[row.Index, 'months_since_last_sale'] = row_prev['months_since_last_sale']
                    else:
                        united_subdf_sorted.at[row.Index, 'months_since_last_sale'] = row.date_block_num - row_prev['date_block_num']

            test_df['is_test'] = 1
            test_df = test_df.merge(united_subdf_sorted, on=['item_id', 'is_test'], how='left')

            test_df.drop(columns=['is_test', 'date_block_num'], inplace=True)
            test_df = test_df.loc[test_df.groupby('ID')['months_since_last_sale'].idxmax()] # drop duplicates along the "ID" feature
            test_df.to_csv('../data/test_months.csv', index=False)

        int_columns = ['ID', 'shop_id', 'item_id', 'item_category_id', 'months_since_last_sale']
        object_columns = ['item_name', 'item_category_name', 'shop_name']

        test_df = transform_df_types(test_df, int_columns, object_columns=object_columns)

        test_df['months_since_last_sale'] = test_df['months_since_last_sale'].replace(-1, 0)

        test_df.reset_index(drop=True, inplace=True)

        return test_df
    

    @staticmethod
    def calculate_historical_mean(df, column_name='item_cnt_month', on_columns=['item_id', 'shop_id']):
        on_columns_name = '_'.join(on_columns)
        df = df.sort_values(by=on_columns + ['date_block_num'])
        df_aggregated = df.groupby(on_columns + ['date_block_num'])[column_name].mean().reset_index()
        df_aggregated[column_name +'_mean_on_' + on_columns_name] = df_aggregated.groupby(on_columns)[column_name].apply(lambda x: x.shift().expanding().mean()).reset_index(name='historical_mean')['historical_mean']
        df_aggregated = df_aggregated.drop(columns=[column_name])
        df = df.merge(df_aggregated, on=on_columns + ['date_block_num'], how='left')
        return df
    

    @staticmethod
    def test_calculate_historical_mean(test, df, column_name='item_cnt_month', on_columns=['item_id', 'shop_id']):
        on_columns_name = '_'.join(on_columns)
        df_aggregated = df.groupby(on_columns)[column_name].mean().reset_index(name=column_name +'_mean_on_' + on_columns_name)
        test = test.merge(df_aggregated, on=on_columns, how='left')
        return test


    @staticmethod
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


    @staticmethod
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
    

    @staticmethod
    def test_add_lag_features_for_month(df, original_df, col, add_name='', on_columns=['shop_id', 'item_id'], operation='mean', lags=[1, 2, 3]):
        new_df = original_df[on_columns + [col]]
        new_df = new_df.groupby(on_columns).agg({col: operation}).reset_index()
        for lag in lags:
            tmp = new_df.copy()
            tmp = tmp[tmp['date_block_num'] == 34 - lag]
            tmp.columns = on_columns + [col + add_name + '_lag_' + str(lag)]
            tmp['date_block_num'] = 34
            df = df.merge(tmp, on=on_columns, how='left')

        for lag in lags:
            df['{}_lag_{}'.format(col + add_name, lag)] = df['{}_lag_{}'.format(col + add_name, lag)].fillna(0)
        return df


    @staticmethod
    def train_transform(df, aggregated_df, num_new_zero_records=0, clip_target=True):
        if num_new_zero_records > 0:
            aggregated_df = FeatureExtractionLayer.add_zero_records(aggregated_df, num_new_records=num_new_zero_records)

        # 1. Add "months_since_last_sale" feature
        aggregated_df = FeatureExtractionLayer.train_add_months_since_last_sale(aggregated_df, load_precalculated=True)

        # 2. Calculate revenue
        df['revenue'] = df['item_price'] * df['item_cnt_day']
        train_aggregated = df.groupby(['date_block_num', 'shop_id']).agg({'revenue': 'sum'}).reset_index()
        aggregated_df = aggregated_df.merge(train_aggregated, on=['date_block_num', 'shop_id'], how='left')
        aggregated_df['revenue'].fillna(0, inplace=True)

        # 3. Calculate historical mean revenue information
        aggregated_df = FeatureExtractionLayer.calculate_historical_mean(aggregated_df, 'revenue', ['shop_id'])
        aggregated_df['revenue_mean_on_shop_id'].fillna(0, inplace=True)

        # 4. Add revenue lagged features
        aggregated_lagged = FeatureExtractionLayer.train_add_lag_features(aggregated_df, 'revenue', on_columns=['shop_id', 'date_block_num'], operation='mean', lags=[1, 2, 3])

        # 5. Clip "item_cnt_month" into [0, 20] range; optional for kaggle
        aggregated_lagged['item_cnt_month'] = aggregated_lagged['item_cnt_month'].clip(lower=0)
        if clip_target:
            aggregated_lagged['item_cnt_month'] = aggregated_lagged['item_cnt_month'].clip(upper=20)

        # 6. Calculate historical mean item_cnt_month information
        aggregated_lagged = FeatureExtractionLayer.calculate_historical_mean(aggregated_lagged, 'item_cnt_month', on_columns=['item_id'])
        aggregated_lagged = FeatureExtractionLayer.calculate_historical_mean(aggregated_lagged, 'item_cnt_month', on_columns=['item_category_id'])
        aggregated_lagged['item_cnt_month_mean_on_item_id'].fillna(aggregated_lagged['item_cnt_month_mean_on_item_category_id'], inplace=True)
        aggregated_lagged.drop('item_cnt_month_mean_on_item_category_id', axis=1, inplace=True)
        aggregated_lagged['item_cnt_month_mean_on_item_id'].fillna(0, inplace=True)

        # 7. Add "item_cnt_month" lagged features
        aggregated_lagged = FeatureExtractionLayer.train_add_lag_features(aggregated_lagged, 'item_cnt_month', on_columns=['item_id', 'shop_id', 'date_block_num'], lags=[1, 2, 3])

        # 8. Add average "item_cnt_month" lagged features by month
        aggregated_lagged = FeatureExtractionLayer.train_add_lag_features(aggregated_lagged, 'item_cnt_month', add_name='_date_', on_columns=['date_block_num'], lags=[1, 2, 3])

        # 9. Add average "item_cnt_month" lagged features by category
        aggregated_lagged = FeatureExtractionLayer.train_add_lag_features(aggregated_lagged, 'item_cnt_month', add_name='_cat_', on_columns=['item_category_id', 'date_block_num'], lags=[1, 2, 3])

        # 10. Add average "item_cnt_month" lagged features by item
        aggregated_lagged = FeatureExtractionLayer.train_add_lag_features(aggregated_lagged, 'item_cnt_month', add_name='_item_', on_columns=['date_block_num', 'item_id'], lags=[1, 2, 3])

        # 11. Add average "item_cnt_month" lagged features by category and shop
        aggregated_lagged = FeatureExtractionLayer.train_add_lag_features(aggregated_lagged, 'item_cnt_month', add_name='_cat_shop_', on_columns=['item_category_id', 'shop_id', 'date_block_num'], lags=[1, 2, 3])

        # 12. Add "number of days in the month" feature
        days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        aggregated_lagged['days'] = aggregated_lagged['month'].map(days)

        # 13. Add "item_price" lag features
        avg_item_price = aggregated_lagged.groupby(['item_id', 'date_block_num'])['item_price'].mean().reset_index()
        avg_item_price.columns = ['item_id', 'date_block_num', 'avg_item_price']
        aggregated_lagged = aggregated_lagged.merge(avg_item_price, on=['item_id', 'date_block_num'], how='left')
        aggregated_lagged = FeatureExtractionLayer.train_add_lag_features(aggregated_lagged, 'avg_item_price', on_columns=['item_id', 'date_block_num'], lags=[1, 2, 3])

        return aggregated_lagged
    

    @staticmethod
    def test_transform(df, aggregated_df, test_df, num_new_zero_records=0, clip_target=True):
        if num_new_zero_records > 0:
            aggregated_df = FeatureExtractionLayer.add_zero_records(aggregated_df, num_new_records=num_new_zero_records)

        # 1. Add "months_since_last_sale" feature
        test_df = FeatureExtractionLayer.test_add_months_since_last_sale(aggregated_df, test_df, load_precalculated=True)

        # 2. Calculate revenue
        df['revenue'] = df['item_price'] * df['item_cnt_day']
        train_aggregated = df.groupby(['date_block_num', 'shop_id']).agg({'revenue': 'sum'}).reset_index()
        aggregated_df = aggregated_df.merge(train_aggregated, on=['date_block_num', 'shop_id'], how='left')
        aggregated_df['revenue'].fillna(0, inplace=True)

        # 3. Calculate historical mean revenue
        test_df = FeatureExtractionLayer.test_calculate_historical_mean(test_df, aggregated_df, 'revenue', ['shop_id'])

        # 4. Add revenue lagged features
        test_lagged = FeatureExtractionLayer.test_add_lag_features(test_df, aggregated_df, col='revenue', on_columns=['shop_id'], lags=[1, 2, 3])

        # 4.5. Clip "item_cnt_month" into [0, 20] range; optional for kaggle
        aggregated_df['item_cnt_month'] = aggregated_df['item_cnt_month'].clip(lower=0)
        if clip_target:
            aggregated_df['item_cnt_month'] = aggregated_df['item_cnt_month'].clip(upper=20)

        # 5. Calculate historical mean "item_cnt_month"
        test_lagged = FeatureExtractionLayer.test_calculate_historical_mean(test_lagged, aggregated_df, 'item_cnt_month', on_columns=['item_id'])
        test_lagged = FeatureExtractionLayer.test_calculate_historical_mean(test_lagged, aggregated_df, 'item_cnt_month', on_columns=['item_category_id'])
        test_lagged['item_cnt_month_mean_on_item_id'].fillna(test_lagged['item_cnt_month_mean_on_item_category_id'], inplace=True)
        test_lagged.drop('item_cnt_month_mean_on_item_category_id', axis=1, inplace=True)

        # 6. Add "item_cnt_month" lagged features
        test_lagged = FeatureExtractionLayer.test_add_lag_features(test_lagged, aggregated_df, col='item_cnt_month', on_columns=['item_id', 'shop_id'], lags=[1, 2, 3])

        # 7. Add average "item_cnt_month" lagged features by month
        test_lagged['date_block_num'] = 34
        test_lagged = FeatureExtractionLayer.test_add_lag_features_for_month(test_lagged, aggregated_df, col='item_cnt_month', add_name='_date_', on_columns=['date_block_num'], lags=[1, 2, 3])
        test_lagged.drop('date_block_num', axis=1, inplace=True)

        # 8. Add average "item_cnt_month" lagged features by category
        test_lagged = FeatureExtractionLayer.test_add_lag_features(test_lagged, aggregated_df, col='item_cnt_month', add_name='_cat_', on_columns=['item_category_id'], lags=[1, 2, 3])

        # 9. Add average "item_cnt_month" lagged features by item
        test_lagged = FeatureExtractionLayer.test_add_lag_features(test_lagged, aggregated_df, col='item_cnt_month', add_name='_item_', on_columns=['item_id'], lags=[1, 2, 3])

        # 10. Add average "item_cnt_month" lagged features by category and shop
        test_lagged = FeatureExtractionLayer.test_add_lag_features(test_lagged, aggregated_df, col='item_cnt_month', add_name='_cat_shop_', on_columns=['item_category_id', 'shop_id'], lags=[1, 2, 3])

        # 11. Add "month", "year" features
        test_lagged['month'] = 10
        test_lagged['year'] = 2

        # 12. Add "number of days in the month" feature
        days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        test_lagged['days'] = test_lagged['month'].map(days)

        # 13. Add "item_price" lag features
        avg_item_price = aggregated_df.groupby(['item_id', 'date_block_num'])['item_price'].mean().reset_index()
        avg_item_price.columns = ['item_id', 'date_block_num', 'avg_item_price']
        aggregated_df = aggregated_df.merge(avg_item_price, on=['item_id', 'date_block_num'], how='left')
        test_lagged = FeatureExtractionLayer.test_add_lag_features(test_lagged, aggregated_df, col='avg_item_price', on_columns=['item_id'], lags=[1, 2, 3])

        return test_lagged