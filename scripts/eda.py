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

def plot_barpot_boxplot(df, column_name, figsize=(14, 7), leave=None):
    # First Plot Data
    column_aggregated = df.groupby([column_name], observed=False)['item_cnt_month'].mean()
    column_aggregated = pd.DataFrame(column_aggregated)
    column_aggregated.reset_index(inplace=True)
    column_aggregated = column_aggregated.sort_values(['item_cnt_month'], ascending=False)
    if leave is not None:
        column_aggregated = column_aggregated[:leave]

    # Second Plot Data
    var = column_name
    data = pd.concat([df['item_cnt_month'], df[var]], axis=1)

    # Create subplots in one row with two columns
    fig, axes = plt.subplots(1, 2, figsize=figsize)  # Adjust the figsize as needed for better appearance

    # First plot: Barplot
    sns.barplot(x=column_aggregated['item_cnt_month'], 
                y=column_aggregated[column_name], 
                orient='h', 
                order=column_aggregated[column_name], 
                ax=axes[0])
    axes[0].set_title(column_name + " Sales Comparison")
    axes[0].set_xlabel("Average Sales Amount")
    axes[0].set_ylabel(column_name)

    # Second plot: Boxplot
    sns.boxplot(x='item_cnt_month', y=var, data=data, orient='h', order=column_aggregated[column_name], ax=axes[1])
    axes[1].set_title("Box Plots for " + column_name)
    axes[1].set_xlabel("Item Count per Month")
    axes[1].set_ylabel("")

    # Display the plots
    plt.tight_layout()
    plt.show()