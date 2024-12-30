import numpy as np
import pandas as pd
import itertools
import re


# I use this function in every single notebook,
# so it's easier to import it as a separate function
def transform_df_types(df, transform_date=False, float_type=np.float32):
    if transform_date:
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")

    float_columns = df.select_dtypes(include=np.number).columns.tolist()
    object_columns = df.select_dtypes(include=object).columns.tolist()
    df[float_columns] = df[float_columns].astype(float_type)
    df[object_columns] = df[object_columns].astype("category")
    return df


def create_full_train_dataset(
    train_df, train_aggregated, items_df, categories_df, shops_df
):
    full_train_df = []
    cols = ["date_block_num", "shop_id", "item_id"]
    for i in range(34):
        sales = train_df[train_df.date_block_num <= i]
        shops = sales.shop_id.unique()
        for shop in shops:
            full_train_df.append(
                np.array(
                    list(
                        itertools.product(
                            [i],
                            [shop],
                            sales[sales["shop_id"] == shop].item_id.unique(),
                        )
                    ),
                    dtype="int16",
                )
            )

    full_train_df = pd.DataFrame(np.vstack(full_train_df), columns=cols)
    full_train_df.sort_values(cols, inplace=True)

    full_train_df = full_train_df.merge(train_aggregated, on=cols, how="left")

    full_train_df.fillna(0, inplace=True)
    full_train_df = transform_df_types(full_train_df)
    full_train_df = ETLTransform.merge_df(
        full_train_df, items_df, categories_df, shops_df
    )

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
        df = df.merge(items_df, on="item_id", how="left")
        df = df.merge(categories_df, on="item_category_id", how="left")
        df = df.merge(shops_df, on="shop_id", how="left")
        return df

    @staticmethod
    def add_month_year_columns(df):
        df["month"] = df["date_block_num"] % 12
        df["year"] = df["date_block_num"] // 12
        return df

    @staticmethod
    def change_shop_ids(df):
        df.loc[df["shop_id"] == 0, "shop_id"] = 57
        df.loc[df["shop_id"] == 1, "shop_id"] = 58
        df.loc[df["shop_id"] == 10, "shop_id"] = 11

        return df

    @staticmethod
    def change_shop_names(df):
        df.loc[df["shop_name"] == "!Якутск Орджоникидзе, 56 фран", "shop_name"] = (
            "Якутск Орджоникидзе, 56"
        )
        df.loc[df["shop_name"] == '!Якутск ТЦ "Центральный" фран', "shop_name"] = (
            'Якутск ТЦ "Центральный"'
        )
        df.loc[df["shop_name"] == "Жуковский ул. Чкалова 39м?", "shop_name"] = (
            "Жуковский ул. Чкалова 39м²"
        )

        return df

    @staticmethod
    def transform(train_df, items_df, categories_df, shops_df, return_aggregated=False):
        """
        This function transforms the datasets by:
            1) adding a column with cleaned item_name
            2) fixing shops_df and trains_df duplicate data
            3) adding year and month columns to received datasets
            4) dropping duplicates of item_name
            5) merging train_df with items_df, categories_df, shops_df

        Parameters:
        train_df (pd.DataFrame): train dataset
        items_df (pd.DataFrame): items dataset
        categories_df (pd.DataFrame): categories dataset
        shops_df (pd.DataFrame): shops dataset
        return_aggregated (bool): whether to return the aggregated dataset or not

        Returns:
        merged_train_df (pd.DataFrame): transformed train dataset
        merged_train_aggregated_df (pd.DataFrame): if return_aggregated is True
        """

        def clean_item_name(name):
            name = re.sub(r"[^\w\s]", "", name)
            name = re.sub(r" D$", "", name)
            return name.lower()

        # 1) add cleaned item_name column to analyze item_names after merging
        items_df["clean_item_name"] = items_df["item_name"].apply(clean_item_name)

        # 2) fix shops_df and trains_df duplicate data
        shops_df = ETLTransform.change_shop_ids(shops_df)
        shops_df = ETLTransform.change_shop_names(shops_df)
        shops_df.drop_duplicates(inplace=True)  # remove possible duplicates
        shops_df.reset_index(drop=True, inplace=True)
        train_df = ETLTransform.change_shop_ids(train_df)

        if return_aggregated:
            train_aggregated = (
                train_df.groupby(["date_block_num", "shop_id", "item_id"])
                .agg({"item_cnt_day": "sum", "item_price": "mean"})
                .reset_index()
            )

            train_aggregated.rename(
                columns={"item_cnt_day": "item_cnt_month"}, inplace=True
            )

        # 3) add year and month columns to received datasets
        train_df = ETLTransform.add_month_year_columns(train_df)

        if return_aggregated:
            train_aggregated = ETLTransform.add_month_year_columns(train_aggregated)

        # 4) drop duplicates of item_name

        # Create a DataFrame to keep only the first occurrence of each 'item_name'
        items_df_unique = items_df.drop_duplicates(
            subset="clean_item_name", keep="first"
        )
        # Merge the original DataFrame with the unique rows to find duplicates
        merged_df = items_df.merge(
            items_df_unique[["clean_item_name", "item_id"]],
            on="clean_item_name",
            how="left",
            suffixes=("", "_to_keep"),
        )
        # Filter out the rows where the 'item_id' is not the same as 'item_id_to_keep'
        duplicates_df = merged_df[merged_df["item_id"] != merged_df["item_id_to_keep"]]
        # Create the dictionary {item_id_that_im_deleting: item_id_to_replace_it_with}
        item_id_mapping = dict(
            zip(duplicates_df["item_id"], duplicates_df["item_id_to_keep"])
        )

        # transform dataframes using the item_id_mapping
        items_df = items_df.drop_duplicates(subset=["clean_item_name"])
        items_df.drop(columns=["clean_item_name"], inplace=True)
        train_df["item_id"] = train_df["item_id"].replace(item_id_mapping)

        if return_aggregated:
            train_aggregated["item_id"] = train_aggregated["item_id"].replace(
                item_id_mapping
            )
            # drop duplicates in train_aggregated to consider removed item_ids
            train_aggregated = train_aggregated.drop_duplicates(
                subset=["date_block_num", "shop_id", "item_id"]
            )

        # 5) merge train_df with items_df, categories_df, shops_df
        merged_train_df = ETLTransform.merge_df(
            train_df, items_df, categories_df, shops_df
        )

        if return_aggregated:
            merged_train_aggregated_df = ETLTransform.merge_df(
                train_aggregated, items_df, categories_df, shops_df
            )

        if return_aggregated:
            return merged_train_df, merged_train_aggregated_df

        return merged_train_df
