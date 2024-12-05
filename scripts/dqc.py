import seaborn as sns


class DataQualityCheck:
    @staticmethod
    def check_if_integers(df, column_name):
        is_integer = df["item_cnt_day"].apply(lambda x: x.is_integer())
        if is_integer.sum() == len(df):
            print("all item_cnt_day values are actually integers")
        else:
            print(
                f"""Only {is_integer.sum() / len(df) * 100}%
                of item_cnt_day values are actually integers"""
            )

    @staticmethod
    def plot_distribution(df, column_name):
        sns.histplot(df[column_name], kde=True)

    @staticmethod
    def check_duplicates(df):
        print("Number of duplicated rows:", len(df[df.duplicated(keep=False)]))

    @staticmethod
    def check_missing_data(df, return_missing_info=False):
        is_null = df.isna().sum()
        if is_null.sum() > 0:
            print("Some missing values found")
            if return_missing_info:
                print(is_null)
        else:
            print("No missing data found")

    @staticmethod
    def check_unique_ids(df, name_column: str, id_column: str):
        if df[name_column].nunique() == df[id_column].nunique():
            print("All names and ids are unique")
        else:
            print("Check {} and {} uniqueness".format(name_column, id_column))

    @staticmethod
    def find_outliers(df, column_name, return_outliers=False):
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_df = df[
            (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
        ]
        print("IQR:", iqr)
        print("Number of outliers:", len(outliers_df))
        print("Percent of outliers:", len(outliers_df) / len(df) * 100)
        print(f"Upper bound value: {upper_bound}, lower bound value: {lower_bound}")
        print(f"Min value: {df[column_name].min()}, Max value: {df[column_name].max()}")

        if return_outliers:
            return outliers_df

    @staticmethod
    def check_negative_values(df, column_name, return_negatives=False):
        negatives = df[df[column_name] < 0]
        if len(negatives) > 0:
            print(f"{len(negatives) / len(df) * 100} percent of values are negative")
            if return_negatives:
                return negatives
        else:
            print("No negative values found")
