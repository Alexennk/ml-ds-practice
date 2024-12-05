import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_barpot_boxplot(df, column_name, figsize=(14, 7), leave=None):
    # First Plot Data
    column_aggregated = df.groupby([column_name], observed=False)[
        "item_cnt_month"
    ].mean()
    column_aggregated = pd.DataFrame(column_aggregated)
    column_aggregated.reset_index(inplace=True)
    column_aggregated = column_aggregated.sort_values(
        ["item_cnt_month"], ascending=False
    )
    if leave is not None:
        column_aggregated = column_aggregated[:leave]

    # Second Plot Data
    var = column_name
    data = pd.concat([df["item_cnt_month"], df[var]], axis=1)

    # Create subplots in one row with two columns
    fig, axes = plt.subplots(
        1, 2, figsize=figsize
    )  # Adjust the figsize as needed for better appearance

    # First plot: Barplot
    sns.barplot(
        x=column_aggregated["item_cnt_month"],
        y=column_aggregated[column_name],
        orient="h",
        order=column_aggregated[column_name],
        ax=axes[0],
    )
    axes[0].set_title(column_name + " Sales Comparison")
    axes[0].set_xlabel("Average Sales Amount")
    axes[0].set_ylabel(column_name)

    # Second plot: Boxplot
    sns.boxplot(
        x="item_cnt_month",
        y=var,
        data=data,
        orient="h",
        order=column_aggregated[column_name],
        ax=axes[1],
    )
    axes[1].set_title("Box Plots for " + column_name)
    axes[1].set_xlabel("Item Count per Month")
    axes[1].set_ylabel("")

    # Display the plots
    plt.tight_layout()
    plt.show()


def plot_feature_comparison(
    df,
    feature_to_compare,
    feature_to_aggregate,
    top_objects=None,
    left_lim_objects=0,
    right_lim_objects=5,
    agg_operation="sum",
    limit_city=None,
    fig_size=(6, 4),
    legend_size=8,
    legend_loc=2,
    ylim=None,
):
    if top_objects is None:
        column_aggregated = df.groupby([feature_to_aggregate], observed=False)[
            feature_to_compare
        ].sum()
        column_aggregated = pd.DataFrame(column_aggregated)
        column_aggregated.reset_index(inplace=True)
        column_aggregated = column_aggregated.sort_values(
            [feature_to_compare], ascending=False
        )

        top_objects = column_aggregated[feature_to_aggregate][
            left_lim_objects:right_lim_objects
        ].values

    if limit_city is not None:
        df = df[df["shop_city"] == limit_city]
    column_month_aggregated = df[df[feature_to_aggregate].isin(top_objects)]
    column_month_aggregated = column_month_aggregated.groupby(
        [feature_to_aggregate, "date_block_num"], observed=False
    )[feature_to_compare].agg(agg_operation)
    column_month_aggregated = pd.DataFrame(column_month_aggregated)
    column_month_aggregated.reset_index(inplace=True)

    plt.figure(figsize=fig_size)
    sns.lineplot(
        x="date_block_num",
        y=feature_to_compare,
        hue=feature_to_aggregate,
        data=column_month_aggregated,
    )
    plt.title(feature_to_compare + " Comparison")
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.legend(loc=legend_loc, prop={"size": legend_size})
    plt.xlabel("Month")
    plt.ylabel("Monthly " + feature_to_compare)
