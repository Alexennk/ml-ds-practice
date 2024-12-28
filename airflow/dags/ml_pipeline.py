from airflow.models import DAG
from airflow.decorators import task
import json
from datetime import datetime
from neptune_airflow import NeptuneLogger
import neptune.integrations.sklearn as npt_utils
from scripts.validate import TimeSeriesSplit
from scripts.etl import transform_df_types, ETLTransform
from scripts.modeling import PredictionVisualizer
from scripts.feature_extr import FeatureExtractionLayer

import numpy as np
import pandas as pd
import gdown
import pickle
from sklearn.metrics import root_mean_squared_error


def log_to_json(data, file_path="neptune_logs.json"):
    try:
        with open(file_path, "r") as file:
            logs = json.load(file)
    except FileNotFoundError:
        logs = {}

    logs.update(data)

    with open(file_path, "w") as file:
        json.dump(logs, file, indent=4)


def log_base_info():
    print('---- "log_base_info" component ----')

    log_to_json(
        {
            "log_base_info/test_pypi_package_version": "1.0.5",
            "log_base_info/algorithm": "Random Forest Regressor",
            "log_base_info/cv_parameters": {"n_splits": 1, "train_start": 0},
        }
    )


def prepare_data():
    print('---- "prepare_data" component ----')

    train_df = pd.read_csv(
        "https://drive.google.com/uc?id=13TAaFKeaSsZDDujuRsi7Gzxqf8M7hXBc"
    )
    items_df = pd.read_csv(
        "https://drive.google.com/uc?id=1uhWrt0jIUe5OgrT8aaicp62NpwWjj41O"
    )
    categories_df = pd.read_csv(
        "https://drive.google.com/uc?id=1DI0fw_uMkn1ME9j4S6HVrxBPO4niWHjA"
    )
    shops_df = pd.read_csv(
        "https://drive.google.com/uc?id=1dm6v3QXEHNFluDJ5AackHDBd5kvJQ6Da"
    )

    print("---- merging dataframes ----")

    # save merged and aggregated merged dataframes
    merged, aggregated_merged = ETLTransform.transform(
        train_df, items_df, categories_df, shops_df, return_aggregated=True
    )

    # saving merged and aggregated merged dataframes
    merged.to_csv("merged_train.csv", index=False)
    aggregated_merged.to_csv("merged_train_aggregated.csv", index=False)

    log_to_json(
        {
            "prepare_data/train_df_shape": str(merged.shape),
            "prepare_data/aggregated_train_df_shape": str(aggregated_merged.shape),
        }
    )


def extract_features():
    print('---- "extract_features" component ----')

    train_df = pd.read_csv("merged_train.csv")
    aggregated_train_df = pd.read_csv("merged_train_aggregated.csv")

    train_df = transform_df_types(train_df)
    aggregated_train_df = transform_df_types(aggregated_train_df)

    print("---- extracting features ----")

    aggregated_lagged = FeatureExtractionLayer.train_transform(
        train_df, aggregated_train_df, for_airflow=True
    )

    print("---- saving the dataframe ewith extracted features ----")

    aggregated_lagged.to_csv("result_train.csv", index=False)

    log_to_json({"extract_features/result_train_shape": str(aggregated_lagged.shape)})


def load_model():
    print('---- "load_model (Random Forest) from Google Drive" component ----')

    drive_url = "https://drive.google.com/uc?id=1YuHIOaqM_O6T-zCgUnDq8E-BRhMFh5qI"
    output_path = "best_rfr.pkl"

    # download and save the model
    gdown.download(drive_url, output_path, quiet=False)

    print(
        """---- " Random Forest Regressor downloaded from
        Google Drive and saved locally ----"""
    )

    parameters = {
        "max_depth": 18,
        "max_features": "sqrt",
        "min_samples_leaf": 7,
        "min_samples_split": 8,
        "n_estimators": 200,
    }
    log_to_json(
        {
            "load_model/model_parameters": parameters,
            "load_model/serialized_model_url": drive_url,
        }
    )


def predict(logger: NeptuneLogger, **context):
    print('---- "predict" component ----')

    train_df = pd.read_csv("result_train.csv")
    train_df = transform_df_types(train_df)

    train_df = train_df.select_dtypes(include=np.number)
    y = train_df["item_cnt_month"]
    X = train_df.drop(
        ["item_cnt_month", "item_price", "revenue", "avg_item_price", "month", "year"],
        axis=1,
        inplace=False,
    )

    print("---- the dataset is split into train and test ----")

    tscv = TimeSeriesSplit(n_splits=1, method="expanding", train_start=0)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print("Train, test shapes:")
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        print("\n")

    print("---- Random Forest Model loaded from local storage ----")

    rfr = pickle.load(open("best_rfr.pkl", "rb"))

    print("---- Random Forest Model prediction calculating ----")

    y_pred = rfr.predict(X_test)
    print("RMSE:", root_mean_squared_error(y_test, y_pred))

    log_to_json(
        {
            "predict/scores/train_rmse": str(
                root_mean_squared_error(y_train, rfr.predict(X_train))
            ),
            "predict/scores/test_rmse": str(root_mean_squared_error(y_test, y_pred)),
        }
    )

    with logger.get_task_handler_from_context(
        context=context, log_context=True
    ) as handler:
        handler["visuals/feature_importances"] = (
            npt_utils.create_feature_importance_chart(rfr, X_train, y_train)
        )

    visuals_to_json = {
        "model_performance_sc_plot": PredictionVisualizer.model_performance_sc_plot(
            y_pred, y_test, "Best parameters RFR", for_neptune=True
        ),
        "plot_predictions_distribution":
        PredictionVisualizer.plot_predictions_distribution(
            y_test, y_pred, model_name="Best parameters RFR", for_neptune=True
        ),
        "plot_residuals": PredictionVisualizer.plot_residuals(
            y_test, y_pred, model_name="Best parameters RFR", for_neptune=True
        ),
    }
    log_to_json(visuals_to_json, "visual_logs.json")

    print("---- Random Forest Model prediction visualizations ready ----")


def log_experiment_results(logger: NeptuneLogger, **context):
    print('---- "log_experiment_results" component ----')

    with open("neptune_logs.json", "r") as file:
        logs = json.load(file)
    with logger.get_task_handler_from_context(
        context=context, log_context=True
    ) as handler:
        for key in logs.keys():
            handler[key] = logs[key]

        with open("visual_logs.json", "r") as file:
            logs = json.load(file)
        for key in logs.keys():
            handler[f"visuals/{key}"].upload(logs[key])


def get_neptune_token_from_variable() -> "dict[str, str]":
    """Reads NEPTUNE_API_TOKEN and NEPTUNE_PROJECT from Airflow variables.

    Returns:
        dict[str,str]: A dict containing the NEPTUNE_API_TOKEN and NEPTUNE_PROJECT
    """
    return {
        "api_token": """eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJs
            IjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YmY3NjQ0ZC0wMzM1LTQ2Mj
                AtYTE4Ny0wZmE2MmYyZWI1ZTUifQ==""",
        "project": "alexennk/innowise-neptune",
    }


with DAG(
    dag_id="ml_pipeline_innowise",
    schedule_interval=None,
    start_date=datetime(2024, 12, 26),
    catchup=False,
) as dag:

    @task(task_id="log_base_info")
    def task_log_base_info():
        return log_base_info()

    @task(task_id="prepare_data")
    def task_prepare_data():
        return prepare_data()

    @task(task_id="extract_features")
    def task_extract_features():
        return extract_features()

    @task(task_id="load_model")
    def task_load_model():
        return load_model()

    @task(task_id="predict")
    def task_predict(**context):
        logger = NeptuneLogger(**get_neptune_token_from_variable())
        return predict(logger, **context)

    @task(task_id="log_experiment_results")
    def task_log_experiment_results(**context):
        logger = NeptuneLogger(**get_neptune_token_from_variable())
        return log_experiment_results(logger, **context)

    (
        task_log_base_info()
        >> task_prepare_data()
        >> task_extract_features()
        >> task_load_model()
        >> task_predict()
        >> task_log_experiment_results()
    )
