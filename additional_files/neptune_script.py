import neptune
import neptune.integrations.sklearn as npt_utils
import subprocess  # for logging dvc info
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from scripts.validate import TimeSeriesSplit
from scripts.etl import transform_df_types
from scripts.modeling import PredictionVisualizer

# init neptune
run = neptune.init_run(monitoring_namespace="monitoring")

# log dvc info
dvc_version = subprocess.getoutput("dvc version")
run["dvc/version"] = dvc_version

git_hash = subprocess.getoutput("git rev-parse HEAD")
run["dvc/git_commit"] = git_hash

dvc_remotes = subprocess.getoutput("dvc remote list")
run["dvc/remotes"] = dvc_remotes

# log basic parameters
run["algorithm"] = "Random Forest Regressor"
run["cv/parameters"] = {"n_splits": 1, "train_start": 0}

# load data
train_df = pd.read_csv("data/result_train.csv")
train_df = transform_df_types(train_df)

train = train_df.select_dtypes(include=np.number)
y = train["item_cnt_month"]
X = train.drop(
    ["item_cnt_month", "item_price", "revenue", "avg_item_price", "month", "year"],
    axis=1,
    inplace=False,
)

# split into train and test
tscv = TimeSeriesSplit(n_splits=1, method="expanding", train_start=0)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

# load model
parameters = {
    "max_depth": 18,
    "max_features": "sqrt",
    "min_samples_leaf": 7,
    "min_samples_split": 8,
    "n_estimators": 200,
}
rfr = RandomForestRegressor(**parameters, n_jobs=-1)
run["model/parameters"] = parameters

# train model
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
run["scores/train/rmse"] = root_mean_squared_error(y_train, rfr.predict(X_train))
run["scores/eval/rmse"] = root_mean_squared_error(y_test, y_pred)
run["summary"] = npt_utils.create_regressor_summary(
    rfr, X_train, X_test, y_train, y_test
)

run["serialized_model"].upload("models/best_rfr.pkl")

# add other logs

run["visuals/feature_importances"] = npt_utils.create_feature_importance_chart(
    rfr, X_train, y_train
)

name = PredictionVisualizer.model_performance_sc_plot(
    y_pred, y_test, "Best parameters RFR", for_neptune=True
)
run["visuals/model_performance_sc_plot"].upload(name)

name = PredictionVisualizer.plot_predictions_distribution(
    y_test, y_pred, model_name="Best parameters RFR", for_neptune=True
)
run["visuals/predictions_distribution"].upload(name)

PredictionVisualizer.plot_residuals(
    y_test, y_pred, model_name="Best parameters RFR", for_neptune=True
)
run["visuals/residuals_plot"].upload(name)

# stop run
run.stop()
