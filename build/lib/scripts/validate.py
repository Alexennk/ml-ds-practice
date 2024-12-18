import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler


class TimeSeriesSplit(BaseCrossValidator):
    """
    The splits are created basing on the 'date_block_num' feature

    Attributes:
        n_splits: the number of train:test pairs to return
        method: 'expanding' or 'sliding'
        trait_start: first month number to be included in splits
        random_state: random seed
    """

    def __init__(
        self, n_splits=5, method="expanding", random_state=None, train_start=0
    ):
        self.n_splits = n_splits
        self.method = method
        self.train_start = train_start
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_months = X["date_block_num"].max() + 1
        method = self.method

        if method == "expanding":

            for i in range(self.n_splits, 0, -1):
                train_idx = (X["date_block_num"] < n_months - i) & (
                    X["date_block_num"] >= self.train_start
                )
                test_idx = X["date_block_num"] == n_months - i

                yield train_idx, test_idx

        elif method == "sliding":

            m_in_split = (
                n_months // self.n_splits
            )  # number of months in a single split (perhaps, except for the last one)

            for i in range(1, self.n_splits):
                train_idx = (X["date_block_num"] < i * m_in_split) & (
                    X["date_block_num"] >= m_in_split * (i - 1)
                )
                test_idx = X["date_block_num"] == i * m_in_split

                yield train_idx, test_idx

            train_idx = (X["date_block_num"] < n_months - 1) & (
                X["date_block_num"] >= m_in_split * (self.n_splits - 1)
            )  # all indexes left go to the last block
            test_idx = X["date_block_num"] == n_months - 1

            yield train_idx, test_idx

        else:

            raise ValueError("'method' parameter should be 'expanding' or 'sliding'")


class ModelTrainer:
    @staticmethod
    def train_model(
        X,
        y,
        model,
        apply_scaling=False,
        cv_method="expanding",
        cv_n_splits=5,
        train_start=0,
        eval_set=False,
        print_splits_scores=True,
        return_scores=False,
        return_model=False,
    ):
        """
        Returns on of these:
            - None
            - model
            - scores
            - model, scores
        """

        scores = []
        tscv = TimeSeriesSplit(
            n_splits=cv_n_splits, method=cv_method, train_start=train_start
        )

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if apply_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if eval_set:
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = root_mean_squared_error(y_test, y_pred)
            scores.append(score)

            if print_splits_scores:
                print(f"{len(scores)} split RMSE: {score:.2f}\n")

        print(f"Average RMSE: {np.mean(scores):.2f}")

        if return_scores or return_model:
            to_return = []
            if return_model:
                to_return.append(model)
            if return_scores:
                to_return.append(scores)

            if len(to_return) == 1:
                return to_return[0]
            return tuple(to_return)


def get_submission(model, test, scaler=None, rounding=False, submission_tag=""):
    test_id = test["ID"]
    test = test.drop(["ID"], axis=1)

    if scaler is not None:
        test = scaler.transform(test)

    y_pred = model.predict(test)

    if rounding:
        y_pred = y_pred.round()

    y_pred = y_pred.clip(0, 20)

    submission = pd.DataFrame({"ID": test_id, "item_cnt_month": y_pred})
    submission["ID"] = submission["ID"].astype(int)
    submission.to_csv("submission_" + submission_tag + ".csv", index=False)
