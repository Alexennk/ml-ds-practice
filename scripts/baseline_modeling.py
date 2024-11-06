import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


class TimeSeriesSplit(BaseCrossValidator):
    """
    The splits are created basing on the 'date_block_num' feature

    Attributes:
        n_splits: the number of train:test pairs to return
        method: 'expanding' or 'sliding'
        trait_start: first month number to be included in splits
        random_state: random seed
    """

    def __init__(self, n_splits=5, method='expanding', random_state=None, train_start=0):
        self.n_splits = n_splits
        self.method = method
        self.train_start = train_start
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_months = X['date_block_num'].max() + 1
        method = self.method
        
        if method == 'expanding':

            for i in range(self.n_splits, 0, -1):
                train_idx = (X['date_block_num'] < n_months - i) & (X['date_block_num'] >= self.train_start)
                test_idx = X['date_block_num'] == n_months - i

                yield train_idx, test_idx
        
        elif method == 'sliding':

            m_in_split = n_months // self.n_splits # number of months in a single split (perhaps, except for the last one)

            for i in range(1, self.n_splits):
                train_idx = (X['date_block_num'] < i * m_in_split) & (X['date_block_num'] >= m_in_split * (i - 1))
                test_idx = X['date_block_num'] == i * m_in_split

                yield train_idx, test_idx
            
            train_idx = (X['date_block_num'] < n_months - 1) & (X['date_block_num'] >= m_in_split * (self.n_splits - 1)) # all indexes left go to the last block
            test_idx = X['date_block_num'] == n_months - 1

            yield train_idx, test_idx

        else:
            
            raise ValueError("'method' parameter should be 'expanding' or 'sliding'")


class TrainModels:
    @staticmethod
    def train_linear_regression(X, y, cv_method='expanding', return_scores=False, return_model=False, cv_n_splits=5):
        
        """
        Returns on of these: 
            - None
            - model
            - scores
            - model, scores
        """

        scores = []
        tscv = TimeSeriesSplit(n_splits=cv_n_splits, method=cv_method)

        for train_idx, test_idx in tscv.split(X):
            X_new = X.copy()
            X_new.drop('date_block_num', axis=1, inplace=True)
            X_train, X_test = X_new[train_idx], X_new[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = LinearRegression()

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)

            print(f"{len(scores)} split RMSE: {score:.2f}\n")
        
        print(f"Average RMSE: {np.mean(scores):.2f}")

        if return_scores or return_model:
            to_return = []
            if return_model: to_return.append(model)
            if return_scores: to_return.append(scores)
            return tuple(to_return)


    @staticmethod
    def train_xgboost(X, y, cv_method='expanding', return_scores=False, return_model=False, verbose=False, cv_n_splits=5,
                    n_estimators=1000, max_depth=7, learning_rate=0.05, early_stopping_rounds=30, subsample=0.8, colsample_bytree=0.8):
        
        """
        Returns on of these: 
            - None
            - model
            - scores
            - model, scores
        """

        scores = []
        tscv = TimeSeriesSplit(n_splits=cv_n_splits, method=cv_method)

        for train_idx, test_idx in tscv.split(X):
            X_new = X.copy()
            X_new.drop('date_block_num', axis=1, inplace=True)
            X_train, X_test = X_new[train_idx], X_new[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = XGBRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=learning_rate,    
                eval_metric="rmse",
                early_stopping_rounds=early_stopping_rounds,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )

            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=verbose)

            y_pred = model.predict(X_test)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)

            print(f"{len(scores)} split RMSE: {score:.2f}\n")
        
        print(f"Average RMSE: {np.mean(scores):.2f}")

        if return_scores or return_model:
            to_return = []
            if return_model: to_return.append(model)
            if return_scores: to_return.append(scores)
            return tuple(to_return)