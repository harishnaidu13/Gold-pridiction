import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueSelector(BaseEstimator, TransformerMixin):
    """Drops features with more than 80% missing values."""

    def fit(self, X, y=None):
        missing_percent = pd.DataFrame(X).isnull().mean()
        self.selected_features_ = missing_percent[
            missing_percent <= 0.80
        ].index.tolist()
        return self

    def transform(self, X):
        return pd.DataFrame(X).loc[:, self.selected_features_]


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Drops one of each pair of features correlated above the threshold."""

    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr().abs()

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        self.selected_features_ = [
            col for col in upper.columns
            if not any(upper[col] > self.threshold)
        ]
        return self

    def transform(self, X):
        return pd.DataFrame(X).loc[:, self.selected_features_]