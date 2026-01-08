# transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class GroupedImputer(BaseEstimator, TransformerMixin):
    """
    Uzupełnia braki danych (NaN) używając mediany/mody z grupy (np. Vehicle_model).
    Jeśli w grupie też są braki, używa globalnej mediany.
    """
    def __init__(self, group_col, target_col, strategy='median'):
        self.group_col = group_col
        self.target_col = target_col
        self.strategy = strategy
        self.mapping_ = None
        self.global_fill_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.strategy == 'median':
            self.global_fill_ = X[self.target_col].median()
            self.mapping_ = X.groupby(self.group_col)[self.target_col].median()
        elif self.strategy == 'most_frequent':
            self.global_fill_ = X[self.target_col].mode()[0]
            def get_mode(x):
                m = x.mode()
                return m.iloc[0] if not m.empty else np.nan
            self.mapping_ = X.groupby(self.group_col)[self.target_col].apply(get_mode)
        
        return self

    def transform(self, X):
        X = X.copy()
        
        filled_values = X[self.group_col].map(self.mapping_)
        
        X[self.target_col] = X[self.target_col].fillna(filled_values)
        
        X[self.target_col] = X[self.target_col].fillna(self.global_fill_)
        
        return X