import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

def get_preprocessor_scaled():
    """Zwraca preprocessor dla modeli liniowych/sieci neuronowych."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    return ColumnTransformer([
        ('num', num_pipeline, make_column_selector(dtype_include='number')),
        ('cat', cat_pipeline, make_column_selector(dtype_include=['object', 'category']))
    ])

def get_preprocessor_tree():
    """Zwraca preprocessor dla modeli drzewiastych (XGBoost, Random Forest)."""
    num_pipeline = SimpleImputer(strategy='median')

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    return ColumnTransformer([
        ('num', num_pipeline, make_column_selector(dtype_include='number')),
        ('cat', cat_pipeline, make_column_selector(dtype_include=['object', 'category']))
    ])

def calculate_metrics(y_true, y_pred, model_name='Model'):
    """Zwraca słownik i DataFrame z metrykami."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    metrics = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    print(f"--- Metrics for {model_name} ---")
    print(metrics)
    
    return metrics

def get_ridge_pipeline(alpha=1.0):
    return Pipeline([
        ('preprocessor', get_preprocessor_scaled()),
        ('model', Ridge(alpha=alpha))
    ])

def get_rf_pipeline(params=None):
    model = RandomForestRegressor(random_state=42)
    if params:
        model.set_params(**params)
    
    return Pipeline([
        ('preprocessor', get_preprocessor_tree()),
        ('model', model)
    ])

def get_xgb_pipeline(params=None):
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    if params:
        model.set_params(**params)
        
    return Pipeline([
        ('preprocessor', get_preprocessor_tree()),
        ('model', model)
    ])