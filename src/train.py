# train.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split

# Nasze moduły
from data_loader import load_dataset, get_current_eur_pln_rate
from preprocessor import run_preprocessing_pipeline, engineer_features, select_features, filter_data
from transformers import GroupedImputer
from model_utils import get_ridge_pipeline, get_rf_pipeline, get_xgb_pipeline, calculate_metrics
from tuning import run_optuna_optimization

print("1. Loading Data...")
df = load_dataset('../data/Car_sale_ads.csv')
rate = get_current_eur_pln_rate()
df = run_preprocessing_pipeline(df, rate)
df = engineer_features(df)
df = select_features(df)

df = filter_data(df, remove_outliers=True)

print("2. Splitting Data...")
X = df.drop('price_PLN', axis=1)
y = df['price_PLN']

y_log = np.log1p(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.3, random_state=42
)

print("3. Handling Missing Values...")
imputers = [
    GroupedImputer(group_col='Vehicle_model', target_col='Power_HP', strategy='median'),
    GroupedImputer(group_col='Vehicle_model', target_col='Displacement_cm3', strategy='median'),
    GroupedImputer(group_col='Vehicle_model', target_col='Mileage_km', strategy='median'),
    GroupedImputer(group_col='Vehicle_model', target_col='Drive', strategy='most_frequent')
]

for imputer in imputers:
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

print("4. Tuning XGBoost with Optuna...")
best_params = run_optuna_optimization(X_train, y_train_log, n_trials=20) 

pipeline_best_params = {f'model__{k}': v for k, v in best_params.items()}

print("5. Training Final Model...")
final_model = get_xgb_pipeline()
final_model.set_params(**pipeline_best_params)
final_model.fit(X_train, y_train_log)

print("6. Evaluation...")
y_pred_log = final_model.predict(X_test)
y_pred = np.expm1(y_pred_log) 
y_test_true = np.expm1(y_test_log)

metrics = calculate_metrics(y_test_true, y_pred, model_name='XGBoost Optuna')

os.makedirs('../models', exist_ok=True)
save_path = '../models/XGBoost_optimized.joblib'
joblib.dump(final_model, save_path)
print(f"Model saved to {save_path}")