import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from src.model_utils import get_xgb_pipeline

def run_optuna_optimization(X, y_log, n_trials=50):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1, log=True)
        }
        
        pipeline_params = {f'model__{k}': v for k, v in params.items()}
        pipeline = get_xgb_pipeline()
        pipeline.set_params(**pipeline_params)

        scores = cross_val_score(
            pipeline, X, y_log, cv=3, 
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        return -scores.mean()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best params found:", study.best_params)
    return study.best_params