import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Importy z Twoich modułów (zakładam, że są w folderze src - wyjaśnienie niżej)
from src.data_loader import load_dataset, get_current_eur_pln_rate
from src.preprocessor import run_preprocessing_pipeline, engineer_features, select_features, filter_data
from src.transformers import GroupedImputer
from src.model_utils import get_xgb_pipeline, calculate_metrics
from src.tuning import run_optuna_optimization

# --- KONFIGURACJA ---
DATA_PATH = 'data/Car_sale_ads.csv'
MODEL_SAVE_PATH = 'models/XGBoost_model.joblib'
TUNE_HYPERPARAMETERS = True  # Zmień na False, jeśli chcesz pominąć Optunę i użyć domyślnych/zapisanych parametrów
REMOVE_OUTLIERS = True
RANDOM_STATE = 42

def main():
    # 1. Ładowanie danych
    print(f"[1/6] Loading dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}. Please check your path.")
        
    df = load_dataset(DATA_PATH)
    eur_rate = get_current_eur_pln_rate()

    # 2. Preprocessing i Feature Engineering
    print("[2/6] Running preprocessing and feature engineering...")
    df = run_preprocessing_pipeline(df, eur_rate)
    df = engineer_features(df)
    df = select_features(df)
    
    if REMOVE_OUTLIERS:
        df = filter_data(df, remove_outliers=True)

    print(f"Data shape after processing: {df.shape}")

    # 3. Podział na zbiór treningowy i testowy
    print("[3/6] Splitting data...")
    X = df.drop('price_PLN', axis=1)
    y = df['price_PLN']

    # Transformacja logarytmiczna celu (log1p) - kluczowe dla cen
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.3, random_state=RANDOM_STATE
    )

    # 4. Uzupełnianie braków (Smart Imputation)
    print("[4/6] Handling missing values (Smart Imputation)...")
    # Definiujemy imputery dla konkretnych kolumn
    imputers = [
        GroupedImputer(group_col='Vehicle_model', target_col='Power_HP', strategy='median'),
        GroupedImputer(group_col='Vehicle_model', target_col='Displacement_cm3', strategy='median'),
        GroupedImputer(group_col='Vehicle_model', target_col='Mileage_km', strategy='median'),
        GroupedImputer(group_col='Vehicle_model', target_col='Drive', strategy='most_frequent')
    ]

    # Fit na train, Transform na train i test (zapobiega wyciekowi danych)
    for imputer in imputers:
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

    # 5. Modelowanie
    best_params = {}
    if TUNE_HYPERPARAMETERS:
        print("[5/6] Tuning hyperparameters with Optuna (this may take a while)...")
        best_params = run_optuna_optimization(X_train, y_train_log, n_trials=20)
        # Dodajemy prefiks wymagany przez Pipeline
        pipeline_params = {f'model__{k}': v for k, v in best_params.items()}
    else:
        print("[5/6] Skipping tuning, using default/hardcoded parameters.")
        # Tutaj możesz wpisać parametry, które Optuna znalazła wcześniej
        pipeline_params = {
            'model__n_estimators': 500,
            'model__learning_rate': 0.05,
            'model__max_depth': 6
        }

    print("Training final XGBoost model...")
    pipeline = get_xgb_pipeline()
    pipeline.set_params(**pipeline_params)
    pipeline.fit(X_train, y_train_log)

    # 6. Ewaluacja i Zapis
    print("[6/6] Evaluating and saving...")
    
    # Predykcja na zbiorze testowym
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)     # Odwracamy logarytm
    y_test_true = np.expm1(y_test_log) # Odwracamy logarytm dla prawdziwych danych

    # Liczenie metryk
    metrics = calculate_metrics(y_test_true, y_pred, model_name='XGBoost Final')
    
    # Zapis modelu
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"Model successfully saved to: {MODEL_SAVE_PATH}")
    print("Done!")

if __name__ == "__main__":
    main()