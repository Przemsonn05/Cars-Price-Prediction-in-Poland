import pandas as pd
import numpy as np

def convert_currency(df: pd.DataFrame, eur_rate: float) -> pd.DataFrame:
    """Tworzy kolumnę price_PLN na podstawie kursu walut."""
    df = df.copy() 
    df['price_PLN'] = df.apply(
        lambda row: row['Price'] * eur_rate if row['Currency'] == 'EUR' else row['Price'],
        axis=1
    )
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wykonuje czyszczenie danych: usuwanie kolumn, NaN i wierszy.
    """
    cols_to_drop = ['CO2_emissions', 'First_registration_date', 
                    'Vehicle_version', 'Vehicle_generation']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    df['First_owner'] = df['First_owner'].fillna('Unknown')
    df['Origin_country'] = df['Origin_country'].fillna('Unknown')

    df = df.dropna(subset=['Transmission'])
    
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje nowe cechy, np. rok publikacji."""
    df['Offer_publication_date'] = pd.to_datetime(
        df['Offer_publication_date'], format='%d/%m/%Y'
    )
    df['Year_publication'] = df['Offer_publication_date'].dt.year
    
    return df

def run_preprocessing_pipeline(df: pd.DataFrame, eur_rate: float) -> pd.DataFrame:
    """Funkcja sterująca (wrapper), która uruchamia wszystkie kroki po kolei."""
    df = convert_currency(df, eur_rate)
    df = clean_data(df)
    df = feature_engineering(df)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy nowe cechy (Feature Engineering)."""
    df = df.copy()
    
    if 'Year_publication' in df.columns:
        df['Vehicle_age'] = df['Year_publication'] - df['Production_year']
    else:
        current_year = pd.Timestamp.now().year
        df['Vehicle_age'] = current_year - df['Production_year']

    df['Annual_mileage'] = df['Mileage_km'] / (df['Vehicle_age'] + 1e-6)

    def get_age_category(age):
        if age <= 3: return 'New'
        elif age <= 7: return 'Young'
        elif age <= 12: return 'Middle_Aged'
        else: return 'Old'
    
    df['Age_category'] = df['Vehicle_age'].apply(get_age_category)

    popular_colors = ['white', 'black', 'blue', 'gray', 'red', 'brown', 'green', 'silver']
    df['Is_popular_color'] = df['Colour'].str.lower().isin(popular_colors).astype(int)

    premium_brands = ['Abarth', 'Alfa Romeo', 'Audi', 'BMW', 'Mercedes-Benz', 'Porsche', 
                      'Lexus', 'Jaguar', 'Land Rover', 'Volvo', 'Infiniti', 'Tesla', 'Mini']
    df['Is_premium_car'] = df['Vehicle_brand'].isin(premium_brands).astype(int)

    df['Power_per_liter'] = df['Power_HP'] / (df['Displacement_cm3'] / 1000)
    
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa zbędne kolumny przed treningiem."""
    cols_drop = ['Beand_popularity', 'Brand_popularity', 'Index', 
                 'Doors_number', 'Price', 'Currency', 'Year_publication', 
                 'Offer_publication_date'] 
    return df.drop(columns=cols_drop, errors='ignore')

def filter_data(df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
    """
    Filtruje dane: usuwa skrajne ceny i zbyt stare roczniki (opcjonalnie).
    """
    df = df.copy()
    
    df = df[df['price_PLN'] > 2000]

    if remove_outliers:
        price_threshold = df['price_PLN'].quantile(0.99)
        df = df[df['price_PLN'] < price_threshold]
        print(f"Removed outliers with price > {price_threshold:.2f} PLN")
        
    return df