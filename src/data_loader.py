import pandas as pd
import requests

def load_dataset(filepath: str) -> pd.DataFrame:
    """Wczytuje surowy zbiór danych."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise

def get_current_eur_pln_rate(default_rate: float = 4.30) -> float:
    """Pobiera aktualny kurs EUR/PLN z API NBP."""
    url = "http://api.nbp.pl/api/exchangerates/rates/a/eur/?format=json"
    
    try:
        response = requests.get(url, timeout=5) 
        response.raise_for_status()
        data = response.json()
        
        rate = data['rates'][0]['mid']
        print(f"Current exchange rate loaded from NBP: {rate}")
        return rate
        
    except Exception as e:
        print(f"Failed to load exchange rate ({e}). Using default value: {default_rate}")
        return default_rate