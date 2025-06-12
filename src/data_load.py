import pandas as pd
import os

def load_data(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist.")
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
