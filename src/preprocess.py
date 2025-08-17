import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------
# File paths
# ----------------------
RAW_DATA_PATH = "data/raw/insurance_data.txt"
PROCESSED_DATA_PATH = "data/processed/processed_insurance_data.csv"

# ----------------------
# Load data
# ----------------------
def load_data(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, sep="|", engine="python", encoding="utf-8", on_bad_lines="skip")
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

# ----------------------
# Clean and preprocess
# ----------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop high-missing columns
    drop_cols = ['CrossBorder', 'Citizenship', 'MaritalStatus', 'Language', 'CustomValueEstimate', 'NumberOfVehiclesInFleet']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    logging.info(f"Dropped columns due to high missing values or low utility: {drop_cols}")

    # Fill moderate missing categorical columns
    for col in ['Bank', 'Gender', 'AccountType']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # Fill risk flag columns
    for col in ['Converted', 'WrittenOff', 'Rebuilt', 'NewVehicle']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # Drop rows missing critical vehicle info
    critical_vehicle_cols = ['VehicleType', 'make', 'Model', 'VehicleIntroDate', 'bodytype', 'CapitalOutstanding']
    df.dropna(subset=[col for col in critical_vehicle_cols if col in df.columns], inplace=True)

    # Convert dates
    for col in ['TransactionMonth', 'VehicleIntroDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert binary columns to boolean
    binary_cols = ['NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({'yes': True, 'no': False, 'unknown': False, '': False, 'nan': False})
            df[col] = df[col].fillna(False).astype(bool)

    # Convert numeric columns
    if 'CapitalOutstanding' in df.columns:
        df['CapitalOutstanding'] = (
            df['CapitalOutstanding'].astype(str)
            .str.replace(',', '')
            .str.extract(r'(\d+\.?\d*)')[0]
        )
        df['CapitalOutstanding'] = pd.to_numeric(df['CapitalOutstanding'], errors='coerce')

    # Create unique RecordID
    df['RecordID'] = df.index
    cols = ['RecordID'] + [col for col in df.columns if col != 'RecordID']
    df = df[cols]

    # Normalize categorical columns
    df['Gender'] = df['Gender'].str.strip().str.lower().replace({'not specified': 'unknown', '': 'unknown'})
    if 'Title' in df.columns:
        df['Title'] = df['Title'].str.strip().str.lower()
    if 'Bank' in df.columns:
        df['Bank'] = df['Bank'].str.strip().str.lower().replace({'firstrand bank': 'first national bank'})
    for col in ['CoverCategory', 'AccountType']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    # Infer missing Gender from Title
    def infer_gender(row):
        if row['Gender'] == 'unknown' and 'Title' in row:
            if row['Title'] == 'mr':
                return 'male'
            elif row['Title'] in ['mrs', 'miss', 'ms']:
                return 'female'
        return row['Gender']

    df['Gender'] = df.apply(infer_gender, axis=1)

    # Drop rows where gender is still unknown
    df = df[df['Gender'] != 'unknown']

    # Remove negative premiums or claims
    if 'TotalPremium' in df.columns:
        df = df[df['TotalPremium'] >= 0]
    if 'TotalClaims' in df.columns:
        df = df[df['TotalClaims'] >= 0]

    logging.info(f"Data cleaning complete. Final shape: {df.shape}")
    return df

# ----------------------
# Feature engineering
# ----------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    current_year = datetime.now().year

    # Transaction date
    if 'transactiondate' in df.columns:
        new_cols['transaction_date'] = pd.to_datetime(df['transactiondate'], errors='coerce')
    elif 'transactionmonth' in df.columns:
        new_cols['transaction_date'] = pd.to_datetime(df['transactionmonth'], errors='coerce')
    else:
        new_cols['transaction_date'] = pd.Series([pd.NaT]*len(df), index=df.index)

    # Fill missing transaction_date with TransactionMonth
    if 'transactionmonth' in df.columns:
        missing_dates = new_cols['transaction_date'].isna()
        new_cols['transaction_date'].loc[missing_dates] = pd.to_datetime(df.loc[missing_dates, 'transactionmonth'], errors='coerce')

    # Vehicle age
    if 'registrationyear' in df.columns:
        reg_year = pd.to_numeric(df['registrationyear'], errors='coerce')
        new_cols['vehicle_age'] = current_year - reg_year
        new_cols['vehicle_age_at_transaction'] = new_cols['transaction_date'].apply(lambda x: x.year if pd.notnull(x) else np.nan) - reg_year
        new_cols['vehicle_age_at_transaction'] = new_cols['vehicle_age_at_transaction'].fillna(new_cols['vehicle_age'])
    else:
        new_cols['vehicle_age'] = 0
        new_cols['vehicle_age_at_transaction'] = 0

    # Vehicle ratios
    if 'capitaloutstanding' in df.columns and 'customvalueestimate' in df.columns:
        new_cols['vehicle_value_ratio'] = (df['capitaloutstanding'] / df['customvalueestimate'].replace(0, np.nan)).fillna(0)
    else:
        new_cols['vehicle_value_ratio'] = 0

    if 'kilowatts' in df.columns and 'cubiccapacity' in df.columns:
        new_cols['engine_power_ratio'] = (df['kilowatts'] / df['cubiccapacity'].replace(0, np.nan)).fillna(0)
        new_cols['is_high_power_vehicle'] = (df['kilowatts'] > 150).astype(int)
    else:
        new_cols['engine_power_ratio'] = 0
        new_cols['is_high_power_vehicle'] = 0

    # Policy features
    if 'suminsured' in df.columns and 'customvalueestimate' in df.columns:
        new_cols['suminsured_ratio'] = (df['suminsured'] / df['customvalueestimate'].replace(0, np.nan)).fillna(0)
    else:
        new_cols['suminsured_ratio'] = 0

    if 'calculatedpremiumperterm' in df.columns and 'suminsured' in df.columns:
        new_cols['premium_per_suminsured'] = (df['calculatedpremiumperterm'] / df['suminsured'].replace(0, np.nan)).fillna(0)
    else:
        new_cols['premium_per_suminsured'] = 0

    # Term frequency encoding
    if 'termfrequency' in df.columns:
        new_cols['term_frequency_encoded'] = df['termfrequency'].map({'monthly':12,'quarterly':4,'half-yearly':2,'annually':1}).fillna(1)
    else:
        new_cols['term_frequency_encoded'] = 1

    # Boolean flags
    boolean_cols = {
        'alarmimmobiliser':'has_alarm',
        'trackingdevice':'has_tracking',
        'writtenoff':'written_off_flag',
        'rebuilt':'rebuilt_flag',
        'newvehicle':'new_vehicle_flag',
        'isvatregistered':'vat_registered_flag'
    }
    for orig, new in boolean_cols.items():
        if orig in df.columns:
            new_cols[new] = df[orig].map({'yes':1,'no':0}).fillna(0)
        else:
            new_cols[new] = 0

    # Transaction features safely
    new_cols['transaction_year'] = new_cols['transaction_date'].apply(lambda x: x.year if pd.notnull(x) else 0).astype(int)
    new_cols['transaction_month_num'] = new_cols['transaction_date'].apply(lambda x: x.month if pd.notnull(x) else 0).astype(int)
    if 'transactionmonth' in df.columns:
        tm_year = pd.to_datetime(df['transactionmonth'], errors='coerce').apply(lambda x: x.year if pd.notnull(x) else 0)
        new_cols['policy_age'] = new_cols['transaction_year'] - tm_year
    else:
        new_cols['policy_age'] = 0

    # Add new columns to dataframe
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # Drop near-zero columns that are mostly zeros
    zero_cols = ['vehicle_value_ratio', 'engine_power_ratio', 'is_high_power_vehicle',
                 'suminsured_ratio', 'premium_per_suminsured', 'term_frequency_encoded',
                 'has_alarm', 'has_tracking', 'written_off_flag', 'rebuilt_flag', 'new_vehicle_flag', 'vat_registered_flag']
    for col in zero_cols:
        if col in df.columns and df[col].sum() == 0:
            logging.info(f"Dropping column {col} because all values are zero")
            df.drop(columns=[col], inplace=True)

    # Ensure object columns are lowercase and stripped
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()

    logging.info("Feature engineering completed.")
    return df

# ----------------------
# Save data
# ----------------------
def save_data(df: pd.DataFrame, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logging.info(f"Full cleaned data saved to {file_path}")

# ----------------------
# Main
# ----------------------
def run_preprocessing():
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = feature_engineering(df)
    save_data(df, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    run_preprocessing()
