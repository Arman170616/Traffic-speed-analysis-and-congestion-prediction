# src/preprocess.py
import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the METR-LA dataset."""
    df = pd.read_csv(filepath)
    df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

def clean_data(df):
    """Replace zero values with NaN and drop rows with all NaNs."""
    df.replace(0, np.nan, inplace=True)
    df.dropna(how='all', inplace=True)
    return df

def resample_data(df, frequency='H'):
    """Resample data to a specified frequency, e.g., hourly."""
    return df.resample(frequency).mean()
