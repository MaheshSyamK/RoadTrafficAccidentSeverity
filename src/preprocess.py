import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os
import time
import psutil

def load_and_sample_data(input_path, chunk_size=10000, sample_frac=1.0):
    """Load dataset in chunks with optional sampling."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}. Please download it to data/raw/RTA Dataset.csv")
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    chunks = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)
    sampled_chunks = []
    for chunk in chunks:
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=0)
        sampled_chunks.append(chunk)
    df = pd.concat(sampled_chunks)
    
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")
    print(f"Memory usage: {process.memory_info()[0]/(1024*1024):.2f} MB")
    return df

def preprocess_data(df):
    """Preprocess data: handle missing values, encode categoricals, apply SMOTE."""
    # Handle missing values
    df = df.replace(['na', 'Unknown'], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with median
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Impute categorical columns with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Feature engineering: extract hour from Time
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
        df = df.drop('Time', axis=1)
    
    # Encode categorical variables and save encoders
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Split features and target
    X = df.drop('Accident_severity', axis=1)
    y = df['Accident_severity']
    
    # Apply SMOTE
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled, encoders

if __name__ == "__main__":
    raw_path = "data/raw/RTA Dataset.csv"
    processed_dir = "data/processed"
    models_dir = "models"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    df = load_and_sample_data(raw_path, sample_frac=0.1)
    
    # Preprocess data
    X, y, encoders = preprocess_data(df)
    
    # Save preprocessed data and encoders
    pd.DataFrame(X).to_parquet(f"{processed_dir}/X_preprocessed.parquet", engine='pyarrow')
    pd.DataFrame(y, columns=['Accident_severity']).to_parquet(f"{processed_dir}/y_preprocessed.parquet", engine='pyarrow')
    joblib.dump(encoders, f"{models_dir}/encoders.pkl")
    print("Preprocessing complete. Data saved in data/processed/, encoders saved in models/.")