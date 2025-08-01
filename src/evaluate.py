import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import psutil

def evaluate_model(model_path, X_test, y_test):
    """Evaluate a model and save confusion matrix plot."""
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    print(f"\nEvaluation for {os.path.basename(model_path)}:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {os.path.basename(model_path)}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig(f"reports/figures/cm_{os.path.basename(model_path)}.png")
    plt.close()
    
    print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
    print(f"Memory usage: {process.memory_info()[0]/(1024*1024):.2f} MB")

if __name__ == "__main__":
    # Check if preprocessed files exist
    x_path = "data/processed/X_preprocessed.parquet"
    y_path = "data/processed/y_preprocessed.parquet"
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError("Preprocessed files not found. Run preprocess.py first.")
    
    # Load test data
    X_test = pd.read_parquet(x_path, engine='pyarrow').sample(frac=0.2, random_state=0)
    y_test = pd.read_parquet(y_path, engine='pyarrow')['Accident_severity'].iloc[X_test.index]
    
    # Evaluate all models
    os.makedirs("models", exist_ok=True)
    for model_file in os.listdir("models"):
        if model_file.endswith("_model.pkl"):  # Only process model files
            evaluate_model(f"models/{model_file}", X_test, y_test)
    print("Evaluation complete. Plots saved in reports/figures/.")