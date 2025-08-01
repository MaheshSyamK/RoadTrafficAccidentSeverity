import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import joblib
import os
import time
import psutil

def train_models(X, y):
    """Train models with hyperparameter tuning."""
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    models = {
        'dt': DecisionTreeClassifier(random_state=0),
        'rf': RandomForestClassifier(random_state=0),
        'et': ExtraTreesClassifier(random_state=0),
        'xgb': XGBClassifier(random_state=0, tree_method='hist')
    }
    
    param_grids = {
        'dt': {'max_depth': [5, 10, 15]},
        'rf': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
        'et': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
        'xgb': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
    }
    
    os.makedirs("models", exist_ok=True)
    best_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best params for {name}: {grid_search.best_params_}")
        joblib.dump(best_models[name], f"models/{name}_model.pkl")
    
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    print(f"Memory usage: {process.memory_info()[0]/(1024*1024):.2f} MB")
    return best_models, X_test, y_test

if __name__ == "__main__":
    # Check if preprocessed files exist
    x_path = "data/processed/X_preprocessed.parquet"
    y_path = "data/processed/y_preprocessed.parquet"
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError("Preprocessed files not found. Run preprocess.py first.")
    
    # Load preprocessed data
    X = pd.read_parquet(x_path, engine='pyarrow')
    y = pd.read_parquet(y_path, engine='pyarrow')['Accident_severity']
    
    # Train models
    best_models, X_test, y_test = train_models(X, y)
    print("Training complete. Models saved in models/.")