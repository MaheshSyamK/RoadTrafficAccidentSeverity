Road Traffic Accident Severity Classification
Overview
This project predicts the severity of road traffic accidents (Slight, Serious, Fatal) using machine learning models, based on the Road Traffic Accidents dataset from Addis Ababa (2017-2020). It replicates the work from sugatagh/Road-Traffic-Accident-Severity-Classification with a streamlined, modular codebase. The project includes exploratory data analysis (EDA), preprocessing, training of Decision Tree, Random Forest, ExtraTrees, and XGBoost models, hyperparameter tuning, and a Streamlit app for deployment. Optimizations like chunked processing and sampling address memory issues.
Dataset

Source: RTA Dataset.csv
Size: 12,316 accidents, 32 features
Target: Accident_severity (Slight, Serious, Fatal)
Features: 31 predictors (e.g., Day_of_week, Age_band_of_driver, Cause_of_accident, Hour derived from Time)
Storage: Place in data/raw/RTA Dataset.csv. Preprocessed data is saved as X_preprocessed.parquet and y_preprocessed.parquet in data/processed/. Encoders are saved as encoders.pkl in models/.

Objectives

Perform EDA to identify patterns.
Preprocess data (handle missing values, encode categoricals, apply SMOTE).
Train and tune machine learning models.
Deploy a Streamlit app with all 31 features, model selection, performance metrics, and visualizations.
Optimize for memory efficiency to prevent system crashes.

Directory Structure
RoadTrafficAccidentSeverity/
├── data/
│   ├── raw/
│   │   └── RTA Dataset.csv
│   ├── processed/
│   │   ├── X_preprocessed.parquet
│   │   ├── y_preprocessed.parquet
├── notebooks/
│   ├── rta_severity_classification.ipynb
│   ├── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
├── app/
│   ├── streamlit_app.py
├── models/
│   ├── encoders.pkl
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   ├── et_model.pkl
│   ├── dt_model.pkl
├── reports/
│   ├── figures/
│   │   ├── cm_xgb_model.pkl.png
│   │   ├── cm_rf_model.pkl.png
│   │   ├── cm_et_model.pkl.png
│   │   ├── cm_dt_model.pkl.png
├── requirements.txt
├── setup_venv.bat
├── README.md
├── .gitignore

Setup Instructions (Windows)

Clone the Repository (if using Git):
git clone https://github.com/your-username/RoadTrafficAccidentSeverity.git
cd RoadTrafficAccidentSeverity


Create Directory Structure:
mkdir data\raw data\processed notebooks src app models reports\figures
echo. > src\__init__.py
echo. > data\raw\.gitkeep
echo. > data\processed\.gitkeep
echo. > models\.gitkeep
echo. > reports\figures\.gitkeep


Download Dataset:
powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/sugatagh/Road-Traffic-Accident-Severity-Classification/main/Dataset/RTA%20Dataset.csv -OutFile data\raw\RTA%20Dataset.csv"


Set Up Virtual Environment:
setup_venv.bat
venv\Scripts\activate


Run the Pipeline:

Preprocessing (generates parquet files and encoders):python src\preprocess.py


Training (generates model files):python src\train.py


Evaluation (generates confusion matrix plots):python src\evaluate.py


EDA (optional):cd notebooks
jupyter notebook

Open eda.ipynb.


Run Streamlit App:
streamlit run app\streamlit_app.py



Optimizations

Chunked Processing: Loads data in chunks (10,000 rows).
Sampling: Uses 10% of data (sample_frac=0.1) in preprocess.py. Reduce to 0.05 if memory issues persist.
Parquet Format: Uses pyarrow for efficient data storage.
Memory Monitoring: Tracks usage with psutil (e.g., ~164–238 MB).

Troubleshooting

No evaluation plots in Streamlit:
Verify reports/figures/ contains cm_xgb_model.pkl.png, cm_rf_model.pkl.png, etc.
Re-run python src\evaluate.py to generate plots.
Check Streamlit output for "Available files in reports/figures/" to debug missing plots.


Encoder file not found:
Ensure preprocess.py was run and models/encoders.pkl exists.
Re-run python src\preprocess.py.


Memory Issues:
Edit preprocess.py and set sample_frac=0.05.
Monitor memory usage in script output.


ModuleNotFoundError:
Ensure virtual environment is activated and run scripts from E:\RoadTrafficAccidentSeverity.
Reinstall dependencies: pip install -r requirements.txt.



Results

Model Performance (from evaluate.py, 2025-08-01):
Decision Tree: Accuracy=0.87, F1-Score=0.87
Random Forest: Accuracy=0.95, F1-Score=0.95
Extra Trees: Accuracy=0.96, F1-Score=0.96
XGBoost: Accuracy=0.93, F1-Score=0.93


Plots: Confusion matrices saved in reports/figures/ (e.g., cm_xgb_model.pkl.png).
Models: Saved in models/ as .pkl files.
Encoders: Saved as encoders.pkl in models/.
Streamlit App: Includes all 31 features, model selection, performance metrics, feature importance, and confusion matrix plots.

Dependencies
See requirements.txt. Key packages:

pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
streamlit>=1.20.0
psutil>=5.9.0
pyarrow>=10.0.0
Pillow>=9.0.0

License
MIT License
Acknowledgements

Dataset: Saurabh Shahane
Original Project: sugatagh/Road-Traffic-Accident-Severity-Classification
