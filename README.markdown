# Road Traffic Accident Severity Classification

## Overview

This project predicts road traffic accident severity (Slight, Serious, Fatal) using machine learning models on the [Road Traffic Accidents dataset](https://www.kaggle.com/saurabhshahane/road-traffic-accidents) from Addis Ababa (2017–2020). It builds on [sugatagh/Road-Traffic-Accident-Severity-Classification](https://github.com/sugatagh/Road-Traffic-Accident-Severity-Classification) with a modular, optimized codebase featuring:

- **Exploratory Data Analysis (EDA)**: Uncovers patterns in accident data.
- **Preprocessing**: Handles missing values, encodes categoricals, and applies SMOTE.
- **Model Training**: Includes Decision Tree, Random Forest, Extra Trees, and XGBoost with hyperparameter tuning.
- **Streamlit App**: Offers interactive predictions with all 31 features, model selection, metrics, and visualizations.
- **Optimizations**: Uses chunked processing, sampling, and parquet storage for efficiency.

## Dataset

- **Source**: [RTA Dataset.csv](https://raw.githubusercontent.com/sugatagh/Road-Traffic-Accident-Severity-Classification/main/Dataset/RTA%20Dataset.csv)
- **Size**: 12,316 accidents, 32 features
- **Target**: `Accident_severity` (Slight, Serious, Fatal)
- **Features**: 31 predictors (e.g., `Day_of_week`, `Age_band_of_driver`, `Cause_of_accident`, `Hour` from `Time`)
- **Storage**:
  - Raw: `data/raw/RTA Dataset.csv`
  - Preprocessed: `data/processed/X_preprocessed.parquet`, `data/processed/y_preprocessed.parquet`
  - Encoders: `models/encoders.pkl`

## Directory Structure

```
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
```

## Setup Instructions (Windows)

1. **Clone Repository** (if using Git):
   ```cmd
   git clone https://github.com/your-username/RoadTrafficAccidentSeverity.git
   cd RoadTrafficAccidentSeverity
   ```

2. **Create Directories**:
   ```cmd
   mkdir data\raw data\processed notebooks src app models reports\figures
   echo. > src\__init__.py
   echo. > data\raw\.gitkeep
   echo. > data\processed\.gitkeep
   echo. > models\.gitkeep
   echo. > reports\figures\.gitkeep
   ```

3. **Download Dataset**:
   ```cmd
   powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/sugatagh/Road-Traffic-Accident-Severity-Classification/main/Dataset/RTA%20Dataset.csv -OutFile data\raw\RTA%20Dataset.csv"
   ```

4. **Set Up Virtual Environment**:
   ```cmd
   setup_venv.bat
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Run Pipeline**:
   - Preprocess data:
     ```cmd
     python src\preprocess.py
     ```
   - Train models:
     ```cmd
     python src\train.py
     ```
   - Evaluate models:
     ```cmd
     python src\evaluate.py
     ```
   - Run Streamlit app:
     ```cmd
     streamlit run app\streamlit_app.py
     ```

6. **Explore EDA** (optional):
   ```cmd
   cd notebooks
   jupyter notebook
   ```
   Open `eda.ipynb`.

## Performance Metrics

Evaluated on August 2, 2025:

| Model          | Accuracy | F1-Score |
|----------------|----------|----------|
| Decision Tree  | 0.87     | 0.87     |
| Random Forest  | 0.95     | 0.95     |
| Extra Trees    | 0.96     | 0.96     |
| XGBoost        | 0.93     | 0.93     |

## Outputs

- **Models**: Saved as `xgb_model.pkl`, `rf_model.pkl`, `et_model.pkl`, `dt_model.pkl` in `models/`.
- **Encoders**: Saved as `encoders.pkl` in `models/`.
- **Plots**: Confusion matrices saved as `cm_<model>_model.pkl.png` in `reports/figures/`.
- **Streamlit App**: Features 31 input fields, model selection, performance metrics, feature importance, and confusion matrix plots.

## Optimizations

- **Chunked Processing**: Loads data in 10,000-row chunks.
- **Sampling**: Uses 10% of data (`sample_frac=0.1`) in `preprocess.py`. Set to `0.05` for low-memory systems.
- **Parquet Storage**: Uses `pyarrow` for efficient data handling.
- **Memory Monitoring**: Tracks usage with `psutil` (~164–238 MB).

## Troubleshooting

- **No Plots in Streamlit**:
  - Check `reports/figures/` for `cm_xgb_model.pkl.png`, `cm_rf_model.pkl.png`, etc.
  - Re-run `python src\evaluate.py`.
  - Verify Streamlit output for "Available files in reports/figures/".
- **Encoder Issues**:
  - Ensure `models/encoders.pkl` exists after `preprocess.py`.
  - Re-run `python src\preprocess.py` if missing.
- **Memory Issues**:
  - Set `sample_frac=0.05` in `preprocess.py`.
  - Check memory usage in script output.
- **Module Errors**:
  - Activate virtual environment: `venv\Scripts\activate`.
  - Run from `E:\RoadTrafficAccidentSeverity`.
  - Reinstall dependencies: `pip install -r requirements.txt`.

## Dependencies

Key packages (see `requirements.txt`):
- `pandas>=1.5.0`
- `numpy>=1.23.0`
- `scikit-learn>=1.2.0`
- `xgboost>=1.7.0`
- `matplotlib>=3.6.0`
- `seaborn>=0.12.0`
- `plotly>=5.10.0`
- `imbalanced-learn>=0.10.0`
- `joblib>=1.2.0`
- `streamlit>=1.20.0`
- `psutil>=5.9.0`
- `pyarrow>=10.0.0`
- `Pillow>=9.0.0`

## License

MIT License

## Acknowledgements

- **Dataset**: [Saurabh Shahane](https://www.kaggle.com/saurabhshahane)
- **Original Project**: [sugatagh/Road-Traffic-Accident-Severity-Classification](https://github.com/sugatagh/Road-Traffic-Accident-Severity-Classification)

## Contact

For support, email [your-email@example.com].