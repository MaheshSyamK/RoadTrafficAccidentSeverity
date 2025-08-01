import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for styling
st.markdown("""
<style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stSelectbox {margin-bottom: 10px;}
    .stSlider {margin-bottom: 10px;}
    .sidebar .sidebar-content {background-color: #e6f3ff;}
    h1, h2, h3 {color: #2c3e50;}
    .stSuccess {background-color: #d4edda; color: #155724;}
</style>
""", unsafe_allow_html=True)

# Model paths
MODEL_PATHS = {
    "XGBoost": "models/xgb_model.pkl",
    "Random Forest": "models/rf_model.pkl",
    "Extra Trees": "models/et_model.pkl",
    "Decision Tree": "models/dt_model.pkl"
}

# Load encoders
@st.cache_resource
def load_encoders():
    encoder_path = "models/encoders.pkl"
    if not os.path.exists(encoder_path):
        st.error(f"Encoder file not found at {encoder_path}. Run preprocess.py first.")
        st.stop()
    return joblib.load(encoder_path)

# Load selected model
@st.cache_resource
def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if not os.path.exists(path):
        st.error(f"Model file for {model_name} not found at {path}. Run train.py first.")
        st.stop()
    return joblib.load(path)

# UI setup
st.set_page_config(page_title="🚦 Accident Severity Predictor", layout="wide")
st.markdown("<h1 style='text-align:center;'>🚦 Road Traffic Accident Severity Prediction</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Predict road accident severity using machine learning models trained on the RTA dataset.")
    st.header("📊 Model Performance")
    performance = {
        "Decision Tree": {"Accuracy": 0.87, "F1-Score": 0.87},
        "Random Forest": {"Accuracy": 0.95, "F1-Score": 0.95},
        "Extra Trees": {"Accuracy": 0.96, "F1-Score": 0.96},
        "XGBoost": {"Accuracy": 0.93, "F1-Score": 0.93}
    }
    st.write("**Performance Metrics** (from evaluate.py):")
    for model, metrics in performance.items():
        st.write(f"**{model}**: Accuracy={metrics['Accuracy']:.2f}, F1-Score={metrics['F1-Score']:.2f}")

# Main content
st.write("Select a model and input accident details to predict severity (Slight, Serious, Fatal).")

# Model selection
model_choice = st.selectbox("🔍 Choose Model", list(MODEL_PATHS.keys()), help="Select the machine learning model to use for prediction.")
model = load_model(model_choice)
encoders = load_encoders()

# Input form
with st.form("prediction_form"):
    st.subheader("🚗 Input Accident Details")
    col1, col2, col3 = st.columns(3)
    
    inputs = {}
    with col1:
        inputs['Day_of_week'] = st.selectbox("🗓️ Day of Week", encoders['Day_of_week'].classes_)
        inputs['Age_band_of_driver'] = st.selectbox("👤 Driver Age Band", encoders['Age_band_of_driver'].classes_)
        inputs['Sex_of_driver'] = st.selectbox("🚻 Sex of Driver", encoders['Sex_of_driver'].classes_)
        inputs['Educational_level'] = st.selectbox("🎓 Educational Level", encoders['Educational_level'].classes_)
        inputs['Vehicle_driver_relation'] = st.selectbox("🚘 Driver Relation", encoders['Vehicle_driver_relation'].classes_)
        inputs['Driving_experience'] = st.selectbox("🛣️ Driving Experience", encoders['Driving_experience'].classes_)
        inputs['Type_of_vehicle'] = st.selectbox("🚜 Vehicle Type", encoders['Type_of_vehicle'].classes_)
        inputs['Owner_of_vehicle'] = st.selectbox("🏢 Vehicle Owner", encoders['Owner_of_vehicle'].classes_)
        inputs['Service_year_of_vehicle'] = st.selectbox("📅 Vehicle Service Year", encoders['Service_year_of_vehicle'].classes_)
        inputs['Defect_of_vehicle'] = st.selectbox("🔧 Vehicle Defect", encoders['Defect_of_vehicle'].classes_)
        
    with col2:
        inputs['Area_accident_occured'] = st.selectbox("🌍 Accident Area", encoders['Area_accident_occured'].classes_)
        inputs['Lanes_or_Medians'] = st.selectbox("🛤️ Lanes/Medians", encoders['Lanes_or_Medians'].classes_)
        inputs['Road_allignment'] = st.selectbox("📏 Road Alignment", encoders['Road_allignment'].classes_)
        inputs['Types_of_Junction'] = st.selectbox("⛔ Junction Type", encoders['Types_of_Junction'].classes_)
        inputs['Road_surface_type'] = st.selectbox("🛢️ Road Surface Type", encoders['Road_surface_type'].classes_)
        inputs['Road_surface_conditions'] = st.selectbox("☔ Surface Conditions", encoders['Road_surface_conditions'].classes_)
        inputs['Light_conditions'] = st.selectbox("💡 Light Conditions", encoders['Light_conditions'].classes_)
        inputs['Weather_conditions'] = st.selectbox("🌦️ Weather Conditions", encoders['Weather_conditions'].classes_)
        inputs['Type_of_collision'] = st.selectbox("💥 Collision Type", encoders['Type_of_collision'].classes_)
        inputs['Number_of_vehicles_involved'] = st.number_input("🚗 Vehicles Involved", min_value=1, max_value=10, value=2)
        
    with col3:
        inputs['Number_of_casualties'] = st.number_input("🩺 Casualties", min_value=1, max_value=10, value=1)
        inputs['Vehicle_movement'] = st.selectbox("🚦 Vehicle Movement", encoders['Vehicle_movement'].classes_)
        inputs['Casualty_class'] = st.selectbox("👥 Casualty Class", encoders['Casualty_class'].classes_)
        inputs['Sex_of_casualty'] = st.selectbox("🚻 Casualty Sex", encoders['Sex_of_casualty'].classes_)
        inputs['Age_band_of_casualty'] = st.selectbox("👤 Casualty Age Band", encoders['Age_band_of_casualty'].classes_)
        inputs['Casualty_severity'] = st.selectbox("⚕️ Casualty Severity", encoders['Casualty_severity'].classes_)
        inputs['Work_of_casuality'] = st.selectbox("💼 Casualty Work", encoders['Work_of_casuality'].classes_)
        inputs['Fitness_of_casuality'] = st.selectbox("🏋️ Casualty Fitness", encoders['Fitness_of_casuality'].classes_)
        inputs['Pedestrian_movement'] = st.selectbox("🚶 Pedestrian Movement", encoders['Pedestrian_movement'].classes_)
        inputs['Cause_of_accident'] = st.selectbox("⚠️ Cause of Accident", encoders['Cause_of_accident'].classes_)
        inputs['Hour'] = st.slider("⏰ Hour of Day", 0, 23, 12)
    
    submitted = st.form_submit_button("🚀 Predict Severity", use_container_width=True)
    
    if submitted:
        with st.spinner("🔄 Predicting..."):
            # Prepare input
            input_df = pd.DataFrame([inputs])
            for col in encoders:
                if col in input_df.columns:
                    try:
                        input_df[col] = encoders[col].transform(input_df[col])
                    except ValueError:
                        st.error(f"Invalid value for {col}. Please select a valid category.")
                        st.stop()
            
            # Add missing columns
            required_features = model.get_booster().feature_names if hasattr(model, 'get_booster') else model.feature_names_in_
            for col in required_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[required_features]
            
            # Predict
            pred = model.predict(input_df)[0]
            severity_map = {0: '🔴 Fatal', 1: '🟠 Serious', 2: '🟢 Slight'}
            st.success(f"🎯 Predicted Severity: **{severity_map[pred]}**")

# Display evaluation results
st.subheader("📈 Model Evaluation Outputs")
fig_dir = "reports/figures"
if os.path.exists(fig_dir):
    # Map model choice to filename prefix
    model_prefix = {
        "XGBoost": "xgb",
        "Random Forest": "rf",
        "Extra Trees": "et",
        "Decision Tree": "dt"
    }
    prefix = model_prefix.get(model_choice, "")
    figs = [f for f in os.listdir(fig_dir) if f.startswith(f"cm_{prefix}_model.pkl") and f.endswith(".png")]
    if figs:
        for fig in figs:
            st.image(Image.open(os.path.join(fig_dir, fig)), caption=fig, use_container_width=True)
    else:
        st.info(f"No evaluation plots found for {model_choice} in `reports/figures/`. Expected filename: cm_{prefix}_model.pkl.png")
else:
    st.info("Directory `reports/figures/` not found.")

# Feature importance
if hasattr(model, 'feature_importances_'):
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    feature_names = model.get_booster().feature_names if hasattr(model, 'get_booster') else model.feature_names_in_
    top_n = 10
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances - {model_choice}")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close()