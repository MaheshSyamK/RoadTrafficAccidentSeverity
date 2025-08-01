import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

def plot_class_distribution(y):
    """Plot class distribution."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.savefig("reports/figures/class_distribution.png")
    plt.close()

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png")
    plt.close()