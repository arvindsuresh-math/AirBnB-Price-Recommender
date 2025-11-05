***Cell 1: Markdown***
````markdown
# 4. Model Comparison and Results Analysis

### Objective

This notebook is the final step in our analysis. Here, we bring together the results from our three distinct modeling approaches to perform a comprehensive comparison:
1.  **Random Forest:** A classic, powerful tree-based model.
2.  **Baseline Neural Network:** A robust, fully-connected deep learning model.
3.  **Additive Neural Network:** Our final, interpretable deep learning model.

We will compare their overall predictive performance, investigate what features each model found important, and, most critically, perform a deep dive into the **explainability** of the `AdditiveModel` to demonstrate its unique value for a price recommendation tool.
````

---
***Cell 2: Code***
````python
# ==============================================================================
# CELL 2: SETUP AND IMPORTS
# ==============================================================================

# --- Environment Setup (for Google Colab) ---
from google.colab import drive
import os

print("--- Setting up Environment ---")
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/Airbnb_Price_Project'
os.chdir(PROJECT_PATH)
print(f"Current working directory: {os.getcwd()}")

# --- Standard and Third-Party Library Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Imports from Custom Project Scripts ---
print("\n--- Importing Custom Modules ---")
from config import config
from plotting import (
    plot_predictions_vs_actual,
    plot_mape_distribution,
    plot_ablation_results,
    plot_additive_contributions
)

# --- Plotting Configuration ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

print("\nSetup and imports complete.")````

---
***Cell 3: Markdown***
````markdown
## Loading All Model Artifacts

We begin by loading all the necessary data artifacts that were generated and saved by our previous notebooks. This includes:
- The prediction DataFrames for each of the three models.
- The results DataFrame from the `AdditiveModel` ablation study.

**Note:** For this notebook to run, you must first execute the other training notebooks (`01`, `02`, `03`) and the separate Random Forest training script to generate these files.
````

---
***Cell 4: Code***
````python
# ==============================================================================
# CELL 4: LOAD PREDICTION AND RESULT FILES
# ==============================================================================

print("--- Loading model prediction artifacts ---")

# NOTE: You will need to find the correct timestamped folder for your artifacts
# For example: '/content/drive/MyDrive/Airbnb_Price_Project/artifacts/baseline_20251104_103000/'
BASELINE_ARTIFACT_PATH = "" # TODO: Fill in the path to your baseline artifacts folder
ADDITIVE_ARTIFACT_PATH = "" # TODO: Fill in the path to your additive model artifacts folder
RF_ARTIFACT_PATH = ""       # TODO: Fill in the path to your Random Forest artifacts folder
ABLATION_RESULTS_PATH = ""  # TODO: Fill in the direct path to your ablation_results.csv file

try:
    # Load predictions only for listings in the validation set for a fair comparison
    baseline_preds_df = pd.read_parquet(os.path.join(BASELINE_ARTIFACT_PATH, "baseline_model_predictions.parquet"))
    baseline_val_df = baseline_preds_df[baseline_preds_df['split'] == 'val'].dropna(subset=['price'])

    additive_preds_df = pd.read_parquet(os.path.join(ADDITIVE_ARTIFACT_PATH, "additive_model_predictions.parquet"))
    additive_val_df = additive_preds_df[additive_preds_df['split'] == 'val'].dropna(subset=['price'])
    
    # Assuming the RF model predictions file is named similarly
    rf_preds_df = pd.read_parquet(os.path.join(RF_ARTIFACT_PATH, "rf_model_predictions.parquet"))
    rf_val_df = rf_preds_df[rf_preds_df['split'] == 'val'].dropna(subset=['price'])

    ablation_df = pd.read_csv(ABLATION_RESULTS_PATH)
    
    print("All prediction and result files loaded successfully.")
    
except FileNotFoundError as e:
    print(f"ERROR: Could not find a required file. Please check your paths.")
    print(e)

models = {
    "Random Forest": rf_val_df,
    "Baseline NN": baseline_val_df,
    "Additive NN": additive_val_df
}
````

---
***Cell 5: Markdown***
````markdown
## Overall Performance Comparison

We first compare the models on their core predictive accuracy. We will use two key visualizations:

1.  **True vs. Predicted Price:** A scatter plot that shows how well the model's predictions align with the actual prices. A perfect model would have all points lying on the y=x line.
2.  **MAPE Distribution:** A histogram showing the distribution of the Mean Absolute Percentage Error for each prediction. This is more informative than a single average MAPE value, as it reveals if a model makes a few very large errors or many small ones.
````

---
***Cell 6: Code***
````python
# ==============================================================================
# CELL 6: PLOT TRUE VS. PREDICTED PRICES
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('Model Performance: True vs. Predicted Prices (Validation Set)', fontsize=20)

for ax, (model_name, df) in zip(axes, models.items()):
    plot_predictions_vs_actual(df, model_name, ax)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
````

---
***Cell 7: Code***
````python
# ==============================================================================
# CELL 7: PLOT MAPE DISTRIBUTIONS
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
fig.suptitle('Model Performance: Distribution of Prediction Errors (MAPE)', fontsize=20)

for ax, (model_name, df) in zip(axes, models.items()):
    plot_mape_distribution(df.copy(), model_name, ax) # Pass copy to avoid SettingWithCopyWarning

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
````

---
***Cell 8: Markdown***
````markdown
## Feature and Axis Importance

Next, we investigate *what* each model learned was important for predicting price.

-   **Ablation Study:** For our `AdditiveModel`, the results from the ablation study provide a powerful, direct measure of axis importance. A larger increase in error when an axis is removed implies it is more critical to the model's performance.
-   **Random Forest:** We also look at the classic Gini importance from our Random Forest model as a point of comparison from a non-neural network approach.
````

---
***Cell 9: Code***
````python
# ==============================================================================
# CELL 9: PLOT ABLATION STUDY RESULTS
# ==============================================================================

print("--- Additive Model: Ablation Study Results ---")
plot_ablation_results(ablation_df)
````

---
***Cell 10: Markdown***
````markdown
## Deep Dive: Explaining a Single Prediction

The standout feature of the `AdditiveModel` is its ability to explain its own predictions. Here, we demonstrate this by selecting a single, high-priced listing from the validation set and visualizing how the model constructed its price recommendation.

The plot below shows the **neighborhood average price** as the baseline. Each bar then represents the positive or negative price adjustment contributed by that specific feature axis, leading to the final predicted price. This level of transparency is invaluable for a real-world price recommendation tool, as it builds trust and provides actionable insights to hosts.
````

---
***Cell 11: Code***
````python
# ==============================================================================
# CELL 11: VISUALIZE ADDITIVE CONTRIBUTIONS FOR ONE LISTING
# ==============================================================================

# Select an interesting listing from the validation set (e.g., one of the most expensive)
selected_listing = additive_val_df.sort_values('price', ascending=False).iloc[0]

print(f"--- Explaining Prediction for Listing: '{selected_listing['name']}' ---")
print(f"Neighborhood: {selected_listing['neighbourhood_cleansed']}")
print(f"Actual Price: ${selected_listing['price']:.2f}")
print(f"Predicted Price: ${selected_listing['predicted_price']:.2f}\n")

# Use our plotting function to generate the breakdown
plot_additive_contributions(selected_listing)
````

---
***Cell 12: Markdown***
````markdown
## Conclusion

**(This is where you will write your final project summary based on the plots above)**

*Example conclusion:*

Across all models, we observed strong predictive performance, with the Additive Neural Network achieving a validation MAPE of XX.X%, comparable to the Baseline NN and the Random Forest.

The key differentiator is the `AdditiveModel`'s inherent **explainability**. The ablation study revealed that **Location** and **Size/Capacity** were the most critical axes for model performance, with their removal causing the largest increase in prediction error. This aligns with our domain knowledge of real estate.

Most importantly, the ability to decompose any single prediction into its constituent parts—as demonstrated in the final plot—is the primary value of this project. While all three models can answer "What is the predicted price?", only the `AdditiveModel` can effectively answer "**Why is that the predicted price?**". This makes it the superior choice for a practical price recommendation tool, as it provides hosts with transparent, actionable insights into how their listing's characteristics translate to market value.
````