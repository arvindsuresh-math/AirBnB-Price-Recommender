***Cell 1: Markdown***
````markdown
# 2. Additive Model Ablation Study

### Objective

To quantify the importance and contribution of each specialized sub-network in our `AdditiveModel`, we conduct a comprehensive ablation study.

An ablation study involves systematically removing one component of the model at a time, retraining the model from scratch, and measuring the impact on its performance. A significant drop in performance (i.e., an increase in RMSE or MAPE) when an axis is removed indicates that the axis is highly important for the model's predictive accuracy.

This process allows us to validate our architectural choices and understand which features are the most critical drivers of price in our dataset.
````

---
***Cell 2: Code***
````python
# ==============================================================================
# CELL 2: SETUP AND IMPORTS
# ==============================================================================

# --- Environment Setup (for Google Colab) ---
from google.colab import drive, userdata
from huggingface_hub import login
import os

print("--- Setting up Environment ---")
drive.mount('/content/drive')

# IMPORTANT: Make sure this path matches your project folder
PROJECT_PATH = '/content/drive/MyDrive/Airbnb_Price_Project'
os.chdir(PROJECT_PATH)
print(f"Current working directory: {os.getcwd()}")

# --- Standard and Third-Party Library Imports ---
import pandas as pd

# --- Imports from Custom Project Scripts ---
print("\n--- Importing Custom Modules ---")
from config import config
from data_processing import load_and_split_data, FeatureProcessor, create_dataloaders
from model import AblationAdditiveModel # Specifically import the ablation model
from train import run_ablation_experiment # Import the high-level experiment runner

print("\nSetup and imports complete.")
````

---
***Cell 3: Markdown***
````markdown
## Data Loading and Preparation

For a fair and controlled experiment, the data pipeline must remain identical for every ablation run. We prepare the `train_loader` and `val_loader` once at the beginning. These exact same data loaders will then be passed to each training run, ensuring that the only variable changing between experiments is the model's architecture (i.e., which axis is being excluded).
````

---
***Cell 4: Code***
````python
# ==============================================================================
# CELL 4: EXECUTE DATA PIPELINE
# ==============================================================================

# Load and split the data
train_df, val_df, neighborhood_log_means, _, _ = load_and_split_data(config)

# Instantiate and fit the feature processor
processor = FeatureProcessor(config)
processor.fit(train_df)

# Transform datasets into feature dictionaries
train_features = processor.transform(train_df, neighborhood_log_means)
val_features = processor.transform(val_df, neighborhood_log_means)

# Create the reusable PyTorch DataLoaders
train_loader, val_loader = create_dataloaders(train_features, val_features, config)

print("\nData pipeline complete. DataLoaders will be reused for all experiments.")
````

---
***Cell 5: Markdown***
````markdown
## Running the Ablation Experiments

We now proceed with the core of the study. The process is as follows:

1.  **Establish a Baseline:** We first run an experiment with **no axes excluded**. This trains the full `AdditiveModel` and gives us the baseline performance metric against which all other runs will be compared.
2.  **Iterate and Ablate:** We define a list of all six model axes. We then loop through this list, and for each axis, we execute our `run_ablation_experiment` function. This powerful wrapper, imported from `train.py`, handles the entire workflow for a single run:
    - Instantiates the `AblationAdditiveModel`, telling it which axis to exclude.
    - Sets up the optimizer and scheduler.
    - Runs the full training loop with early stopping.
    - Evaluates the final model and returns a dictionary of performance metrics.
3.  **Collect Results:** The metrics from each run are collected in a list.
````

---
***Cell 6: Code***
````python
# ==============================================================================
# CELL 6: EXPERIMENT LOOP
# ==============================================================================

# List to store the results from each experimental run
ablation_results = []

# --- Run 0: Establish the baseline with the full model ---
baseline_metrics = run_ablation_experiment(
    exclude_axes=[], # An empty list means all axes are included
    config=config,
    processor=processor,
    train_loader=train_loader,
    val_loader=val_loader
)
ablation_results.append(baseline_metrics)

# --- Define the axes to remove one by one ---
axes_to_ablate = [
    'location',
    'size_capacity',
    'quality',
    'amenities',
    'description',
    'seasonality'
]

# --- Loop through each axis and run an experiment ---
for axis in axes_to_ablate:
    experiment_metrics = run_ablation_experiment(
        exclude_axes=[axis], # Exclude the current axis
        config=config,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader
    )
    ablation_results.append(experiment_metrics)

print("\n\nAll ablation experiments have been completed.")
````

---
***Cell 7: Markdown***
````markdown
## Compiling and Saving Results

With all experiments complete, we compile the collected list of metric dictionaries into a single, clean pandas DataFrame. This table provides a clear overview of the study's findings.

We display the summary directly in the notebook and then save the DataFrame to a timestamped CSV file in our artifacts directory. This file will be loaded by our `04_results_and_analysis.ipynb` notebook to generate visualizations and draw final conclusions.
````

---
***Cell 8: Code***
````python
# ==============================================================================
# CELL 8: COMPILE, DISPLAY, AND SAVE RESULTS
# ==============================================================================

# Convert the list of dictionaries into a DataFrame
results_df = pd.DataFrame(ablation_results)

# Format MAPE columns into percentages for easier reading
results_df['train_mape_pct'] = results_df['train_mape'] * 100
results_df['val_mape_pct'] = results_df['val_mape'] * 100

# Define the columns to display and their order
display_cols = ['excluded_axes', 'train_rmse', 'val_rmse', 'train_mape_pct', 'val_mape_pct']

# --- Final Summary of Results ---
print("\n\n" + "="*80)
print(f"{'ABLATION STUDY SUMMARY':^80}")
print("="*80)
print(results_df[display_cols].to_string(index=False, float_format="%.4f"))
print("="*80)

# --- Save the results DataFrame to a CSV file ---
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
filename = f"{config['CITY']}_ablation_results_{timestamp}.csv"
save_path = os.path.join(config['DRIVE_SAVE_PATH'], filename)

# Ensure the target directory exists
os.makedirs(config['DRIVE_SAVE_PATH'], exist_ok=True)

# Save the complete DataFrame
results_df.to_csv(save_path, index=False, float_format="%.6f")

print(f"\nAblation study results successfully saved to:\n{save_path}")
````