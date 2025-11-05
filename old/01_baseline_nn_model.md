***Cell 1: Markdown***
````markdown
# 1. Baseline Neural Network Model

### Objective

In this notebook, we train and evaluate a fully-connected deep learning model. This model will serve as a robust performance baseline against which we can compare our more complex, interpretable `AdditiveModel`.

The architecture is a standard Multi-Layer Perceptron (MLP) that takes all available features—including location, size, quality, and text embeddings—concatenates them into a single vector, and processes them through several layers to predict the final log-price deviation. Regularization techniques like Dropout, Batch Normalization, and Weight Decay are used to prevent overfitting.
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

# IMPORTANT: Make sure this path matches your project folder in Google Drive
PROJECT_PATH = '/content/drive/MyDrive/Airbnb_Price_Project'
os.chdir(PROJECT_PATH)
print(f"Current working directory: {os.getcwd()}")

# --- Standard and Third-Party Library Imports ---
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# --- Imports from Custom Project Scripts ---
print("\n--- Importing Custom Modules ---")
from config import config
from data_processing import load_and_split_data, FeatureProcessor, create_dataloaders, AirbnbPriceDataset
from model import BaselineModel
from train import train_model, evaluate_model
from inference import run_inference

print("\nSetup and imports complete.")
````

---
***Cell 3: Markdown***
````markdown
## Data Loading and Preprocessing

We begin by loading the dataset and performing our custom stratified group split. This method ensures that all records for a single listing (`listing_id`) are confined to either the training or the validation set, which is crucial for preventing data leakage and obtaining a reliable performance estimate.

Once split, we instantiate and `fit` our `FeatureProcessor` exclusively on the training data. This learns the necessary vocabularies and scaling parameters, which are then used to `transform` both the training and validation sets into numerical tensors ready for the model.
````

---
***Cell 4: Code***
````python
# ==============================================================================
# CELL 4: EXECUTE DATA PIPELINE
# ==============================================================================

# Load and split the data
train_df, val_df, neighborhood_log_means, train_ids, val_ids = load_and_split_data(config)

# Instantiate and fit the feature processor on the training data
processor = FeatureProcessor(config)
processor.fit(train_df)

# Transform both datasets into feature dictionaries
train_features = processor.transform(train_df, neighborhood_log_means)
val_features = processor.transform(val_df, neighborhood_log_means)

# Create the PyTorch DataLoaders
train_loader, val_loader = create_dataloaders(train_features, val_features, config)

print("\nData pipeline complete. DataLoaders are ready for training.")
````

---
***Cell 5: Markdown***
````markdown
## Model Initialization

Here, we instantiate our `BaselineModel` from the `model.py` script. The architecture is defined by the parameters in our central `config` file.

We then define the `AdamW` optimizer, which is a robust choice for deep learning models. Crucially, we include a `weight_decay` parameter, which applies L2 regularization to help prevent overfitting. Finally, we set up a `ReduceLROnPlateau` scheduler, which will automatically decrease the learning rate if the validation performance stagnates.
````

---
***Cell 6: Code***
````python
# ==============================================================================
# CELL 6: INSTANTIATE MODEL AND OPTIMIZER
# ==============================================================================

# Instantiate the baseline model
model = BaselineModel(processor, config)
model.to(config['DEVICE'])

# Instantiate the optimizer with weight decay for regularization
optimizer = optim.AdamW(
    model.parameters(),
    lr=config['LEARNING_RATE'],
    weight_decay=config['WEIGHT_DECAY']
)

# Instantiate the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config['SCHEDULER_FACTOR'],
    patience=config['SCHEDULER_PATIENCE']
)

print(f"BaselineModel, AdamW optimizer, and ReduceLROnPlateau scheduler have been initialized.")
````

---
***Cell 7: Markdown***
````markdown
## Model Training

We now pass all the prepared components—the model, data loaders, optimizer, and scheduler—to our reusable `train_model` function from the `train.py` script. This function encapsulates the entire training process:
- It iterates through epochs.
- It performs forward and backward passes.
- It calculates validation metrics after each epoch.
- It implements early stopping to halt training if the validation MAPE fails to improve, preventing overfitting and saving time.
- It returns the best performing model state and a history of the training metrics.
````

---
***Cell 8: Code***
````python
# ==============================================================================
# CELL 8: RUN TRAINING
# ==============================================================================

trained_model, history_df = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config
)
````

---
***Cell 9: Markdown***
````markdown
## Final Evaluation

With the best model checkpoint from the training process automatically loaded, we perform a final, definitive evaluation on both the training and validation sets. This provides the final performance metrics (RMSE and MAPE) that we will use to compare this baseline against other models.
````

---
***Cell 10: Code***
````python
# ==============================================================================
# CELL 10: RUN FINAL EVALUATION
# ==============================================================================

print("\n--- Final Model Evaluation ---")
final_train_mse, final_train_mape = evaluate_model(trained_model, train_loader, config['DEVICE'])
final_val_mse, final_val_mape = evaluate_model(trained_model, val_loader, config['DEVICE'])

final_metrics = {
    "train_rmse": np.sqrt(final_train_mse),
    "train_mape": final_train_mape,
    "val_rmse": np.sqrt(final_val_mse),
    "val_mape": final_val_mape
}

print("\n" + "="*50)
print(f"{'Final Baseline Performance Metrics':^50}")
print("="*50)
print(f"Train RMSE:      {final_metrics['train_rmse']:.4f}")
print(f"Validation RMSE: {final_metrics['val_rmse']:.4f}")
print("-" * 50)
print(f"Train MAPE:      {final_metrics['train_mape'] * 100:.2f}%")
print(f"Validation MAPE: {final_metrics['val_mape'] * 100:.2f}%")
print("=" * 50)
````

---
***Cell 11: Markdown***
````markdown
## Generating Predictions for the Full Dataset

For our final analysis and to provide data for potential applications, we need predictions for every listing for every month of the year. We create a "full panel" dataset by taking all unique listings and creating a row for each of the 12 months. We then run our trained model on this complete panel to generate a `predicted_price` for every entry.
````

---
***Cell 12: Code***
````python
# ==============================================================================
# CELL 12: BUILD FULL PANEL AND RUN INFERENCE
# ==============================================================================
from build_app_dataset import create_full_panel_dataset # Re-use the panel creation logic
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

print("--- Preparing full panel dataset for inference ---")

# 1. Load the original raw data to get all listings
raw_df = pd.read_parquet(f"./{config['CITY']}_dataset_oct_20.parquet")
panel_df = create_full_panel_dataset(raw_df, train_ids, val_ids)

# 2. Transform features for the entire panel using the fitted processor
panel_features = processor.transform(panel_df, neighborhood_log_means)

# 3. Create a DataLoader for the panel
tokenizer = AutoTokenizer.from_pretrained(config['TEXT_MODEL_NAME'], use_fast=True)
panel_dataset = AirbnbPriceDataset(panel_features, tokenizer)
panel_loader = DataLoader(
    panel_dataset,
    batch_size=config['VALIDATION_BATCH_SIZE'],
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# 4. Run inference to get predictions
predictions_df = run_inference(trained_model, panel_loader, config['DEVICE'])

# 5. Combine the panel data with the predictions
final_predictions_df = pd.concat([panel_df, predictions_df], axis=1)

print("\nInference complete. Final predictions DataFrame created.")
display(final_predictions_df.head())
````

---
***Cell 13: Markdown***
````markdown
## Save Artifacts

Finally, we save all the essential outputs of this notebook. This includes:
- The trained model's state dictionary (`.pt` file).
- The fitted `FeatureProcessor` instance.
- A dictionary containing the final performance metrics.
- The complete DataFrame with predictions for every listing-month.

These artifacts ensure our work is reproducible and can be easily loaded into our final analysis notebook (`04_results_and_analysis.ipynb`) without needing to retrain the model.
````

---
***Cell 14: Code***
````python
# ==============================================================================
# CELL 14: SAVE ALL OUTPUTS
# ==============================================================================
import pickle

print("--- Saving all artifacts ---")

# 1. Define paths
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
artifacts_dir = os.path.join(config['DRIVE_SAVE_PATH'], f"baseline_{timestamp}")
os.makedirs(artifacts_dir, exist_ok=True)

model_save_path = os.path.join(artifacts_dir, "baseline_model.pt")
processor_save_path = os.path.join(artifacts_dir, "feature_processor.pkl")
metrics_save_path = os.path.join(artifacts_dir, "final_metrics.pkl")
predictions_save_path = os.path.join(artifacts_dir, "baseline_model_predictions.parquet")

# 2. Save the model state and metrics
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'final_metrics': final_metrics
}, model_save_path)
print(f"Model and metrics saved to: {model_save_path}")

# 3. Save the feature processor
with open(processor_save_path, 'wb') as f:
    pickle.dump(processor, f)
print(f"Feature processor saved to: {processor_save_path}")

# 4. Save the predictions DataFrame
final_predictions_df.to_parquet(predictions_save_path, index=False)
print(f"Predictions DataFrame saved to: {predictions_save_path}")

print("\nAll baseline model artifacts have been saved successfully.")
````