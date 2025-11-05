***Cell 1: Markdown***
````markdown
# 3. Final Additive Explainable Model

### Objective

This notebook details the training and finalization of our primary model: the `AdditiveModel`. This model is the core of our project, designed specifically for **explainability**.

Its unique architecture predicts an Airbnb's price not as a single opaque number, but as a sum of contributions from six distinct feature axes: Location, Size & Capacity, Quality, Amenities, Description, and Seasonality. This allows us to understand *why* a listing is priced the way it is.

After training, we will use this model to generate the final, enriched dataset that will power our interactive Streamlit application.
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
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# --- Imports from Custom Project Scripts ---
print("\n--- Importing Custom Modules ---")
from config import config
from data_processing import load_and_split_data, FeatureProcessor, create_dataloaders
from model import AdditiveModel # Import the final AdditiveModel
from train import train_model, evaluate_model
from build_app_dataset import build_dataset # Import the final dataset builder

print("\nSetup and imports complete.")
````

---
***Cell 3: Markdown***
````markdown
## Data Loading and Preprocessing

As with our baseline model, we begin by loading the data and applying our stratified group split to create reliable training and validation sets. We then fit our `FeatureProcessor` on the training data to learn all necessary transformations, ensuring a consistent and leak-free data pipeline.
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

We instantiate our final `AdditiveModel`. A key aspect of this model is its use of a pre-trained `SentenceTransformer` for understanding text features. To improve its performance on our specific dataset, we "fine-tune" it.

This is achieved by using **differential learning rates**:
- The newly added MLP layers of our model train with a standard learning rate (e.g., `1e-3`).
- The pre-trained transformer layers are trained with a much smaller learning rate (e.g., `1e-5`).

This allows the transformer to adapt to the nuances of Airbnb descriptions without catastrophically forgetting the general language knowledge it already possesses.
````

---
***Cell 6: Code***
````python
# ==============================================================================
# CELL 6: INSTANTIATE MODEL AND OPTIMIZER
# ==============================================================================

# Instantiate the AdditiveModel
model = AdditiveModel(processor, config)
model.to(config['DEVICE'])

# --- Create parameter groups for differential learning rates ---
# Parameters from the pre-trained text transformer
transformer_params = model.text_transformer.parameters()
# All other parameters in our model (MLPs, embeddings)
other_params = [p for n, p in model.named_parameters() if 'text_transformer' not in n]

# Instantiate the optimizer with two parameter groups
optimizer = optim.AdamW([
    {'params': other_params, 'lr': config['LEARNING_RATE'], 'weight_decay': config['WEIGHT_DECAY']},
    {'params': transformer_params, 'lr': config['TRANSFORMER_LEARNING_RATE']}
])

# Instantiate the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config['SCHEDULER_FACTOR'],
    patience=config['SCHEDULER_PATIENCE']
)

print(f"AdditiveModel and optimizer with differential learning rates have been initialized.")
````

---
***Cell 7: Markdown***
````markdown
## Model Training

We now pass all components to our reusable `train_model` function. It will handle the entire training process, including early stopping, and return the best-performing model state along with a history of the training process.
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
```markdown
## Final Evaluation

With the best model checkpoint loaded, we perform a final evaluation on the validation set to confirm its performance.
````

---
***Cell 10: Code***
````python
# ==============================================================================
# CELL 10: RUN FINAL EVALUATION
# ==============================================================================

print("\n--- Final Model Evaluation ---")
_, final_val_mape = evaluate_model(trained_model, val_loader, config['DEVICE'])
final_val_rmse = np.sqrt(evaluate_model(trained_model, val_loader, config['DEVICE'])[0])

final_metrics = {
    "val_rmse": final_val_rmse,
    "val_mape": final_val_mape
}

print("\n" + "="*50)
print(f"{'Final Additive Model Performance Metrics':^50}")
print("="*50)
print(f"Validation RMSE: {final_metrics['val_rmse']:.4f}")
print(f"Validation MAPE: {final_metrics['val_mape'] * 100:.2f}%")
print("=" * 50)
````

---
***Cell 11: Markdown***
````markdown
## Building the Final Application Dataset

This is the final "production" step of our modeling pipeline. With our trained `AdditiveModel`, we now execute the `build_dataset` function from our `build_app_dataset.py` script.

This function performs several computationally intensive, one-time tasks:
1.  Creates a complete "panel" of every listing for all 12 months.
2.  Runs inference on this entire panel using our model.
3.  Captures not only the final `predicted_price` but also all the explainable components (`p_*` for log-contributions, `pm_*` for multiplicative factors) and the hidden state vectors (`h_*`) needed for the similarity search.
4.  Saves this single, enriched, and self-contained DataFrame to a Parquet file, which is the sole data artifact required by our Streamlit application.
````

---
***Cell 12: Code***
````python
# ==============================================================================
# CELL 12: EXECUTE THE BUILD PROCESS
# ==============================================================================

print("--- Starting build process for the application dataset ---")

# Pass all necessary objects to the build function
build_dataset(
    model=trained_model,
    processor=processor,
    config=config,
    train_ids=train_ids,
    val_ids=val_ids
)
````

---
***Cell 13: Markdown***
````markdown
## Save Core Artifacts

While the main data artifact has already been created by the build script, we also save the core modeling components for reproducibility and potential future analysis. This includes the trained model's state dictionary and the fitted `FeatureProcessor`.
````

---
***Cell 14: Code***
````python
# ==============================================================================
# CELL 14: SAVE MODELING ARTIFACTS
# ==============================================================================
import pickle

print("--- Saving core model artifacts ---")

# 1. Define paths
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
artifacts_dir = os.path.join(config['DRIVE_SAVE_PATH'], f"additive_{timestamp}")
os.makedirs(artifacts_dir, exist_ok=True)

model_save_path = os.path.join(artifacts_dir, "additive_model.pt")
processor_save_path = os.path.join(artifacts_dir, "feature_processor.pkl")

# 2. Save the model state
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'final_metrics': final_metrics
}, model_save_path)
print(f"Model and metrics saved to: {model_save_path}")

# 3. Save the feature processor
with open(processor_save_path, 'wb') as f:
    pickle.dump(processor, f)
print(f"Feature processor saved to: {processor_save_path}")

print("\nAll additive model artifacts have been saved successfully.")
````