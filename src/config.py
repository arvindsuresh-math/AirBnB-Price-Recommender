# config.py

"""
Central configuration file for the Airbnb price prediction model.

This file contains all hyperparameters, file paths, and settings used across the
various scripts in the project. Modifying these values allows for easy
experimentation and tuning.
"""

import torch

config = {
    # --- Data and Environment ---
    "CITY": "toronto",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DRIVE_SAVE_PATH": "/content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/artifacts/",
    "TEXT_MODEL_NAME": 'BAAI/bge-small-en-v1.5',
    "VAL_SIZE": 0.05,  # Updated to 5% for the final model, split by listing_id
    "SEED": 42,

    # --- Model Training ---
    "BATCH_SIZE": 256,
    "VALIDATION_BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "TRANSFORMER_LEARNING_RATE": 1e-5, # Separate, lower LR for fine-tuning
    "N_EPOCHS": 100, # Increased epoch limit, will be controlled by early stopping

    # --- Model Architecture ---
    # The number of input features for each sub-network is hardcoded in the model
    # definition based on the feature engineering process.
    "HIDDEN_LAYERS_LOCATION": [32, 16],
    "HIDDEN_LAYERS_SIZE_CAPACITY": [32, 16],
    "HIDDEN_LAYERS_QUALITY": [32, 16],
    "HIDDEN_LAYERS_AMENITIES": [64, 32],
    "HIDDEN_LAYERS_DESCRIPTION": [64, 32],
    "HIDDEN_LAYERS_SEASONALITY": [16],
    "GEO_EMBEDDING_DIM": 32, # Dimension for the geospatial positional encoding

    # --- Early Stopping & Scheduler ---
    "EARLY_STOPPING_PATIENCE": 10,
    "EARLY_STOPPING_MIN_DELTA": 0.001, # 0.1% change in validation MAPE
    "SCHEDULER_PATIENCE": 2,
    "SCHEDULER_FACTOR": 0.5,
}