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
    "DRIVE_SAVE_PATH": "/content/drive/MyDrive/Airbnb_Price_Project/artifacts/",
    "TEXT_MODEL_NAME": 'BAAI/bge-small-en-v1.5',
    "VAL_SIZE": 0.05, #5% validation split
    "SEED": 42,

    # --- Model Training ---
    "BATCH_SIZE": 256,
    "VALIDATION_BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "TRANSFORMER_LEARNING_RATE": 1e-5, # Separate, lower LR for fine-tuning
    "N_EPOCHS": 100,
    "WEIGHT_DECAY": 1e-4, # L2 Regularization for non-transformer params

    # --- Model Architecture ---
    "DROPOUT_RATE": 0.2,
    "GEO_EMBEDDING_DIM": 32,
    "HIDDEN_LAYERS_LOCATION": [32, 16],
    "HIDDEN_LAYERS_SIZE_CAPACITY": [32, 16],
    "HIDDEN_LAYERS_QUALITY": [32, 16],
    "HIDDEN_LAYERS_AMENITIES": [64, 32],
    "HIDDEN_LAYERS_DESCRIPTION": [64, 32],
    "HIDDEN_LAYERS_SEASONALITY": [16],

    # --- Early Stopping & Scheduler ---
    "EARLY_STOPPING_PATIENCE": 5,
    "EARLY_STOPPING_MIN_DELTA": 0.0, 
    "SCHEDULER_PATIENCE": 2,
    "SCHEDULER_FACTOR": 0.5,
}